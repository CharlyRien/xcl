//! Title memory outside the unified-memory windows, readable by translated shaders
//! (`XCL_VIEW_MEM=1` with `XCL_HEAP_WINDOWS=1` and the 4-window unified block; off by default).
//!
//! A retail title keeps its streamed geometry (meshlet, index and vertex buffers) at
//! addresses of its "GPU view" (`kernelx::mirrors`: a reservation with no CPU access, the GPU's
//! view of a CPU reservation of the same size), outside the unified block. Translated shaders
//! reach memory through the windows (`xcl_w` = address >> 32 - the unified block's first
//! window): with 4 windows every load past them reads 0. Here windows 4..15 (heap window slots
//! `MEM_HEAP_BASE + 4..16`, readable since every heap-window program selects all 16) are
//! reserved buffers whose 64 KiB tiles are mapped and filled on demand: a draw or dispatch
//! names the ranges it can read (`ensure_for_draw`: the V#s of its descriptor tables, its root
//! addresses) and the tiles under them are copied from where the title's bytes are (`source`:
//! the page the page-mapping method put there, else the CPU view behind the GPU view) on a
//! queue of XCL's own, finished before `ensure` returns – so before the list being recorded
//! can run. A tile is checked again (a sampled signature) at most once a second. GPU writes to
//! these windows are not copied back.
//!
//! Locks: `STATE` is taken only by `ensure` and `init`, which are called with no other XCL lock
//! held (`bind::bind_abi` before it takes its own, `real::open_uma`). Under it `source` takes
//! kernelx's page-mapping, mirror and physical-page locks one at a time; nothing here takes the
//! sparse windows' state (`sparse::lock_state` checks it in debug builds: `holding`). So the
//! order is STATE -> kernelx, and no kernelx or sparse path calls back in here.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;

const TILE: usize = 0x1_0000;
const WINDOW: usize = 1 << 32;
/// first window here and one past the last (window k = addresses (base window + k) << 32)
pub const FIRST: usize = 4;
pub const END: usize = 16;
const POOL_TILES: u32 = 1024;
const STAGING_TILES: usize = 128;

struct Tile {
    sig: u64,
    checked: std::time::Instant,
}

struct State {
    dev: ID3D12Device,
    queue: ID3D12CommandQueue,
    fence: ID3D12Fence,
    value: u64,
    alloc: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    /// window FIRST + i
    windows: Vec<ID3D12Resource>,
    /// address of window 0 (the unified block's base)
    base: usize,
    pools: Vec<ID3D12Heap>,
    next: u32,
    tiles: HashMap<usize, Tile>,
    staging: ID3D12Resource,
    staging_at: *mut u8,
    cap_tiles: usize,
    /// (va, len) → when it was last looked at
    recent: HashMap<(usize, usize), std::time::Instant>,
}
unsafe impl Send for State {}

static STATE: Mutex<Option<State>> = Mutex::new(None);
static UPLOADED: AtomicU64 = AtomicU64::new(0);
static CHECKED: AtomicU64 = AtomicU64::new(0);
static NO_SOURCE: AtomicU64 = AtomicU64::new(0);
static FULL: AtomicU64 = AtomicU64::new(0);
/// tiles and pools of the state, for `stats` (which must not wait for an `ensure` in progress)
static TILES: AtomicU64 = AtomicU64::new(0);
/// the windows were made (`init`): draws look for ranges in them
static ON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static POOLS: AtomicU64 = AtomicU64::new(0);

thread_local! {
    /// this thread holds `STATE` (`holding`)
    static HOLDING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Whether this thread holds the view memory's state: the sparse windows' state must never be
/// taken under it (the lock order of the module's doc; checked by `sparse::lock_state`).
pub(super) fn holding() -> bool {
    HOLDING.with(|h| h.get())
}

/// `STATE`, marked as held by this thread until the guard goes.
struct Held(std::sync::MutexGuard<'static, Option<State>>);
impl Drop for Held {
    fn drop(&mut self) {
        HOLDING.with(|h| h.set(false));
    }
}
fn lock() -> Held {
    let g = STATE.lock().unwrap();
    HOLDING.with(|h| h.set(true));
    Held(g)
}

fn buffer_desc(bytes: u64, flags: D3D12_RESOURCE_FLAGS) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: bytes,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: flags,
        ..Default::default()
    }
}

/// `XCL_VIEW_MEM=1`: on (off by default).
pub fn enabled() -> bool {
    std::env::var("XCL_VIEW_MEM").is_ok_and(|v| v.trim() == "1")
}

/// The windows' buffers `(resource, bytes)` for descriptors `MEM_HEAP_BASE + FIRST..END` of the
/// shared heap; empty when they could not be made or `XCL_VIEW_MEM` is not 1. `base`: the unified
/// block's window 0, which must have exactly `FIRST` windows.
pub unsafe fn init(dev: &ID3D12Device, base: usize) -> Vec<(usize, usize)> {
    if !enabled() {
        return Vec::new();
    }
    let made = (|| -> windows::core::Result<State> {
        let mut windows = Vec::new();
        for _ in FIRST..END {
            let mut r: Option<ID3D12Resource> = None;
            dev.CreateReservedResource(&buffer_desc(WINDOW as u64, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS), D3D12_RESOURCE_STATE_COMMON, None, &mut r)?;
            windows.push(r.ok_or_else(windows::core::Error::empty)?);
        }
        let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() })?;
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
        let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
        let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)?;
        list.Close()?;
        let mut staging: Option<ID3D12Resource> = None;
        dev.CreateCommittedResource(
            &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, ..Default::default() },
            D3D12_HEAP_FLAG_NONE,
            &buffer_desc((STAGING_TILES * TILE) as u64, D3D12_RESOURCE_FLAG_NONE),
            D3D12_RESOURCE_STATE_GENERIC_READ,
            None,
            &mut staging,
        )?;
        let staging = staging.ok_or_else(windows::core::Error::empty)?;
        let mut p: *mut c_void = std::ptr::null_mut();
        staging.Map(0, None, Some(&mut p))?;
        let cap_mb: usize = std::env::var("XCL_VIEW_POOL_MB").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(3072);
        Ok(State {
            dev: dev.clone(),
            queue,
            fence,
            value: 0,
            alloc,
            list,
            windows,
            base,
            pools: Vec::new(),
            next: POOL_TILES,
            tiles: HashMap::new(),
            staging,
            staging_at: p as *mut u8,
            cap_tiles: cap_mb << 4,
            recent: HashMap::new(),
        })
    })();
    match made {
        Ok(s) => {
            let out = s.windows.iter().map(|w| (w.as_raw() as usize, WINDOW)).collect();
            crate::xlog_info!("D3D12.X: view memory: windows {FIRST}..{END} ({:#x}..{:#x}) are filled on demand, up to {} MB", base + FIRST * WINDOW, base + END * WINDOW, s.cap_tiles >> 4);
            *lock().0 = Some(s);
            ON.store(true, Ordering::Release);
            out
        }
        Err(e) => {
            crate::xlog_warn!("D3D12.X: view memory: not available ({e})");
            Vec::new()
        }
    }
}

/// Where the 64 KiB page at `va` has its bytes for the CPU (`None`: nowhere readable).
fn source(va: usize) -> Option<usize> {
    use crate::hle::kernelx;
    let a = match kernelx::mirrors::mapped_source(va) {
        Some(shown) => shown?,
        None => match kernelx::mirrors::source(va) {
            Some(cpu) if !kernelx::physical_page_at(va) => cpu,
            _ => va,
        },
    };
    crate::gpu::shader::readable(a, TILE).then_some(a)
}

/// A sampled signature of a page (a word every 256 bytes and its last one).
unsafe fn signature(a: usize) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let mut o = 0;
    while o < TILE {
        h = (h ^ ((a + o) as *const u64).read_unaligned()).wrapping_mul(0x0000_0100_0000_01b3).rotate_left(17);
        o += 256;
    }
    h ^ ((a + TILE - 8) as *const u64).read_unaligned()
}

/// Shaders may read `[va, va + len)`: the tiles there hold the title's bytes when this returns.
pub unsafe fn ensure(va: usize, len: usize) {
    let mut g = lock();
    let Some(s) = g.0.as_mut() else { return };
    let (lo, hi) = (s.base + FIRST * WINDOW, s.base + END * WINDOW);
    if va < lo || va >= hi || len == 0 {
        return;
    }
    let now = std::time::Instant::now();
    let fresh = std::time::Duration::from_millis(1000);
    if s.recent.get(&(va, len)).is_some_and(|t| now.duration_since(*t) < fresh) {
        return;
    }
    if s.recent.len() > 200_000 {
        s.recent.clear();
    }
    s.recent.insert((va, len), now);
    let end = va.saturating_add(len).min(hi);
    let mut t = va & !(TILE - 1);
    // (tile, source, new)
    let mut todo: Vec<(usize, usize, bool)> = Vec::new();
    while t < end {
        let tile = t;
        t += TILE;
        let known = s.tiles.get(&tile);
        if known.is_some_and(|k| now.duration_since(k.checked) < fresh) {
            continue;
        }
        let Some(src) = source(tile) else {
            // the first few say why (asked range, what the page maps show): a range past a
            // buffer's end is expected here, a mapped page that cannot be read is not
            if NO_SOURCE.fetch_add(1, Ordering::Relaxed) < 12 {
                use crate::hle::kernelx;
                crate::xlog_info!(
                    "D3D12.X: view memory: no source for tile {tile:#x} of {va:#x}+{len:#x}: page mapping {:x?}, CPU view {:x?}, physical page here {}",
                    kernelx::mirrors::mapped_source(tile),
                    kernelx::mirrors::source(tile),
                    kernelx::physical_page_at(tile)
                );
            }
            continue;
        };
        CHECKED.fetch_add(1, Ordering::Relaxed);
        let sig = signature(src);
        match s.tiles.get_mut(&tile) {
            Some(k) if k.sig == sig => k.checked = now,
            Some(k) => {
                k.sig = sig;
                k.checked = now;
                todo.push((tile, src, false));
            }
            None => {
                if s.tiles.len() >= s.cap_tiles {
                    if FULL.fetch_add(1, Ordering::Relaxed) == 0 {
                        crate::xlog_warn!("D3D12.X: view memory: the pool is full ({} MB, XCL_VIEW_POOL_MB) – further ranges read 0", s.cap_tiles >> 4);
                    }
                    continue;
                }
                s.tiles.insert(tile, Tile { sig, checked: now });
                todo.push((tile, src, true));
            }
        }
    }
    TILES.store(s.tiles.len() as u64, Ordering::Relaxed);
    if todo.is_empty() {
        return;
    }
    // new tiles: a pool tile each, mapped under the window
    for &(tile, _, new) in &todo {
        if !new {
            continue;
        }
        if s.next >= POOL_TILES {
            let desc = D3D12_HEAP_DESC {
                SizeInBytes: POOL_TILES as u64 * TILE as u64,
                Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
                Alignment: 0,
                Flags: D3D12_HEAP_FLAG_DENY_RT_DS_TEXTURES | D3D12_HEAP_FLAG_DENY_NON_RT_DS_TEXTURES,
            };
            let mut heap: Option<ID3D12Heap> = None;
            if s.dev.CreateHeap(&desc, &mut heap).is_err() || heap.is_none() {
                crate::xlog_warn!("D3D12.X: view memory: no video memory for another pool");
                s.cap_tiles = 0;
                s.tiles.remove(&tile);
                continue;
            }
            s.pools.push(heap.unwrap());
            POOLS.store(s.pools.len() as u64, Ordering::Relaxed);
            s.next = 0;
        }
        let k = (tile - s.base) / WINDOW - FIRST;
        let coord = D3D12_TILED_RESOURCE_COORDINATE { X: ((tile - s.base) % WINDOW / TILE) as u32, ..Default::default() };
        let region = D3D12_TILE_REGION_SIZE { NumTiles: 1, UseBox: false.into(), ..Default::default() };
        let (flag, offset, count) = (D3D12_TILE_RANGE_FLAG_NONE, s.next, 1u32);
        s.queue.UpdateTileMappings(&s.windows[k], 1, Some(&coord), Some(&region), s.pools.last().unwrap(), 1, Some(&flag), Some(&offset), Some(&count), D3D12_TILE_MAPPING_FLAG_NONE);
        s.next += 1;
    }
    for batch in todo.chunks(STAGING_TILES) {
        if s.alloc.Reset().is_err() || s.list.Reset(&s.alloc, None).is_err() {
            return;
        }
        for (i, &(tile, src, _)) in batch.iter().enumerate() {
            if !s.tiles.contains_key(&tile) {
                continue;
            }
            std::ptr::copy_nonoverlapping(src as *const u8, s.staging_at.add(i * TILE), TILE);
            let k = (tile - s.base) / WINDOW - FIRST;
            s.list.CopyBufferRegion(&s.windows[k], ((tile - s.base) % WINDOW) as u64, &s.staging, (i * TILE) as u64, TILE as u64);
        }
        if s.list.Close().is_err() {
            return;
        }
        let Ok(cl) = s.list.cast::<ID3D12CommandList>() else { return };
        s.queue.ExecuteCommandLists(&[Some(cl)]);
        s.value += 1;
        if s.queue.Signal(&s.fence, s.value).is_err() {
            return;
        }
        if let Err(e) = crate::gpu::fence::wait(&s.fence, s.value, &s.dev, None, 5000) {
            static N: AtomicU64 = AtomicU64::new(0);
            if N.fetch_add(1, Ordering::Relaxed) < 4 {
                crate::xlog_warn!("D3D12.X: view memory: upload not finished: {e}");
            }
        }
        UPLOADED.fetch_add(batch.len() as u64, Ordering::Relaxed);
    }
}

/// The ranges a draw with these root arguments can read outside the unified windows: its root
/// addresses (`root_bytes` each – their sizes are not known) and the buffer V#s (16 bytes at
/// every 32 of a descriptor table: address, stride << 16 in word 1, records, type in the top
/// bits of word 3) of its tables' first `slots` descriptors.
pub unsafe fn ensure_for_draw(layout: Option<&super::console_list::Layout>, args: &[u8]) {
    use super::console_list::Param;
    if !ON.load(Ordering::Acquire) {
        return;
    }
    let (base, on) = match lock().0.as_ref() {
        Some(s) => (s.base, true),
        None => (0, false),
    };
    let Some(layout) = layout.filter(|_| on) else { return };
    static ROOT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    static SLOTS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let root_bytes = *ROOT.get_or_init(|| std::env::var("XCL_VIEW_ROOT_KB").ok().and_then(|v| v.trim().parse::<usize>().ok()).unwrap_or(1024) << 10);
    let slots = *SLOTS.get_or_init(|| std::env::var("XCL_VIEW_TABLE_SLOTS").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(64));
    let word = |at: usize| args.get(at..at + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap()));
    let (lo, hi) = (base + FIRST * WINDOW, base + END * WINDOW);
    for p in &layout.params {
        match *p {
            Param::Descriptor { at } => {
                let (Some(l), Some(h)) = (word(at as usize), word(at as usize + 4)) else { continue };
                let va = ((h as usize & 0xFFFF) << 32) | l as usize;
                if va >= lo && va < hi {
                    ensure(va, root_bytes);
                }
            }
            Param::Table { at } => {
                let Some(t) = word(at as usize) else { continue };
                if t == 0 {
                    continue;
                }
                let table = base + t as usize;
                if !crate::gpu::shader::readable(table, 32 * slots) {
                    continue;
                }
                for k in 0..slots {
                    let d = ((table + 32 * k) as *const [u32; 4]).read_unaligned();
                    if let Some((va, bytes)) = buffer_range(d, lo, hi) {
                        ensure(va, bytes);
                    }
                }
            }
            Param::Constants { .. } => {}
        }
    }
}

/// The bytes a buffer V# `d` names in `[lo, hi)`: its address (48 bits), records × stride
/// (records alone with stride 0) clamped to 4 B..16 MiB; None for another type (texture
/// descriptors have type bits in word 3's top two) or an address outside.
fn buffer_range(d: [u32; 4], lo: usize, hi: usize) -> Option<(usize, usize)> {
    let va = ((d[1] as usize & 0xFFFF) << 32) | d[0] as usize;
    if va < lo || va >= hi || d[3] >> 30 != 0 {
        return None;
    }
    let stride = (d[1] >> 16) as usize & 0x3FFF;
    let bytes = if stride == 0 { d[2] as usize } else { (d[2] as usize).saturating_mul(stride) };
    Some((va, bytes.clamp(4, 16 << 20)))
}

/// One line for the activity log.
pub fn stats() -> String {
    let (tiles, pools) = (TILES.load(Ordering::Relaxed), POOLS.load(Ordering::Relaxed));
    format!(
        "view memory: {} MB in {tiles} tiles ({pools} pools), {} tile uploads, {} tile checks, {} without a source",
        tiles >> 4,
        UPLOADED.load(Ordering::Relaxed),
        CHECKED.load(Ordering::Relaxed),
        NO_SOURCE.load(Ordering::Relaxed)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The lock order's check: `holding` is true exactly while this thread holds `STATE`.
    #[test]
    fn holding_is_true_only_under_the_state_lock() {
        assert!(!holding());
        {
            let _g = lock();
            assert!(holding());
        }
        assert!(!holding());
        // another thread does not see this one's lock
        let g = lock();
        assert!(!std::thread::spawn(holding).join().unwrap());
        drop(g);
    }

    /// Without the state (`XCL_VIEW_MEM` unset, or windows not made) a draw asks for nothing.
    #[test]
    fn nothing_happens_without_the_state() {
        assert!(lock().0.is_none() || enabled());
        let layout = super::super::console_list::Layout { params: vec![super::super::console_list::Param::Descriptor { at: 0 }], size: 8, inline: 0, inline_gfx: 0 };
        unsafe {
            ensure(0xD_0000_0000, 0x1000);
            ensure_for_draw(Some(&layout), &0xD_0000_0000u64.to_le_bytes());
        }
        assert_eq!(UPLOADED.load(Ordering::Relaxed), 0);
        assert!(!holding());
    }

    /// Buffer V#s: records × stride (records alone with stride 0), clamped; other types and
    /// addresses outside the view windows are not asked for.
    #[test]
    fn buffer_vsharps_name_their_bytes_in_the_view_windows() {
        let (lo, hi) = (0x8_0000_0000usize, 0x14_0000_0000usize);
        let d = |va: usize, stride: u32, records: u32, w3: u32| [va as u32, ((va >> 32) as u32 & 0xFFFF) | (stride << 16), records, w3];
        assert_eq!(buffer_range(d(0xD_FFFF_0000, 0, 0x100, 0), lo, hi), Some((0xD_FFFF_0000, 0x100)));
        assert_eq!(buffer_range(d(0xD_FFFF_0000, 16, 0x100, 0), lo, hi), Some((0xD_FFFF_0000, 0x1000)));
        assert_eq!(buffer_range(d(0xD_FFFF_0000, 0, 0, 0), lo, hi), Some((0xD_FFFF_0000, 4)));
        assert_eq!(buffer_range(d(0xD_FFFF_0000, 0x3FFF, u32::MAX, 0), lo, hi), Some((0xD_FFFF_0000, 16 << 20)));
        assert_eq!(buffer_range(d(0xD_FFFF_0000, 0, 0x100, 2 << 30), lo, hi), None, "an image descriptor");
        assert_eq!(buffer_range(d(0x7_FFFF_0000, 0, 0x100, 0), lo, hi), None, "in the unified block");
        assert_eq!(buffer_range(d(0x14_0000_0000, 0, 0x100, 0), lo, hi), None, "past the last window");
    }

    /// A page's signature sees a change in a sampled word and in its last word.
    #[test]
    fn the_signature_samples_the_page() {
        let mut page = vec![0u8; TILE];
        let a = page.as_mut_ptr() as usize;
        let s0 = unsafe { signature(a) };
        page[256] = 1;
        let s1 = unsafe { signature(a) };
        page[TILE - 1] = 1;
        let s2 = unsafe { signature(a) };
        assert!(s0 != s1 && s1 != s2);
    }
}
