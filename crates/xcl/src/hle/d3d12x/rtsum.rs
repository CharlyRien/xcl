// LOCAL
//! LOCAL DIAGNOSTIC, not for upstream: `XCL_LOCAL_RTSUM=1`.
//!
//! Per presented frame (CPU recording order between two PresentX calls, so approximate): what
//! the title's lists drew into which targets – render-target / depth binds, clears, draws by
//! kind (translated, PC or stand-in pipeline, skipped), ExecuteIndirect by signature kind and
//! path, dispatches – and how many times XCL dropped or emptied something (the warnings that
//! are logged only once or four times). Logged at each PresentX later than
//! `XCL_LOCAL_RTSUM_AFTER_S` seconds (default 0) after the first one.
//!
//! `XCL_LOCAL_RTSUM_DUMP_DIR=<dir>`: on the frames the frame dump writes (`XCL_FRAME_DUMP`,
//! `XCL_LOCAL_FRAME_DUMP_AFTER_S`), up to `XCL_LOCAL_RTSUM_DUMP_MAX` (default 14) targets of that
//! frame are read back (subresource 0, after the title's work on the present queue) and written
//! to `<dir>` as BMP previews (each channel stretched min..max) with texel statistics in the log.
//! Nothing here changes what the title's work does: it only counts, and copies at present time.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::System::Threading::CreateEventW;

pub fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("XCL_LOCAL_RTSUM").is_some())
}

// draw kinds
pub const DRAW_TR: usize = 0;
pub const DRAW_PC: usize = 1;
pub const DRAW_SKIP: usize = 2;
pub const DRAW_EI: usize = 3;

// silent-drop counters (all calls, not only the logged ones)
pub const NULL_DEPTH_UAV: usize = 0;
pub const TEXTURE_NOT_KEPT: usize = 1;
pub const NON_UMA_BUFFER_VIEW: usize = 2;
pub const TRANSLATION_REFUSED: usize = 3;
pub const NOT_DONE: usize = 4;
pub const DISPATCH_SKIPPED: usize = 5;
pub const DISPATCH_TR: usize = 6;
pub const CONSOLE_DISPATCH: usize = 7;
pub const CONSOLE_DISPATCH_SKIPPED: usize = 8;
const DROP_NAMES: [&str; 9] = [
    "null UAV of a depth buffer",
    "placed texture not kept in step",
    "buffer view not in UMA (empty V#)",
    "pipeline translation refused",
    "console op 'not done'",
    "Dispatch skipped",
    "Dispatch translated",
    "console-list dispatch",
    "console-list dispatch skipped",
];
static TOTAL: [AtomicU64; 9] = [const { AtomicU64::new(0) }; 9];
static FRAME_DROPS: [AtomicU64; 9] = [const { AtomicU64::new(0) }; 9];

pub fn count(k: usize) {
    if on() {
        TOTAL[k].fetch_add(1, Ordering::Relaxed);
        FRAME_DROPS[k].fetch_add(1, Ordering::Relaxed);
    }
}

#[derive(Default, Clone, Copy)]
struct Stat {
    rt_binds: u64,
    ds_binds: u64,
    clears: u64,
    draws: [u64; 4],
}

#[derive(Default)]
struct Frame {
    targets: HashMap<usize, Stat>,
    /// ExecuteIndirect: (signature's last argument type, path) -> count
    indirect: HashMap<(u32, &'static str), u64>,
    /// draws with no render target nor depth bound (depth-only would show the DSV)
    draws_no_target: [u64; 4],
    /// "not done" reasons this frame
    not_done: HashMap<String, u64>,
}

static FRAME: Mutex<Option<Frame>> = Mutex::new(None);
/// list -> (render targets, depth) resources as bound
static BOUND: Mutex<Option<HashMap<usize, (Vec<usize>, usize)>>> = Mutex::new(None);
/// DSV handle -> resource (AddRef'd)
static DSVS: Mutex<Option<HashMap<usize, usize>>> = Mutex::new(None);
/// command signature -> its last argument's PC type
static SIGS: Mutex<Option<HashMap<usize, u32>>> = Mutex::new(None);

fn with_frame<R>(f: impl FnOnce(&mut Frame) -> R) -> R {
    f(FRAME.lock().unwrap().get_or_insert_with(Frame::default))
}

unsafe fn add_ref(p: usize) {
    let f: unsafe extern "system" fn(usize) -> u32 = std::mem::transmute(*(*(p as *const *const usize)).add(1));
    f(p);
}

unsafe fn release(p: usize) {
    let f: unsafe extern "system" fn(usize) -> u32 = std::mem::transmute(*(*(p as *const *const usize)).add(2));
    f(p);
}

pub unsafe fn note_dsv(handle: usize, res: *mut c_void) {
    if !on() {
        return;
    }
    let mut g = DSVS.lock().unwrap();
    let m = g.get_or_insert_with(HashMap::new);
    if !res.is_null() {
        add_ref(res as usize);
    }
    let old = if res.is_null() { m.remove(&handle) } else { m.insert(handle, res as usize) };
    if let Some(o) = old {
        release(o);
    }
}

fn dsv_resource(handle: usize) -> Option<usize> {
    DSVS.lock().unwrap().as_ref()?.get(&handle).copied()
}

pub fn note_signature(sig: usize, last: Option<u32>) {
    if on() && sig != 0 {
        SIGS.lock().unwrap().get_or_insert_with(HashMap::new).insert(sig, last.unwrap_or(u32::MAX));
    }
}

/// OMSetRenderTargets as the PC list got it
pub unsafe fn bind(list: *mut c_void, n: u32, rtvs: *const usize, single: i32, dsv: *const usize) {
    if !on() {
        return;
    }
    let mut rts = Vec::new();
    if !rtvs.is_null() {
        for k in 0..n.min(8) as usize {
            let h = if single != 0 { *rtvs + k * 32 } else { *rtvs.add(k) };
            if let Some(r) = super::present::rtv_resource(h) {
                rts.push(r);
            }
        }
    }
    let ds = dsv.as_ref().and_then(|&h| dsv_resource(h)).unwrap_or(0);
    with_frame(|f| {
        for &r in &rts {
            f.targets.entry(r).or_default().rt_binds += 1;
        }
        if ds != 0 {
            f.targets.entry(ds).or_default().ds_binds += 1;
        }
    });
    BOUND.lock().unwrap().get_or_insert_with(HashMap::new).insert(list as usize, (rts, ds));
}

pub fn clear_rtv(rtv: usize) {
    if !on() {
        return;
    }
    if let Some(r) = super::present::rtv_resource(rtv) {
        with_frame(|f| f.targets.entry(r).or_default().clears += 1);
    }
}

pub fn clear_dsv(dsv: usize) {
    if !on() {
        return;
    }
    if let Some(r) = dsv_resource(dsv) {
        with_frame(|f| f.targets.entry(r).or_default().clears += 1);
    }
}

pub fn draw(list: *mut c_void, kind: usize) {
    if !on() {
        return;
    }
    let bound = BOUND.lock().unwrap().as_ref().and_then(|m| m.get(&(list as usize)).cloned());
    with_frame(|f| match bound {
        Some((rts, ds)) if !rts.is_empty() || ds != 0 => {
            for r in rts {
                f.targets.entry(r).or_default().draws[kind] += 1;
            }
            if ds != 0 {
                f.targets.entry(ds).or_default().draws[kind] += 1;
            }
        }
        _ => f.draws_no_target[kind] += 1,
    });
}

/// An ExecuteIndirect on `list` with signature `sig` took `path`.
pub fn indirect(list: *mut c_void, sig: usize, path: &'static str) {
    if !on() {
        return;
    }
    let last = SIGS.lock().unwrap().as_ref().and_then(|m| m.get(&sig).copied()).unwrap_or(u32::MAX);
    // draws (DRAW 0, DRAW_INDEXED 1, DISPATCH_MESH 10) also count on the bound targets
    if matches!(last, 0 | 1 | 10) {
        draw(list, DRAW_EI);
    }
    with_frame(|f| *f.indirect.entry((last, path)).or_default() += 1);
}

pub fn not_done(what: &str) {
    if !on() {
        return;
    }
    count(NOT_DONE);
    with_frame(|f| *f.not_done.entry(what.to_string()).or_default() += 1);
}

fn after_s() -> u64 {
    static A: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *A.get_or_init(|| std::env::var("XCL_LOCAL_RTSUM_AFTER_S").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(0))
}

fn sig_name(t: u32) -> String {
    match t {
        0 => "Draw".into(),
        1 => "DrawIndexed".into(),
        2 => "Dispatch".into(),
        10 => "DispatchMesh".into(),
        u32::MAX => "unknown".into(),
        t => format!("type{t}"),
    }
}

unsafe fn desc_of(r: usize) -> Option<D3D12_RESOURCE_DESC> {
    let raw = r as *mut c_void;
    ID3D12Resource::from_raw_borrowed(&raw).map(|x| x.GetDesc())
}

/// At PresentX of `res` on `queue` (after XCL's copy of it): the frame's summary; `dump`:
/// also read its targets back.
pub unsafe fn on_present(queue: *mut c_void, res: usize, dump: bool) {
    if !on() {
        return;
    }
    static FIRST: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    static N: AtomicU64 = AtomicU64::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    let first = *FIRST.get_or_init(std::time::Instant::now);
    let frame = FRAME.lock().unwrap().take().unwrap_or_default();
    let drops: Vec<u64> = FRAME_DROPS.iter().map(|c| c.swap(0, Ordering::Relaxed)).collect();
    if first.elapsed().as_secs() < after_s() {
        return;
    }
    let mut t: Vec<(usize, Stat)> = frame.targets.iter().map(|(&r, &s)| (r, s)).collect();
    t.sort_by_key(|(_, s)| std::cmp::Reverse(s.draws.iter().sum::<u64>() * 4 + s.clears + s.rt_binds + s.ds_binds));
    let total = |k: usize| t.iter().map(|(_, s)| s.draws[k]).sum::<u64>();
    let mut ei: Vec<String> = frame.indirect.iter().map(|((s, p), c)| format!("{} {p} {c}", sig_name(*s))).collect();
    ei.sort();
    let d: Vec<String> = drops.iter().enumerate().filter(|(_, &c)| c > 0).map(|(k, c)| format!("{} {c} (total {})", DROP_NAMES[k], TOTAL[k].load(Ordering::Relaxed))).collect();
    let pres = desc_of(res).map(|d| format!("{}x{} f{}", d.Width, d.Height, d.Format.0)).unwrap_or_default();
    crate::xlog_info!(
        "rtsum: frame {n} (presents {res:#x} {pres}): {} targets; draws per target sum: translated {}, PC/stand-in {}, skipped {}, EI {}; draws without target {:?}; ExecuteIndirect [{}]; drops/counts [{}]; not done {:?}",
        t.len(),
        total(DRAW_TR),
        total(DRAW_PC),
        total(DRAW_SKIP),
        total(DRAW_EI),
        frame.draws_no_target,
        ei.join(", "),
        d.join(", "),
        frame.not_done
    );
    for (k, (r, s)) in t.iter().take(24).enumerate() {
        let dd = desc_of(*r);
        crate::xlog_info!(
            "rtsum: frame {n} target {k} {r:#x} {}: rt binds {}, ds binds {}, clears {}, draws tr {} pc {} skip {} ei {}{}",
            dd.map(|d| format!("{}x{}x{} f{} mips {} samples {} flags {:#x}", d.Width, d.Height, d.DepthOrArraySize, d.Format.0, d.MipLevels, d.SampleDesc.Count, d.Flags.0)).unwrap_or_else(|| "?".into()),
            s.rt_binds,
            s.ds_binds,
            s.clears,
            s.draws[0],
            s.draws[1],
            s.draws[2],
            s.draws[3],
            if *r == res { " (presented)" } else { "" }
        );
    }
    if dump {
        if let Some(dir) = std::env::var_os("XCL_LOCAL_RTSUM_DUMP_DIR") {
            let max: usize = std::env::var("XCL_LOCAL_RTSUM_DUMP_MAX").ok().and_then(|v| v.parse().ok()).unwrap_or(14);
            let mut list: Vec<usize> = t.iter().map(|(r, _)| *r).filter(|&r| r != res).take(max).collect();
            list.push(res);
            dump_targets(queue, &list, std::path::Path::new(&dir), n);
        }
    }
}

/// bytes per texel of a copyable footprint format (0: not handled)
fn texel_bytes(f: DXGI_FORMAT) -> usize {
    match f.0 {
        1..=4 => 16,
        5..=8 => 12,
        9..=22 => 8,
        23..=47 | 67 | 87..=93 => 4,
        48..=59 | 85 | 86 | 115 => 2,
        60..=65 => 1,
        _ => 0,
    }
}

fn f16(h: u16) -> f32 {
    let (s, e, m) = ((h >> 15) as u32, ((h >> 10) & 0x1F) as i32, (h & 0x3FF) as f32);
    let v = match e {
        0 => m / 1024.0 * 2f32.powi(-14),
        31 => {
            if m == 0.0 {
                f32::INFINITY
            } else {
                f32::NAN
            }
        }
        _ => (1.0 + m / 1024.0) * 2f32.powi(e - 15),
    };
    if s == 1 {
        -v
    } else {
        v
    }
}

fn small_float(v: u32, mantissa_bits: u32) -> f32 {
    let (e, m) = (v >> mantissa_bits, v & ((1 << mantissa_bits) - 1));
    let m = m as f32 / (1u32 << mantissa_bits) as f32;
    match e {
        0 => m * 2f32.powi(-14),
        31 => 65504.0,
        _ => (1.0 + m) * 2f32.powi(e as i32 - 15),
    }
}

/// up to 3 channels of the texel at `p` as f32 (None: format not decoded)
unsafe fn channels(f: DXGI_FORMAT, p: *const u8) -> Option<[f32; 3]> {
    let u32_at = |o: usize| (p.add(o) as *const u32).read_unaligned();
    let u16_at = |o: usize| (p.add(o) as *const u16).read_unaligned();
    Some(match f.0 {
        1..=4 | 5..=8 => {
            let g = |k: usize| if f.0 == 2 || f.0 == 6 { f32::from_bits(u32_at(4 * k)) } else { u32_at(4 * k) as f32 };
            [g(0), g(1), g(2)]
        }
        9..=14 => {
            let g = |k: usize| if f.0 == 10 || f.0 == 9 { f16(u16_at(2 * k)) } else { u16_at(2 * k) as f32 };
            [g(0), g(1), g(2)]
        }
        15..=18 => {
            let g = |k: usize| if f.0 == 16 || f.0 == 15 { f32::from_bits(u32_at(4 * k)) } else { u32_at(4 * k) as f32 };
            [g(0), g(1), 0.0]
        }
        19..=22 => [f32::from_bits(u32_at(0)), (u32_at(4) & 0xFF) as f32, 0.0],
        23..=25 => {
            let t = u32_at(0);
            [(t & 0x3FF) as f32, (t >> 10 & 0x3FF) as f32, (t >> 20 & 0x3FF) as f32]
        }
        26 => {
            let t = u32_at(0);
            [small_float(t & 0x7FF, 6), small_float(t >> 11 & 0x7FF, 6), small_float(t >> 22, 5)]
        }
        27..=32 => [*p as f32, *p.add(1) as f32, *p.add(2) as f32],
        87 | 88 | 90 | 91 => [*p.add(2) as f32, *p.add(1) as f32, *p as f32],
        33..=38 => {
            let g = |k: usize| if f.0 == 34 || f.0 == 33 { f16(u16_at(2 * k)) } else { u16_at(2 * k) as f32 };
            [g(0), g(1), 0.0]
        }
        39..=43 => {
            let t = u32_at(0);
            [if f.0 == 41 || f.0 == 39 || f.0 == 40 { f32::from_bits(t) } else { t as f32 }, 0.0, 0.0]
        }
        44..=47 => {
            let t = u32_at(0);
            [(t & 0xFF_FFFF) as f32, (t >> 24) as f32, 0.0]
        }
        67 => {
            let t = u32_at(0);
            let e = (t >> 27) as i32 - 15 - 9;
            let s = 2f32.powi(e);
            [(t & 0x1FF) as f32 * s, (t >> 9 & 0x1FF) as f32 * s, (t >> 18 & 0x1FF) as f32 * s]
        }
        48..=52 => [*p as f32, *p.add(1) as f32, 0.0],
        53..=59 => [if f.0 == 54 || f.0 == 53 { f16(u16_at(0)) } else { u16_at(0) as f32 }, 0.0, 0.0],
        60..=65 => [*p as f32, 0.0, 0.0],
        _ => return None,
    })
}

struct Pending {
    res: usize,
    desc: D3D12_RESOURCE_DESC,
    rb: ID3D12Resource,
    fp: D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
}

unsafe fn dump_targets(queue: *mut c_void, targets: &[usize], dir: &std::path::Path, frame: u64) {
    let dev_raw = super::real::device_raw();
    let Some(dev) = ID3D12Device::from_raw_borrowed(&dev_raw) else { return };
    let qraw = queue;
    let Some(q) = ID3D12CommandQueue::from_raw_borrowed(&qraw) else { return };
    let Ok(alloc) = dev.CreateCommandAllocator::<ID3D12CommandAllocator>(D3D12_COMMAND_LIST_TYPE_DIRECT) else { return };
    let Ok(list) = dev.CreateCommandList::<_, _, ID3D12GraphicsCommandList>(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None) else { return };
    let mut pending = Vec::new();
    for &r in targets {
        let raw = r as *mut c_void;
        let Some(res) = ID3D12Resource::from_raw_borrowed(&raw) else { continue };
        let d = res.GetDesc();
        if d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || d.SampleDesc.Count > 1 || d.Width < 16 || d.Height < 16 {
            crate::xlog_info!("rtsum: frame {frame} dump: {r:#x} not dumped (dimension {}, {} samples)", d.Dimension.0, d.SampleDesc.Count);
            continue;
        }
        let mut fp = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let mut total = 0u64;
        dev.GetCopyableFootprints(&d, 0, 1, 0, Some(&mut fp), None, None, Some(&mut total));
        if texel_bytes(fp.Footprint.Format) == 0 || total == 0 || total == u64::MAX {
            crate::xlog_info!("rtsum: frame {frame} dump: {r:#x} not dumped (footprint format {}, {total} bytes)", fp.Footprint.Format.0);
            continue;
        }
        let mut rb: Option<ID3D12Resource> = None;
        let bd = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: total,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        if dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut rb).is_err() {
            continue;
        }
        let Some(rb) = rb else { continue };
        let dst = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&rb), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: fp } };
        let src = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(res), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 } };
        list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
        add_ref(r);
        pending.push(Pending { res: r, desc: d, rb, fp });
    }
    if let Err(e) = list.Close() {
        crate::xlog_warn!("rtsum: frame {frame} dump: the copy list does not close ({e}) – nothing dumped");
        pending.iter().for_each(|p| release(p.res));
        return;
    }
    let Ok(fence) = dev.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE) else { return };
    let Ok(cl) = list.cast::<ID3D12CommandList>() else { return };
    q.ExecuteCommandLists(&[Some(cl)]);
    let _ = q.Signal(&fence, 1);
    let wait_ms = std::env::var("XCL_LOCAL_FRAME_DUMP_WAIT_MS").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(20000u32);
    let event = CreateEventW(None, false, false, None).ok();
    if let Err(e) = crate::gpu::fence::wait(&fence, 1, dev, event, wait_ms) {
        crate::xlog_warn!("rtsum: frame {frame} dump: copies not done ({e}) – nothing dumped");
        // the copies may still run: keep everything alive
        std::mem::forget((alloc, list, fence, pending));
        return;
    }
    if let Some(e) = event {
        let _ = windows::Win32::Foundation::CloseHandle(e);
    }
    let _ = std::fs::create_dir_all(dir);
    for (k, p) in pending.iter().enumerate() {
        write_target(p, dir, frame, k);
        release(p.res);
    }
}

unsafe fn write_target(p: &Pending, dir: &std::path::Path, frame: u64, k: usize) {
    let mut m: *mut c_void = std::ptr::null_mut();
    if p.rb.Map(0, None, Some(&mut m)).is_err() || m.is_null() {
        return;
    }
    let fp = &p.fp;
    let f = fp.Footprint.Format;
    let bpt = texel_bytes(f);
    let (w, h, pitch) = (fp.Footprint.Width as usize, fp.Footprint.Height as usize, fp.Footprint.RowPitch as usize);
    // a texel every `step` in each direction: at most ~1000 samples a side
    let step = (w.max(h) / 960).max(1);
    let (ow, oh) = ((w / step).max(1), (h / step).max(1));
    let base = (m as *const u8).add(fp.Offset as usize);
    let mut raw: HashMap<Vec<u8>, u64> = HashMap::new();
    let mut vals: Vec<[f32; 3]> = Vec::with_capacity(ow * oh);
    let mut decoded = true;
    let (mut lo, mut hi) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
    let mut nonfinite = 0u64;
    for y in 0..oh {
        for x in 0..ow {
            let t = base.add(step * y * pitch + bpt * step * x);
            let key = std::slice::from_raw_parts(t, bpt.min(16)).to_vec();
            if raw.len() < 100_000 || raw.contains_key(&key) {
                *raw.entry(key).or_default() += 1;
            }
            match channels(f, t) {
                Some(c) => {
                    for i in 0..3 {
                        if c[i].is_finite() {
                            lo[i] = lo[i].min(c[i]);
                            hi[i] = hi[i].max(c[i]);
                        } else {
                            nonfinite += 1;
                        }
                    }
                    vals.push(c);
                }
                None => {
                    decoded = false;
                    vals.push([0.0; 3]);
                }
            }
        }
    }
    p.rb.Unmap(0, None);
    let samples = (ow * oh) as u64;
    let mut top: Vec<(Vec<u8>, u64)> = raw.into_iter().collect();
    let distinct = top.len();
    top.sort_by_key(|x| std::cmp::Reverse(x.1));
    let hex = |b: &[u8]| b.iter().rev().map(|v| format!("{v:02x}")).collect::<String>();
    // BMP: each channel stretched lo..hi (a constant channel is mid grey)
    let row = (ow * 3).div_ceil(4) * 4;
    let mut bmp = Vec::with_capacity(54 + row * oh);
    let size = (54 + row * oh) as u32;
    bmp.extend_from_slice(b"BM");
    for v in [size, 0, 54, 40, ow as u32, oh as u32] {
        bmp.extend_from_slice(&v.to_le_bytes());
    }
    bmp.extend_from_slice(&1u16.to_le_bytes());
    bmp.extend_from_slice(&24u16.to_le_bytes());
    for _ in 0..6 {
        bmp.extend_from_slice(&0u32.to_le_bytes());
    }
    let single = matches!(f.0, 39..=47 | 53..=65 | 19..=22);
    for y in (0..oh).rev() {
        let start = bmp.len();
        for x in 0..ow {
            let c = vals[y * ow + x];
            let s = |i: usize| -> u8 {
                if !(hi[i] > lo[i]) {
                    return 128;
                }
                let v = (c[i] - lo[i]) / (hi[i] - lo[i]);
                if v.is_finite() { (v.clamp(0.0, 1.0) * 255.0) as u8 } else { 255 }
            };
            let (r, g, b) = if single { (s(0), s(0), s(0)) } else { (s(0), s(1), s(2)) };
            bmp.extend_from_slice(&[b, g, r]);
        }
        bmp.resize(start + row, 0);
    }
    let name = format!("rt_f{frame}_{k:02}_{}x{}_fmt{}.bmp", p.desc.Width, p.desc.Height, p.desc.Format.0);
    let path = dir.join(&name);
    let _ = std::fs::write(&path, &bmp);
    crate::xlog_info!(
        "rtsum: frame {frame} dump {k} {:#x} {}x{} res f{} footprint f{}: {name}; {samples} samples (step {step}), {distinct} distinct, most common {}; ch min {:?} max {:?}{}{}",
        p.res,
        p.desc.Width,
        p.desc.Height,
        p.desc.Format.0,
        f.0,
        top.iter().take(4).map(|(v, c)| format!("{}×{c} ({:.1}%)", hex(v), 100.0 * *c as f64 / samples as f64)).collect::<Vec<_>>().join(" "),
        lo,
        hi,
        if nonfinite > 0 { format!(", {nonfinite} non-finite channel values") } else { String::new() },
        if decoded { "" } else { " (not decoded)" }
    );
}
