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

/// LOCAL (agent H): the targets OMSetRenderTargets last bound on `list` (rts, ds)
pub fn bound(list: *mut c_void) -> Option<(Vec<usize>, usize)> {
    BOUND.lock().unwrap().as_ref()?.get(&(list as usize)).cloned()
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
    static AGENT_I: std::sync::Once = std::sync::Once::new();
    AGENT_I.call_once(|| {
        crate::xlog_info!(
            "LOCAL agentI w29: TWIN_AT_USE {:?} TWIN_MINW {:?} TWIN_SNAP_CS {:?} ALIAS_AT_USE {:?} ALIAS_LOG {:?}",
            std::env::var("XCL_LOCAL_TWIN_AT_USE").ok(),
            std::env::var("XCL_LOCAL_TWIN_MINW").ok(),
            std::env::var("XCL_LOCAL_TWIN_SNAP_CS").ok(),
            std::env::var("XCL_LOCAL_ALIAS_AT_USE").ok(),
            std::env::var("XCL_LOCAL_ALIAS_LOG").ok()
        )
    });
    static FIRST: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    static N: AtomicU64 = AtomicU64::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    let first = *FIRST.get_or_init(std::time::Instant::now);
    let frame = FRAME.lock().unwrap().take().unwrap_or_default();
    let drops: Vec<u64> = FRAME_DROPS.iter().map(|c| c.swap(0, Ordering::Relaxed)).collect();
    // LOCAL (agent Z, `XCL_LOCAL_CB_READBACK=1`): the GPU's counter buffer back to the CPU at each present
    if let Some((a, len)) = super::bind::local_cb_range() {
        let before: Vec<u32> = (0..len.min(88) / 4).map(|k| (a as *const u32).add(k).read_unaligned()).collect();
        super::sparse::readback_now(a, len);
        let after: Vec<u32> = (0..len.min(88) / 4).map(|k| (a as *const u32).add(k).read_unaligned()).collect();
        static L: AtomicU64 = AtomicU64::new(0);
        let k = L.fetch_add(1, Ordering::Relaxed);
        if k < 4 || k % 100 == 0 || (before != after && k < 400 && k % 10 == 0) {
            crate::xlog_info!("LOCAL cb-readback: frame {n}: CPU words before {before:x?} after {after:x?}");
        }
    }
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
    for (k, (r, s)) in t.iter().take(if fs_frames() > 0 { 64 } else { 24 }).enumerate() { // LOCAL: 64 with FSTRACE (agent Z)
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
    oow_copy_reset(queue, n); // LOCAL experiment (agent L, ported by agent Z; segments: agent G)
    mirror_refresh(n); // LOCAL experiment (agent L, ported by agent Z)
    super::real::atv_summary(n); // LOCAL (agent V, ported by agent Z)
    eicap_present(queue, n, total(DRAW_EI)); // ExecuteIndirect capture
    let fs_extra = fs_present(n, dump && total(DRAW_EI) >= 100, &t.iter().map(|(r, _)| *r).collect::<Vec<_>>()); // LOCAL (agent Z)
    if dump {
        if let Some(dir) = std::env::var_os("XCL_LOCAL_RTSUM_DUMP_DIR") {
            let max: usize = std::env::var("XCL_LOCAL_RTSUM_DUMP_MAX").ok().and_then(|v| v.parse().ok()).unwrap_or(14);
            let mut list: Vec<usize> = t.iter().map(|(r, _)| *r).filter(|&r| r != res).take(max).collect();
            list.push(res);
            // LOCAL (agent Z): the textures the traced full-screen dispatches named, after the targets
            list.extend(fs_extra.iter().copied().filter(|r| *r != res && !t.iter().take(max).any(|(x, _)| x == r)));
            list.extend(super::depth_uav::local_dump_extra(n)); // LOCAL agent I: depth twins and their snapshots
            list.extend(local_3d_list()); // LOCAL agent J: small 3D title textures (XCL_LOCAL_LUT_DUMP)
            dump_targets(queue, &list, std::path::Path::new(&dir), n);
        }
    }
    fs_extra.iter().for_each(|&r| release(r)); // LOCAL (agent Z)
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
        let local_3d = d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D && local_lut_on(); // LOCAL agent J
        if local_3d {
        } else if d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || d.SampleDesc.Count > 1 || ((d.Width < 16 || d.Height < 16) && fs_frames() == 0) { // LOCAL: small ones too with FSTRACE (agent Z)
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
        if p.desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
            write_3d(p, dir, frame, k); // LOCAL agent J
        } else {
            write_target(p, dir, frame, k);
        }
        release(p.res);
    }
}

// ---------------------------------------------------------------------------------------
// LOCAL agent J (`XCL_LOCAL_LUT_DUMP=1`, with the frame dumps): the title's small 3D textures
// (colour look-up tables a compute program bakes) are kept (32 to 256 wide; one reference each, at most 16) and
// read back whole with each dump: distinct texels over all slices, raw bytes to `<dir>`.
// ---------------------------------------------------------------------------------------
static LOCAL_3D: Mutex<Vec<usize>> = Mutex::new(Vec::new());

fn local_lut_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| on() && std::env::var("XCL_LOCAL_LUT_DUMP").is_ok_and(|v| v == "1"))
}

pub unsafe fn local_note_3d(res: usize, pc: &D3D12_RESOURCE_DESC) {
    // (32 to 256 wide only: the title also makes dozens of 17-cube and 1-cube ones)
    if !local_lut_on() || pc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE3D || pc.Width > 256 || pc.Width < 32 {
        return;
    }
    let mut g = LOCAL_3D.lock().unwrap();
    if g.len() >= 16 || g.contains(&res) {
        return;
    }
    add_ref(res);
    g.push(res);
    crate::xlog_info!("LOCAL agentJ lut: 3D title texture {res:#x} {}x{}x{} f{} mips {} flags {:#x} (#{}, kept for the dumps)", pc.Width, pc.Height, pc.DepthOrArraySize, pc.Format.0, pc.MipLevels, pc.Flags.0, g.len());
}

fn local_3d_list() -> Vec<usize> {
    LOCAL_3D.lock().unwrap().clone()
}

unsafe fn write_3d(p: &Pending, dir: &std::path::Path, frame: u64, k: usize) {
    let mut m: *mut c_void = std::ptr::null_mut();
    if p.rb.Map(0, None, Some(&mut m)).is_err() || m.is_null() {
        return;
    }
    let fp = &p.fp;
    let f = fp.Footprint.Format;
    let bpt = texel_bytes(f);
    let (w, h, d, pitch) = (fp.Footprint.Width as usize, fp.Footprint.Height as usize, fp.Footprint.Depth as usize, fp.Footprint.RowPitch as usize);
    let base = (m as *const u8).add(fp.Offset as usize);
    let mut raw: HashMap<Vec<u8>, u64> = HashMap::new();
    let (mut lo, mut hi) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
    let mut slice0: std::collections::HashSet<Vec<u8>> = std::collections::HashSet::new();
    let mut bytes = Vec::with_capacity(w * h * d * bpt);
    for z in 0..d {
        for y in 0..h {
            let row = base.add((z * h + y) * pitch);
            bytes.extend_from_slice(std::slice::from_raw_parts(row, w * bpt));
            for x in 0..w {
                let t = row.add(bpt * x);
                let key = std::slice::from_raw_parts(t, bpt.min(16)).to_vec();
                if z == 0 {
                    slice0.insert(key.clone());
                }
                *raw.entry(key).or_default() += 1;
                if let Some(c) = channels(f, t) {
                    for i in 0..3 {
                        if c[i].is_finite() {
                            lo[i] = lo[i].min(c[i]);
                            hi[i] = hi[i].max(c[i]);
                        }
                    }
                }
            }
        }
    }
    p.rb.Unmap(0, None);
    let mut top: Vec<(Vec<u8>, u64)> = raw.into_iter().collect();
    let distinct = top.len();
    top.sort_by_key(|x| std::cmp::Reverse(x.1));
    let hex = |b: &[u8]| b.iter().rev().map(|v| format!("{v:02x}")).collect::<String>();
    let name = format!("lut_f{frame}_{k:02}_{w}x{h}x{d}_fmt{}.bin", p.desc.Format.0);
    let _ = std::fs::write(dir.join(&name), &bytes);
    crate::xlog_info!(
        "LOCAL agentJ lut: frame {frame} dump {k} {:#x} {w}x{h}x{d} res f{} footprint f{}: {name}; {} texels, {distinct} distinct ({} distinct in slice 0), most common {}; ch min {lo:?} max {hi:?}",
        p.res,
        p.desc.Format.0,
        f.0,
        w * h * d,
        slice0.len(),
        top.iter().take(4).map(|(v, c)| format!("{}x{c}", hex(v))).collect::<Vec<_>>().join(" ")
    );
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
    // LOCAL agent G (`XCL_LOCAL_RTSUM_PCT=1`): for float colour targets also a BMP stretched to the
    // 99th percentile of the brightest channel (one scale for all channels, gamma 1/2.2), with the percentiles logged
    if std::env::var_os("XCL_LOCAL_RTSUM_PCT").is_some() && matches!(f.0, 2 | 10 | 24 | 26) {
        let mut lum: Vec<f32> = vals.iter().map(|c| c[0].max(c[1]).max(c[2])).filter(|v| v.is_finite() && *v > 0.0).collect();
        lum.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q = |x: f64| if lum.is_empty() { 0.0 } else { lum[((lum.len() - 1) as f64 * x) as usize] };
        let (p50, p90, p99) = (q(0.5), q(0.9), q(0.99));
        let scale = if p99 > 0.0 { p99 } else { 1.0 };
        let row = (ow * 3).div_ceil(4) * 4;
        let mut bmp = Vec::with_capacity(54 + row * oh);
        bmp.extend_from_slice(b"BM");
        for v in [(54 + row * oh) as u32, 0, 54, 40, ow as u32, oh as u32] {
            bmp.extend_from_slice(&v.to_le_bytes());
        }
        bmp.extend_from_slice(&1u16.to_le_bytes());
        bmp.extend_from_slice(&24u16.to_le_bytes());
        for _ in 0..6 {
            bmp.extend_from_slice(&0u32.to_le_bytes());
        }
        for y in (0..oh).rev() {
            let start = bmp.len();
            for x in 0..ow {
                let c = vals[y * ow + x];
                let g = |v: f32| if v.is_finite() { ((v / scale).clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0) as u8 } else { 255 };
                bmp.extend_from_slice(&[g(c[2]), g(c[1]), g(c[0])]);
            }
            bmp.resize(start + row, 0);
        }
        let name = format!("rt_f{frame}_{k:02}_{}x{}_fmt{}_pct.bmp", p.desc.Width, p.desc.Height, p.desc.Format.0);
        let _ = std::fs::write(dir.join(&name), &bmp);
        crate::xlog_info!("LOCAL rtsum-pct: frame {frame} dump {k} {:#x}: {name}; non-zero texels {} of {}; brightest channel p50 {p50} p90 {p90} p99 {p99}", p.res, lum.len(), vals.len());
    }
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

// ---------------------------------------------------------------------------------------
// LOCAL: ExecuteIndirect capture (`XCL_LOCAL_EICAP_FRAMES=n`, default off): the first n frames
// after `XCL_LOCAL_RTSUM_AFTER_S` that follow a frame with at least 100 indirect draws get, after
// each translated ExecuteIndirect, a copy of its count and first argument records (on the
// title's list, so as the GPU has them after that call) into a readback buffer; read and logged
// two presents later. Never freed (a list run late must not write freed memory).
// ---------------------------------------------------------------------------------------

static STRIDES: Mutex<Option<HashMap<usize, u32>>> = Mutex::new(None);
static GFX: Mutex<Option<HashMap<usize, [u64; 2]>>> = Mutex::new(None);
static CSH: Mutex<Option<HashMap<usize, u64>>> = Mutex::new(None);

pub fn note_stride(sig: usize, stride: u32) {
    if on() && sig != 0 {
        STRIDES.lock().unwrap().get_or_insert_with(HashMap::new).insert(sig, stride);
    }
}

unsafe fn shader_hash(b: &D3D12_SHADER_BYTECODE) -> u64 {
    let (p, n) = (b.pShaderBytecode as usize, b.BytecodeLength);
    if p == 0 || n == 0 || !crate::gpu::shader::readable(p, n) {
        return 0;
    }
    crate::gpu::shader::fnv64(std::slice::from_raw_parts(p as *const u8, n))
}

pub unsafe fn note_graphics(pso: usize, desc: *const c_void) {
    if !on() || pso == 0 || !crate::gpu::shader::readable(desc as usize, std::mem::size_of::<D3D12_GRAPHICS_PIPELINE_STATE_DESC>()) {
        return;
    }
    let d = std::ptr::read(desc as *const D3D12_GRAPHICS_PIPELINE_STATE_DESC);
    let h = [shader_hash(&d.VS), shader_hash(&d.PS)];
    std::mem::forget(d);
    GFX.lock().unwrap().get_or_insert_with(HashMap::new).insert(pso, h);
}

pub unsafe fn note_compute(pso: usize, desc: *const c_void) {
    if !on() || pso == 0 || !crate::gpu::shader::readable(desc as usize, std::mem::size_of::<D3D12_COMPUTE_PIPELINE_STATE_DESC>()) {
        return;
    }
    let d = std::ptr::read(desc as *const D3D12_COMPUTE_PIPELINE_STATE_DESC);
    let h = shader_hash(&d.CS);
    std::mem::forget(d);
    CSH.lock().unwrap().get_or_insert_with(HashMap::new).insert(pso, h);
}

pub fn shaders(pso: usize) -> String { // LOCAL: pub (agent Z)
    let real = super::pso_async::resolve(pso);
    if let Some(h) = [pso, real].iter().find_map(|p| GFX.lock().unwrap().as_ref().and_then(|m| m.get(p).copied())) {
        return format!("vs {:016x} ps {:016x}", h[0], h[1]);
    }
    if let Some(h) = [pso, real].iter().find_map(|p| CSH.lock().unwrap().as_ref().and_then(|m| m.get(p).copied())) {
        return format!("cs {h:016x}");
    }
    format!("pso {pso:#x} (shaders not seen)")
}

const SLOT: u64 = 256;
const SLOTS: u64 = 8192;

struct EiCall {
    slot: u64,
    text: String,
    stride: u32,
    max: u32,
    has_count: bool,
    gfx: bool,
    args_len: u64,
    last: u32,
}

struct Cap {
    frame: u64,
    rb: ID3D12Resource,
    used: u64,
    calls: Vec<EiCall>,
    draws: Vec<String>,
    /// LOCAL (agent J): record headers the two-pass compute pass wrote: (slot, records copied, text)
    hdrs: Vec<(u64, u64, String)>,
    /// direct two-pass DispatchMesh calls: [0 groups, groups > 0], and the groups summed
    dm: [u64; 3],
}

static CAP: Mutex<Option<Cap>> = Mutex::new(None);
static PENDING: Mutex<Vec<Cap>> = Mutex::new(Vec::new());
static CAPTURED: AtomicU64 = AtomicU64::new(0);

fn cap_frames() -> u64 {
    static N: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("XCL_LOCAL_EICAP_FRAMES").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(0))
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn after_indirect(list: *mut c_void, sig: usize, max: u32, args: usize, args_off: u64, count_buf: usize, count_off: u64, gfx: bool, pso: usize) {
    // LOCAL (agent Z, `XCL_LOCAL_ZARGS_VS=<hex prefix>`): the argument records of that vertex shader's
    // indirect draws become the watched range of `bind::local_zwatch` (title address of the first one)
    if gfx && on() {
        static VS: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
        if let Some(v) = VS.get_or_init(|| std::env::var("XCL_LOCAL_ZARGS_VS").ok().map(|s| format!("vs {}", s.trim().to_lowercase()))) {
            if shaders(pso).contains(v.as_str()) {
                if let Some(ta) = super::real::title_address(args) {
                    let at = (ta as u64 + args_off).wrapping_sub(super::real::buffer_delta(args)); // LOCAL: args_off here includes the PC delta
                    if super::bind::local_zwatch_set(at, 0x60) {
                        let raw = args as *mut c_void;
                        let info = ID3D12Resource::from_raw_borrowed(&raw).map(|r| {
                            let d = r.GetDesc();
                            let mut hp = D3D12_HEAP_PROPERTIES::default();
                            let mut hf = D3D12_HEAP_FLAGS::default();
                            let heap = r.GetHeapProperties(Some(&mut hp), Some(&mut hf)).map(|_| (hp.Type.0, hf.0)).ok();
                            (d.Width, r.GetGPUVirtualAddress(), heap)
                        });
                        let win = super::real::uma_window_buffer(at as usize, 16);
                        crate::xlog_info!(
                            "zargs: {} args {args:#x}+{args_off:#x} = title {at:#x}; console_buffer {:x?}, buffer_delta {:#x}, resource (width, gpu va, heap type/flags) {:x?}; window buffer of the title address {:x?}; window VAs {:x?}",
                            shaders(pso), super::real::console_buffer(args), super::real::buffer_delta(args), info, win, super::real::uma_window_vas()
                        );
                    }
                }
            }
        }
    }
    if !on() || cap_frames() == 0 {
        return;
    }
    let mut g = CAP.lock().unwrap();
    let Some(c) = g.as_mut() else { return };
    if c.used >= SLOTS {
        return;
    }
    if super::list_x::tracked_list_type(list).is_none_or(|t| t == D3D12_COMMAND_LIST_TYPE_BUNDLE) {
        return;
    }
    let Some(l) = ID3D12GraphicsCommandList::from_raw_borrowed(&list) else { return };
    let stride = STRIDES.lock().unwrap().as_ref().and_then(|m| m.get(&sig).copied()).unwrap_or(0);
    let last = SIGS.lock().unwrap().as_ref().and_then(|m| m.get(&sig).copied()).unwrap_or(u32::MAX);
    let slot = c.used;
    c.used += 1;
    let base = slot * SLOT;
    let (araw, craw) = (args as *mut c_void, count_buf as *mut c_void);
    let mut args_len = 0;
    if let Some(a) = ID3D12Resource::from_raw_borrowed(&araw) {
        let want = (stride as u64 * max.min(4) as u64).min(SLOT - 16);
        let w = a.GetDesc().Width;
        args_len = want.min(w.saturating_sub(args_off)) & !3;
        if args_len > 0 {
            l.CopyBufferRegion(&c.rb, base + 16, a, args_off, args_len);
        }
    }
    let has_count = count_buf != 0;
    if let Some(cb) = ID3D12Resource::from_raw_borrowed(&craw) {
        if count_off + 4 <= cb.GetDesc().Width {
            l.CopyBufferRegion(&c.rb, base, cb, count_off, 4);
        }
    }
    let rts = BOUND.lock().unwrap().as_ref().and_then(|m| m.get(&(list as usize)).cloned());
    let tgt = rts.map(|(r, d)| format!("rt {:x?} ds {d:#x}", r)).unwrap_or_default();
    let text = format!(
        "list {:#x} {} {} args {args:#x}+{args_off:#x} count {count_buf:#x}+{count_off:#x}; {tgt}; {}",
        list as usize,
        sig_name(last),
        shaders(pso),
        if gfx { super::list_x::local_draw_state(list) } else { String::new() }
    );
    c.calls.push(EiCall { slot, text, stride, max, has_count, gfx, args_len, last });
}

pub fn direct_draw(list: *mut c_void, pso: usize, a: (bool, u32, u32, u32, i32)) {
    if !on() || cap_frames() == 0 {
        return;
    }
    let mut g = CAP.lock().unwrap();
    let Some(c) = g.as_mut() else { return };
    if c.draws.len() >= 2000 {
        return;
    }
    let rts = BOUND.lock().unwrap().as_ref().and_then(|m| m.get(&(list as usize)).cloned());
    let tgt = rts.map(|(r, d)| format!("rt {:x?} ds {d:#x}", r)).unwrap_or_default();
    c.draws.push(format!(
        "list {:#x} {} {} count {} x{} first {} base {}; {tgt}; {}",
        list as usize,
        if a.0 { "DrawIndexed" } else { "Draw" },
        shaders(pso),
        a.1,
        a.2,
        a.3,
        a.4,
        super::list_x::local_draw_state(list)
    ));
}

/// At each present (`on_present`, after the summary): close / open captures, log old ones.
unsafe fn eicap_present(queue: *mut c_void, n: u64, prev_ei_draws: u64) {
    if cap_frames() == 0 {
        return;
    }
    // a capture that covered the frame just presented
    if let Some(mut c) = CAP.lock().unwrap().take() {
        c.frame = n;
        PENDING.lock().unwrap().push(c);
        oow_log(n); // LOCAL out-of-window scan of that frame (agent L, ported by agent Z)
    }
    if CAPTURED.load(Ordering::Relaxed) < cap_frames() && prev_ei_draws >= 100 {
        let dev_raw = super::real::device_raw();
        if let Some(dev) = ID3D12Device::from_raw_borrowed(&dev_raw) {
            let bd = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: SLOT * SLOTS,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            };
            let mut rb: Option<ID3D12Resource> = None;
            if dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut rb).is_ok() {
                if let Some(rb) = rb {
                    // marker: a slot the GPU never wrote keeps it
                    let mut m: *mut c_void = std::ptr::null_mut();
                    if rb.Map(0, None, Some(&mut m)).is_ok() && !m.is_null() {
                        std::ptr::write_bytes(m as *mut u8, 0xCD, (SLOT * SLOTS) as usize);
                        rb.Unmap(0, None);
                    }
                    CAPTURED.fetch_add(1, Ordering::Relaxed);
                    crate::xlog_info!("eicap: capturing the frame after present {n}");
                    *CAP.lock().unwrap() = Some(Cap { frame: 0, rb, used: 0, calls: Vec::new(), draws: Vec::new(), hdrs: Vec::new(), dm: [0; 3] });
                }
            }
        }
    }
    // captures two presents old: wait for the present queue, then log
    let ready: Vec<Cap> = {
        let mut p = PENDING.lock().unwrap();
        let (r, keep): (Vec<Cap>, Vec<Cap>) = p.drain(..).partition(|c| c.frame + 2 <= n);
        *p = keep;
        r
    };
    if ready.is_empty() {
        return;
    }
    let dev_raw = super::real::device_raw();
    let Some(dev) = ID3D12Device::from_raw_borrowed(&dev_raw) else { return };
    let Some(q) = ID3D12CommandQueue::from_raw_borrowed(&queue) else { return };
    if let Ok(fence) = dev.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE) {
        let _ = q.Signal(&fence, 1);
        let event = CreateEventW(None, false, false, None).ok();
        if let Err(e) = crate::gpu::fence::wait(&fence, 1, dev, event, 20000) {
            crate::xlog_warn!("eicap: present queue not done ({e}); reading anyway (unwritten slots show cdcdcdcd)");
        }
        if let Some(e) = event {
            let _ = windows::Win32::Foundation::CloseHandle(e);
        }
    }
    for c in ready {
        log_cap(&c);
        std::mem::forget(c); // the buffer is never freed (see above)
    }
}

unsafe fn log_cap(c: &Cap) {
    let mut m: *mut c_void = std::ptr::null_mut();
    if c.rb.Map(0, None, Some(&mut m)).is_err() || m.is_null() {
        return;
    }
    let w = |o: u64| ((m as *const u8).add(o as usize) as *const u32).read_unaligned();
    let f = c.frame;
    let (mut g_calls, mut g_zero_count, mut g_zero_inst, mut g_unwritten, mut g_work) = (0usize, 0, 0, 0, 0);
    let mut c_zero = 0;
    let (mut m_calls, mut m_zero, mut m_work, mut m_unwritten, mut m_groups) = (0usize, 0usize, 0usize, 0usize, 0u64);
    for e in &c.calls {
        let base = e.slot * SLOT;
        let count = if e.has_count { Some(w(base)) } else { None };
        let words: Vec<u32> = (0..e.args_len / 4).map(|k| w(base + 16 + 4 * k)).collect();
        let unwritten = words.first() == Some(&0xCDCD_CDCD) || count == Some(0xCDCD_CDCD);
        // the draw / dispatch at the end of each record
        let tail: usize = match e.last {
            0 | 2 => 4,
            1 => 5,
            10 => 3,
            _ => 0,
        };
        let tail = if e.last == 2 { 3 } else { tail };
        let per = (e.stride / 4) as usize;
        let recs: Vec<Vec<u32>> = if per >= tail && per > 0 && tail > 0 { words.chunks(per).filter(|r| r.len() == per).map(|r| r[per - tail..].to_vec()).collect() } else { Vec::new() };
        let n_eff = count.map_or(e.max, |c| c.min(e.max));
        if e.last == 10 {
            // LOCAL: DispatchMesh records (x, y, z groups)
            m_calls += 1;
            if unwritten {
                m_unwritten += 1;
            } else if n_eff == 0 || recs.first().is_none_or(|r| r.iter().take(3).any(|&v| v == 0)) {
                m_zero += 1;
            } else {
                m_work += 1;
                m_groups += recs.first().map_or(0, |r| r[0] as u64 * r[1] as u64 * r[2] as u64);
            }
        } else if e.gfx {
            g_calls += 1;
            if unwritten {
                g_unwritten += 1;
            } else if n_eff == 0 {
                g_zero_count += 1;
            } else if recs.first().is_some_and(|r| r[0] == 0 || r[1] == 0) {
                g_zero_inst += 1;
            } else {
                g_work += 1;
            }
        } else if !unwritten && (n_eff == 0 || recs.first().is_some_and(|r| r.iter().take(3).any(|&v| v == 0))) {
            c_zero += 1;
        }
        crate::xlog_info!(
            "eicap: frame {f} #{} {}: max {} count {:?} stride {} records {:x?}{} | {}",
            e.slot,
            if e.gfx { "gfx" } else { "cs" },
            e.max,
            count.map(|c| if c == 0xCDCD_CDCD { "unwritten".to_string() } else { c.to_string() }),
            e.stride,
            recs.iter().take(4).collect::<Vec<_>>(),
            if unwritten { " (slot never written)" } else { "" },
            e.text
        );
    }
    // LOCAL: the headers (vertices, primitives) of the first records the two-pass compute pass wrote
    let (mut h_n, mut h_empty, mut h_verts, mut h_prims, mut h_bad) = (0u64, 0u64, 0u64, 0u64, 0u64);
    for (slot, n, text) in &c.hdrs {
        let base = slot * SLOT;
        let hs: Vec<(u32, u32)> = (0..*n).map(|k| (w(base + 16 + 16 * k), w(base + 20 + 16 * k))).collect();
        for &(nv, np) in &hs {
            h_n += 1;
            if nv == 0xCDCD_CDCD {
                h_bad += 1;
            } else if nv == 0 || np == 0 {
                h_empty += 1;
            } else {
                h_verts += nv as u64;
                h_prims += np as u64;
            }
        }
        // LOCAL (agent N): header dwords 2 / 3 (`XCL_LOCAL_MS_PROBE`: r0 = (s17, s18), r1 = (V#.x, V#.y), r2 = (V#.z, count), r3 = (entry 0, entry 1))
        let probe: Vec<(u32, u32)> = (0..*n).map(|k| (w(base + 24 + 16 * k), w(base + 28 + 16 * k))).collect();
        crate::xlog_info!("eicap: frame {f} two-pass records (vertices, primitives) {:x?} probe {:x?} | {text}", hs, probe);
    }
    c.rb.Unmap(0, None);
    crate::xlog_info!(
        "eicap: frame {f} mesh summary: {m_calls} indirect DispatchMesh: {m_zero} with 0 groups (or count 0), {m_work} with groups ({m_groups} groups in their first records), {m_unwritten} not executed by read time; direct two-pass DispatchMesh: {} with 0 groups, {} with groups ({} groups); record headers read: {h_n}: {h_empty} empty (0 vertices or primitives), {} with geometry ({h_verts} vertices, {h_prims} primitives), {h_bad} unwritten",
        c.dm[0], c.dm[1], c.dm[2], h_n - h_empty - h_bad
    );
    for (k, d) in c.draws.iter().enumerate() {
        crate::xlog_info!("eicap: frame {f} direct draw {k}: {d}");
    }
    crate::xlog_info!(
        "eicap: frame {f} summary: {g_calls} indirect draws: {g_zero_count} with count 0, {g_zero_inst} with a first record of 0 vertices/instances, {g_work} with work, {g_unwritten} not executed by read time; {} indirect dispatches ({c_zero} with no groups); {} direct draws",
        c.calls.len() - g_calls,
        c.draws.len()
    );
}

/// LOCAL (agent N): a frame is being captured and has room for record headers (`bind::local_ms_probe`).
pub fn capturing() -> bool {
    on() && cap_frames() > 0 && CAP.lock().unwrap().as_ref().is_some_and(|c| c.used < SLOTS && c.hdrs.len() < 400)
}

/// LOCAL (agent J): a direct two-pass DispatchMesh of `groups` (x·y·z) in a captured frame.
pub fn direct_mesh(groups: u64) {
    if !on() || cap_frames() == 0 {
        return;
    }
    if let Some(c) = CAP.lock().unwrap().as_mut() {
        c.dm[if groups == 0 { 0 } else { 1 }] += 1;
        c.dm[2] += groups;
    }
}

/// LOCAL (agent J): after a two-pass compute pass on `list`, copy the headers (16 bytes) of its first
/// `n` records (`record_bytes` apart from `res_va`'s buffer offset 0) for the log of a captured frame.
pub unsafe fn two_pass_headers(list: *mut c_void, res: &ID3D12Resource, record_bytes: u64, n: u64, text: String) {
    if !on() || cap_frames() == 0 || n == 0 {
        return;
    }
    let mut g = CAP.lock().unwrap();
    let Some(c) = g.as_mut() else { return };
    if c.used >= SLOTS || c.hdrs.len() >= 400 {
        return;
    }
    let Some(l) = ID3D12GraphicsCommandList::from_raw_borrowed(&list) else { return };
    let n = n.min((SLOT - 16) / 16);
    let slot = c.used;
    c.used += 1;
    let barrier = |before: D3D12_RESOURCE_STATES, after: D3D12_RESOURCE_STATES| {
        let b = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                    pResource: std::mem::ManuallyDrop::new(Some(res.clone())),
                    Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                    StateBefore: before,
                    StateAfter: after,
                }),
            },
        };
        l.ResourceBarrier(&[b.clone()]);
        let mut b = b;
        std::mem::ManuallyDrop::drop(&mut (*b.Anonymous.Transition).pResource);
    };
    barrier(D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    for k in 0..n {
        l.CopyBufferRegion(&c.rb, slot * SLOT + 16 + 16 * k, res, k * record_bytes, 16);
    }
    barrier(D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    c.hdrs.push((slot, n, text));
}

// ======== LOCAL (agent Z): agent L's out-of-window scan, root-view copies and empty-view mirrors, with
// agent V's ATV packing, ported from S/w25 rtsum.rs (lines 1105-1594 there) without the page-map census hook.
// ---------------------------------------------------------------------------------------
// LOCAL: out-of-window scan (`XCL_LOCAL_OOW=1`, in the ExecuteIndirect capture frames). At each
// translated draw / dispatch (`bind::bind_abi`), the addresses its root arguments give it are
// classified the way a translated memory access resolves them (`xcl_w` = high half minus the
// unified memory's): root CBV/SRV/UAV addresses, 64-bit-looking pairs in root constants, the
// buffer descriptors (V#) of its descriptor tables (first 64 slots of 16 bytes, window 0), and
// 64-bit-looking pairs in the first 256 bytes of small buffers those point at. An address
// outside the windows reads 0 and drops writes on the GPU; the CPU view of it is sampled.
// Per program, one line when the capture frame is logged. Heuristic (table slots and pairs are
// guessed, not taken from the program's code): a lead, not a proof.
// ---------------------------------------------------------------------------------------

#[derive(Default)]
struct OowProg {
    calls: u64,
    gfx: bool,
    inside: u64,
    /// calls given a root CBV/SRV/UAV address outside the windows (not a guess), and how many such
    rv_calls: u64,
    rv_addrs: std::collections::HashSet<u64>,
    /// address -> (where it came from, what the CPU sees there)
    outside: HashMap<u64, (String, String)>,
}

static OOW: Mutex<Option<HashMap<String, OowProg>>> = Mutex::new(None);

fn oow_on() -> bool {
    static W: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *W.get_or_init(|| std::env::var_os("XCL_LOCAL_OOW").is_some())
}

/// What the CPU sees at `a` (or at the CPU view behind it when `a` is in a GPU view).
unsafe fn oow_cpu(a: u64) -> String {
    let mut kind = String::new();
    let mut p = a as usize;
    if let Some(src) = crate::hle::kernelx::mirrors::source(p) {
        kind.push_str(&format!("GPU view of {src:#x}; "));
        p = src;
    } else if crate::hle::kernelx::in_graphics_reservation(p) {
        kind.push_str("title reservation (plain memory); ");
    }
    if crate::gpu::shader::readable(p & !3, 64) {
        let w = std::slice::from_raw_parts((p & !3) as *const u32, 16);
        let nz = w.iter().filter(|&&v| v != 0).count();
        kind.push_str(&format!("CPU: {nz}/16 words non-zero [{:x} {:x} {:x} {:x}]", w[0], w[1], w[2], w[3]));
    } else {
        kind.push_str("CPU: not readable");
    }
    kind
}

pub unsafe fn oow_scan(pso: usize, gfx: bool, layout: Option<&super::console_list::Layout>, args: &[u8]) {
    if !on() || !oow_on() || cap_frames() == 0 || CAP.lock().unwrap().is_none() {
        return;
    }
    let Some(u) = crate::gpu::uma::get() else { return };
    let Some(l) = layout else { return };
    let w32 = |b: &[u8], at: usize| b.get(at..at + 4).map(|x| u32::from_le_bytes(x.try_into().unwrap()));
    let inside_uma = |a: u64| (a >> 32) == 0 || u.window_of(a as usize).is_some();
    let mut hits: Vec<(u64, String)> = Vec::new();
    let mut inside = 0u64;
    // 64-bit-looking pairs in the first 256 bytes the CPU sees at `a` (inside the windows)
    let level2 = |a: u64, from: &str, hits: &mut Vec<(u64, String)>| {
        let p = if (a >> 32) == 0 { u.window(0).0 + a as usize } else { a as usize };
        if !crate::gpu::shader::readable(p & !7, 256) {
            return;
        }
        let q = std::slice::from_raw_parts((p & !7) as *const u64, 32);
        for (k, &v) in q.iter().enumerate() {
            let hi = v >> 32;
            if (1..0x100).contains(&hi) && v & 3 == 0 && !inside_uma(v) {
                hits.push((v, format!("{from} -> +{:#x} (64-bit value)", 8 * k)));
            }
        }
    };
    for (i, p) in l.params.iter().enumerate() {
        match *p {
            super::console_list::Param::Descriptor { at } => {
                let (Some(lo), Some(hi)) = (w32(args, at as usize), w32(args, at as usize + 4)) else { continue };
                let a = lo as u64 | (hi as u64) << 32;
                if a == 0 {
                    continue;
                }
                if inside_uma(a) {
                    inside += 1;
                    level2(a, &format!("root view p{i} {a:#x}"), &mut hits);
                } else {
                    hits.push((a, format!("root view p{i}")));
                }
            }
            super::console_list::Param::Constants { at, count } => {
                for k in 0..count.saturating_sub(1) {
                    let (Some(lo), Some(hi)) = (w32(args, (at + 4 * k) as usize), w32(args, (at + 4 * k + 4) as usize)) else { continue };
                    let a = lo as u64 | (hi as u64) << 32;
                    if (1..0x100).contains(&(hi as u64)) && lo & 3 == 0 && !inside_uma(a) {
                        hits.push((a, format!("root constants p{i} word {k} (64-bit value)")));
                    }
                }
            }
            super::console_list::Param::Table { at } => {
                let Some(t) = w32(args, at as usize) else { continue };
                if t == 0 {
                    continue;
                }
                let base = u.window(0).0 + (t & !3) as usize;
                if !crate::gpu::shader::readable(base, 1024) {
                    continue;
                }
                let words = std::slice::from_raw_parts(base as *const u32, 256);
                for s in 0..64 {
                    let v = &words[4 * s..4 * s + 4];
                    // a buffer V#: type 0 (bits 30-31 of word 3), some DST_SEL, records, a base
                    if v[3] >> 30 != 0 || v[3] & 0xFFF == 0 || v[2] == 0 {
                        continue;
                    }
                    let a = v[0] as u64 | ((v[1] & 0xFFFF) as u64) << 32;
                    if a == 0 {
                        continue;
                    }
                    if inside_uma(a) {
                        inside += 1;
                        let stride = (v[1] >> 16) & 0x3FFF;
                        let limit = if stride == 0 { v[2] as u64 } else { v[2] as u64 * stride as u64 };
                        if limit <= 0x1_0000 {
                            level2(a, &format!("table p{i} slot {s} V# {a:#x}"), &mut hits);
                        }
                    } else {
                        hits.push((a, format!("table p{i} (window 0 +{t:#x}) slot {s} V# records {:#x} word3 {:#x}", v[2], v[3])));
                    }
                }
            }
        }
    }
    let key = shaders(pso);
    let mut g = OOW.lock().unwrap();
    let e = g.get_or_insert_with(HashMap::new).entry(key).or_default();
    e.calls += 1;
    e.gfx = gfx;
    e.inside += inside;
    let rv: Vec<u64> = hits.iter().filter(|h| h.1.starts_with("root view") && !h.1.contains("->")).map(|h| h.0).collect();
    if !rv.is_empty() {
        e.rv_calls += 1;
        e.rv_addrs.extend(rv);
    }
    for (a, from) in hits {
        if e.outside.len() < 64 && !e.outside.contains_key(&a) {
            let cpu = oow_cpu(a);
            e.outside.insert(a, (from, cpu));
        }
    }
}

/// Logs and clears the scan of the capture frame `f`.
fn oow_log(f: u64) {
    let Some(m) = OOW.lock().unwrap().take() else { return };
    let (mut progs, mut with_out) = (0, 0);
    let mut v: Vec<(String, OowProg)> = m.into_iter().collect();
    v.sort_by(|a, b| b.1.outside.len().cmp(&a.1.outside.len()));
    for (k, p) in &v {
        progs += 1;
        if p.outside.is_empty() {
            continue;
        }
        with_out += 1;
        let mut ex: Vec<(&u64, &(String, String))> = p.outside.iter().collect();
        ex.sort();
        let s: Vec<String> = ex.iter().take(6).map(|(a, (w, c))| format!("{a:#x} from {w} ({c})")).collect();
        let mut rvs: Vec<(&u64, &(String, String))> = p.outside.iter().filter(|(_, (w, _))| w.starts_with("root view") && !w.contains("->")).collect();
        rvs.sort();
        let r: Vec<String> = rvs.iter().take(4).map(|(a, (w, c))| format!("{a:#x} from {w} ({c})")).collect();
        crate::xlog_info!("oow: frame {f} {} {k}: {} calls, {} with a root view outside the windows ({} distinct){}; {} addresses inside the windows, {} outside: {}", if p.gfx { "draw" } else { "dispatch" }, p.calls, p.rv_calls, p.rv_addrs.len(), if r.is_empty() { String::new() } else { format!(" [{}]", r.join(" | ")) }, p.inside, p.outside.len(), s.join(" | "));
    }
    crate::xlog_info!("oow: frame {f} summary: {progs} programs scanned, {with_out} given addresses outside the memory windows");
}

// ---------------------------------------------------------------------------------------
// LOCAL experiment, not a fix: `XCL_LOCAL_OOW_COPY=<bytes>`. A root CBV/SRV/UAV address in a GPU
// view of the title's memory outside the windows (`kernelx::mirrors`) is replaced, at record
// time, by a copy of `<bytes>` of the CPU view behind it in unified memory (a 256 MB pool, one
// copy per address and frame, reset at each present). The program then reads the data the CPU
// wrote; its writes there go to the copy (lost). To show whether the data being invisible is
// what empties the chain.
// ---------------------------------------------------------------------------------------

struct CopyPool {
    base: usize,
    next: usize,
    map: HashMap<u64, u64>,
    copied: u64,
    reused: u64,
    full: u64,
    // LOCAL (agent G): the pool in `SEGS` segments, one per frame in turn; a segment is used again
    // only once the presenting queue has passed the present that closed its frame (`seg_fence`).
    // Before: one pool reset at each present while the GPU, behind, still had to run the lists
    // that read it: they read the copies of a later frame (capture R4: 465 of 489 mesh draws' root
    // buffers, positions garbage; `D/mesures/agentG-image-hors-jeu.md`, defect D1).
    seg: usize,
    seg_fence: Vec<u64>,
    fence: Option<ID3D12Fence>,
    value: u64,
    blocked: u64,
    faults: u64,
}

/// LOCAL (agent G): segments of the copy pool (`XCL_LOCAL_OOW_COPY_SEGS`, 4; 1 = the old behaviour,
/// reset at each present without waiting for the GPU)
fn copy_segs() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("XCL_LOCAL_OOW_COPY_SEGS").ok().and_then(|v| v.trim().parse().ok()).filter(|&n: &usize| n >= 1).unwrap_or(4))
}

/// LOCAL (agent G): `src` -> `dst` without faulting when a page of `src` goes away meanwhile (the
/// plain copy crashed runs G1 and E2: `readable` checked, then the title decommitted). False: not copied.
unsafe fn copy_no_fault(src: usize, dst: usize, len: usize) -> bool {
    use windows_sys::Win32::System::Diagnostics::Debug::ReadProcessMemory;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut read = 0usize;
    ReadProcessMemory(GetCurrentProcess(), src as *const c_void, dst as *mut c_void, len, &mut read) != 0 && read == len
}

static COPY_POOL: Mutex<Option<CopyPool>> = Mutex::new(None);
static POOL_BYTES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn copy_bytes() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("XCL_LOCAL_OOW_COPY").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(0))
}

/// At each present: the next segment of the pool is used from its start (and its use logged now
/// and then) – LOCAL (agent G): once the GPU is past the frame that last used it.
unsafe fn oow_copy_reset(queue: *mut c_void, n: u64) {
    if copy_bytes() == 0 {
        return;
    }
    if let Some(p) = COPY_POOL.lock().unwrap().as_mut() {
        if n % 60 == 0 || (p.full > 0 && n % 10 == 0) {
            crate::xlog_info!("oow copy: frame {n}: {} root views copied ({} MB), {} reused, {} not copied (pool full), {} not copied (segment {} still in use by the GPU), {} copies faulted", p.copied, p.next >> 20, p.reused, p.full, p.blocked, p.seg, p.faults);
        }
        let segs = copy_segs();
        if p.seg_fence.len() != segs {
            p.seg_fence = vec![0; segs];
        }
        // the frame just closed: its segment is free once the presenting queue is past this point
        if segs > 1 {
            if let Some(q) = ID3D12CommandQueue::from_raw_borrowed(&queue) {
                if p.fence.is_none() {
                    let mut d: Option<ID3D12Device> = None;
                    if q.GetDevice(&mut d).is_ok() {
                        p.fence = d.and_then(|d| d.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE).ok());
                    }
                }
                if let Some(f) = &p.fence {
                    p.value += 1;
                    if q.Signal(f, p.value).is_ok() {
                        p.seg_fence[p.seg] = p.value;
                    }
                }
            }
        }
        p.seg = (p.seg + 1) % segs;
        p.next = 0;
        p.map.clear();
        p.copied = 0;
        p.reused = 0;
        p.full = 0;
        p.blocked = 0;
        p.faults = 0;
    }
}

/// Replaces GPU-view root view addresses in `args` by copies in unified memory. Returns how many.
pub unsafe fn oow_copy_root_views(layout: Option<&super::console_list::Layout>, args: &mut [u8]) -> u32 {
    // LOCAL (agent Z): one line proving the w26 binary and its switches
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| crate::xlog_info!("LOCAL agentZ w26: mesh set + mirrors; OOW_COPY {} OOW_SCRATCH {} EMPTYV_MIRROR {} ATV_MIRROR {} OOW {}", copy_bytes(), scratch_mb(), mirror_on(), super::real::atv_mirror_on(), oow_on()));
    let n = copy_bytes();
    if n == 0 {
        return 0;
    }
    let (Some(l), Some(u)) = (layout, crate::gpu::uma::get()) else { return 0 };
    let mut g = COPY_POOL.lock().unwrap();
    let p = g.get_or_insert_with(|| CopyPool { base: 0, next: 0, map: HashMap::new(), copied: 0, reused: 0, full: 0, seg: 0, seg_fence: Vec::new(), fence: None, value: 0, blocked: 0, faults: 0 });
    // the pool: retried (256 / 128 / 64 MB) until the coherent window has room, at most once per frame
    if p.base == 0 && p.full == 0 {
        for mb in [256usize, 128, 64] {
            if let Some(b) = super::sparse::alloc(mb << 20, 0x1_0000).map(|q| q.0) {
                if u.is_sparse(b) {
                    p.base = b;
                    POOL_BYTES.store(mb << 20, std::sync::atomic::Ordering::Relaxed);
                    crate::xlog_info!("oow copy: pool of {mb} MB at {b:#x}, {n} bytes per root view");
                    break;
                }
            }
        }
        if p.base == 0 {
            p.full = 1;
            static W: AtomicU64 = AtomicU64::new(0);
            if W.fetch_add(1, Ordering::Relaxed) < 4 {
                crate::xlog_warn!("oow copy: no pool in a coherent window yet");
            }
        }
    }
    if p.base == 0 {
        return 0;
    }
    // LOCAL (agent G): this frame's segment, and whether the GPU still reads what it holds
    let segs = copy_segs();
    let seg_len = (POOL_BYTES.load(std::sync::atomic::Ordering::Relaxed) / segs) & !0xFFFF;
    let seg_base = p.base + p.seg * seg_len;
    let busy = match (&p.fence, p.seg_fence.get(p.seg)) {
        (Some(f), Some(&v)) if v > 0 => f.GetCompletedValue() < v,
        _ => false,
    };
    let mut done = 0;
    for prm in &l.params {
        let super::console_list::Param::Descriptor { at } = *prm else { continue };
        let at = at as usize;
        let Some(b) = args.get(at..at + 8) else { continue };
        let a = u64::from_le_bytes(b.try_into().unwrap());
        if a == 0 || (a >> 32) == 0 || u.window_of(a as usize).is_some() {
            continue;
        }
        let Some(src) = crate::hle::kernelx::mirrors::source(a as usize) else { continue };
        let to = if let Some(&c) = p.map.get(&a) {
            p.reused += 1;
            c
        } else {
            let mut len = n;
            while len >= 0x1000 && !crate::gpu::shader::readable(src, len) {
                len /= 2;
            }
            if busy {
                p.blocked += 1;
                continue;
            }
            if len < 0x1000 || p.next + len > seg_len {
                p.full += 1;
                continue;
            }
            let dst = seg_base + p.next;
            if !copy_no_fault(src, dst, len) {
                p.faults += 1;
                continue;
            }
            p.next += (len + 0xFF) & !0xFF;
            p.copied += 1;
            p.map.insert(a, dst as u64);
            dst as u64
        };
        args[at..at + 8].copy_from_slice(&to.to_le_bytes());
        done += 1;
    }
    done
}

// ---------------------------------------------------------------------------------------
// LOCAL experiment, not a fix: `XCL_LOCAL_OOW_SCRATCH=<MB>`. A root CBV/SRV/UAV address in one of
// the title's GPU-visible reservations that fell back to plain memory (`kernelx` "GPU-visible,
// write-watched") and that the CPU cannot read (GPU-only memory: nothing committed here) is
// moved into unified memory: each 16 MB block of such addresses gets a zeroed block of the
// coherent window on first use, kept for the run (so what one program writes there, the next
// one reads). Addresses the title puts inside buffers or descriptors are NOT moved.
// ---------------------------------------------------------------------------------------

static SCRATCH: Mutex<Option<(HashMap<u64, u64>, u64, u64)>> = Mutex::new(None);

fn scratch_mb() -> u64 {
    static N: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("XCL_LOCAL_OOW_SCRATCH").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(0))
}

pub unsafe fn oow_scratch_root_views(layout: Option<&super::console_list::Layout>, args: &mut [u8]) -> u32 {
    let mb = scratch_mb();
    if mb == 0 {
        return 0;
    }
    let (Some(l), Some(u)) = (layout, crate::gpu::uma::get()) else { return 0 };
    let mut g = SCRATCH.lock().unwrap();
    let s = g.get_or_insert_with(|| (HashMap::new(), 0, 0));
    let mut done = 0;
    for prm in &l.params {
        let super::console_list::Param::Descriptor { at } = *prm else { continue };
        let at = at as usize;
        let Some(b) = args.get(at..at + 8) else { continue };
        let a = u64::from_le_bytes(b.try_into().unwrap());
        if a == 0 || (a >> 32) == 0 || u.window_of(a as usize).is_some() {
            continue;
        }
        if crate::hle::kernelx::mirrors::source(a as usize).is_some() || !crate::hle::kernelx::in_graphics_reservation(a as usize) || crate::gpu::shader::readable(a as usize, 4) {
            continue;
        }
        let blk = a >> 24;
        let base = if let Some(&b) = s.0.get(&blk) {
            b
        } else {
            if (s.0.len() as u64 + 1) * 16 > mb {
                s.2 += 1;
                continue;
            }
            let Some(p) = u.alloc(16 << 20, 0x1_0000).map(|p| p as usize).filter(|&p| u.is_committed(p)).or_else(|| super::sparse::alloc(16 << 20, 0x1_0000).map(|q| q.0)) else {
                s.2 += 1;
                continue;
            };
            std::ptr::write_bytes(p as *mut u8, 0, 16 << 20);
            crate::xlog_info!("oow scratch: block {:#x} (addresses {:#x}..) -> {p:#x} ({} blocks)", blk, blk << 24, s.0.len() + 1);
            s.0.insert(blk, p as u64);
            p as u64
        };
        let to = base + (a & 0xFF_FFFF);
        args[at..at + 8].copy_from_slice(&to.to_le_bytes());
        s.1 += 1;
        done += 1;
    }
    done
}

// ---------------------------------------------------------------------------------------
// LOCAL experiment, not a fix: `XCL_LOCAL_EMPTYV_MIRROR=1`. A buffer view (SRV / UAV / CBV in a
// console heap) whose buffer is not in unified memory gets an empty V# on this base. With the
// switch it gets a V# on a mirror instead: a block of the sparse windows as large as the buffer,
// filled from the title's memory behind it (the CPU view when the buffer is in a GPU view,
// `kernelx::mirrors`), readable 64 KiB chunks only. Mirrors of buffers that only ever get SRVs /
// CBVs are filled again at each present (CPU writes reach them); a mirror with a UAV is filled
// once (the GPU owns it; its writes stay in the mirror and never reach the title's memory).
// ---------------------------------------------------------------------------------------

struct Mirror {
    src: usize,
    len: usize,
    dst: usize,
    uav: bool,
}

static MIRRORS: Mutex<Vec<Mirror>> = Mutex::new(Vec::new());
static MIRROR_STATS: Mutex<(u64, u64, u64)> = Mutex::new((0, 0, 0)); // made, bytes, refused

pub fn mirror_on() -> bool {
    static W: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *W.get_or_init(|| std::env::var_os("XCL_LOCAL_EMPTYV_MIRROR").is_some())
}

/// Copies the readable 64 KiB chunks of `[src, src + len)` to `dst`; returns the bytes copied.
unsafe fn mirror_fill(src: usize, dst: usize, len: usize) -> usize {
    let mut done = 0;
    let mut o = 0;
    while o < len {
        let a = src + o;
        let end = ((a & !0xFFFF) + 0x1_0000).min(src + len);
        let n = end - a;
        if crate::gpu::shader::readable(a, n) {
            std::ptr::copy_nonoverlapping(a as *const u8, (dst + o) as *mut u8, n);
            done += n;
        }
        o += n;
    }
    done
}

/// The mirror address standing for title address `va` (`len` bytes from it), made on first use.
/// LOCAL (agent V): bytes of mirrors made so far.
pub fn mirror_bytes() -> u64 {
    MIRROR_STATS.lock().unwrap().1
}

pub unsafe fn mirror_range(va: usize, len: usize, uav: bool, what: &str) -> Option<usize> {
    if !(mirror_on() || super::real::atv_mirror_on()) || va == 0 || len == 0 { // LOCAL: or agent V's switch
        return None;
    }
    let src = crate::hle::kernelx::mirrors::source(va).unwrap_or(va);
    let mut g = MIRRORS.lock().unwrap();
    if let Some(m) = g.iter_mut().find(|m| src >= m.src && src + len <= m.src + m.len) {
        m.uav |= uav;
        return Some(m.dst + (src - m.src));
    }
    // LOCAL (agent V): ATV mirrors under 1 MiB are packed (256-byte aligned) into 16 MiB blocks
    let atv = what.starts_with("ATV");
    let packed = || -> Option<(usize, usize)> {
        static BUMP: Mutex<(usize, usize)> = Mutex::new((0, 0));
        let mut b = BUMP.lock().unwrap();
        let at = (b.0 + 0xFF) & !0xFF;
        if b.0 == 0 || at + len > b.1 {
            let (d, n) = unsafe { super::sparse::alloc(0x100_0000, 0x1_0000) }?;
            MIRROR_STATS.lock().unwrap().1 += n as u64;
            *b = (d + len, d + n);
            return Some((d, 0));
        }
        b.0 = at + len;
        Some((at, 0))
    };
    let got = if atv && len < 0x10_0000 {
        packed()
    } else {
        super::sparse::alloc(len, 0x1_0000).map(|(d, n)| {
            if atv {
                MIRROR_STATS.lock().unwrap().1 += n as u64;
            }
            (d, n)
        })
    };
    let Some((dst, _)) = got else {
        MIRROR_STATS.lock().unwrap().2 += 1;
        crate::xlog_warn!("emptyv mirror: no sparse room for {len:#x} bytes ({what} at {va:#x})");
        return None;
    };
    let filled = mirror_fill(src, dst, len);
    let kind = if src != va { "GPU view" } else if crate::hle::kernelx::in_graphics_reservation(va) { "title reservation" } else { "other" };
    let mut st = MIRROR_STATS.lock().unwrap();
    st.0 += 1;
    if !atv {
        st.1 += len as u64; // LOCAL: ATV mirrors count the committed blocks above
    }
    if st.0 <= 400 {
        crate::xlog_info!("emptyv mirror #{}: {what} at {va:#x} ({kind}, source {src:#x}) {len:#x} bytes -> {dst:#x}; {filled:#x} bytes readable; total {} MB", st.0, st.1 >> 20);
    }
    g.push(Mirror { src, len, dst, uav });
    Some(dst)
}

/// At each present: SRV/CBV-only mirrors are filled again from the title's memory.
fn mirror_refresh(n: u64) {
    if !(mirror_on() || super::real::atv_mirror_on()) { // LOCAL: or agent V's switch
        return;
    }
    let g = MIRRORS.lock().unwrap();
    let (mut bytes, mut count) = (0usize, 0usize);
    for m in g.iter().filter(|m| !m.uav) {
        bytes += unsafe { mirror_fill(m.src, m.dst, m.len) };
        count += 1;
    }
    static FIRST: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
    if n % 20 == 0 || FIRST.swap(false, Ordering::Relaxed) {
        // what the CPU can read behind every mirror now (UAV ones included)
        let readable: usize = g.iter().map(|m| {
            let mut r = 0;
            let mut o = 0;
            while o < m.len {
                let n = 0x1_0000.min(m.len - o);
                if crate::gpu::shader::readable(m.src + o, n) {
                    r += n;
                }
                o += n;
            }
            r
        }).sum();
        crate::xlog_info!("emptyv mirror: frame {n}: {} MB of the mirrored title ranges readable by the CPU now", readable >> 20);
        let st = *MIRROR_STATS.lock().unwrap();
        crate::xlog_info!("emptyv mirror: frame {n}: {} mirrors ({} MB, {} refused), {} with a UAV; {count} refilled ({} MB)", st.0, st.1 >> 20, st.2, g.iter().filter(|m| m.uav).count(), bytes >> 20);
    }
}

// ---------------------------------------------------------------------------------------
// LOCAL (agent Z): `XCL_LOCAL_FSTRACE=<frames>` (with RTSUM and the target dumps): the frame
// recorded after a dumped present (at most <frames> frames) gets a line per translated direct
// dispatch of at least 30x17 groups (`bind::local_fs_trace`: shader, groups, constants, the
// resources its tables / indices name, with their title addresses); those resources are dumped
// with the targets at the next dumped present. Also a line per target of that frame with its
// title address. Read-only.
// ---------------------------------------------------------------------------------------

static FS_ACTIVE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static FS_DONE: AtomicU64 = AtomicU64::new(0);
static FS_SEQ: AtomicU64 = AtomicU64::new(0);
static FS_RES: Mutex<Vec<usize>> = Mutex::new(Vec::new());

fn fs_frames() -> u64 {
    static N: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("XCL_LOCAL_FSTRACE").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(0))
}

/// A frame is being traced (`bind::translated_dispatch`).
pub fn fs_active() -> bool {
    FS_ACTIVE.load(Ordering::Relaxed)
}

/// The next sequence number of a traced call in this frame.
pub fn fs_seq() -> u64 {
    FS_SEQ.fetch_add(1, Ordering::Relaxed)
}

/// Whether `r` is a render target / depth of the frame being recorded.
pub fn fs_is_target(r: usize) -> bool {
    FRAME.lock().unwrap().as_ref().is_some_and(|f| f.targets.contains_key(&r))
}

/// A texture a traced dispatch names: dumped at the next dumped present (at most 40).
pub fn fs_note(r: usize) {
    let mut g = FS_RES.lock().unwrap();
    let cap = if std::env::var_os("XCL_LOCAL_GPROBE").is_some() { 80 } else { 56 }; // LOCAL agent G
    if g.len() < cap && !g.contains(&r) {
        unsafe { add_ref(r) };
        g.push(r);
    }
}

/// At a present (from `on_present`): close the traced frame (its resources to dump), and arm
/// the next one when this present dumps a world frame (at least 100 indirect draws).
fn fs_present(n: u64, dump: bool, targets: &[usize]) -> Vec<usize> {
    if fs_frames() == 0 {
        return Vec::new();
    }
    let mut out = Vec::new();
    let closing = FS_ACTIVE.swap(false, Ordering::Relaxed);
    unsafe { zsnap_present(n, closing) }; // LOCAL (agent Z)
    if closing {
        for &r in targets.iter().take(40) {
            let d = unsafe { desc_of(r) };
            crate::xlog_info!(
                "fstrace: frame {n} target {r:#x} {:?} title {:x?} / {:x?}",
                d.map(|d| (d.Width, d.Height, d.Format.0, d.Flags.0)),
                super::real::title_address(r),
                super::title_textures::address(r)
            );
        }
        out = std::mem::take(&mut *FS_RES.lock().unwrap());
        crate::xlog_info!("fstrace: frame {n} closed: {} traced calls, {} resources to dump", FS_SEQ.load(Ordering::Relaxed), out.len());
    }
    if dump && FS_DONE.load(Ordering::Relaxed) < fs_frames() {
        FS_DONE.fetch_add(1, Ordering::Relaxed);
        FS_SEQ.store(0, Ordering::Relaxed);
        FS_ACTIVE.store(true, Ordering::Relaxed);
    }
    out
}

// ---------------------------------------------------------------------------------------
// LOCAL (agent Z): GPU snapshots of the watched range (`bind::local_zwatch`) in the traced frame:
// before each compute call that names it, on its list (UAV barrier, window buffer COMMON ->
// COPY_SOURCE -> COMMON), into a readback buffer prefilled with 0xCD; logged two presents after
// the traced frame closes, per list in record order, with the words that changed since the
// previous snapshot of the same list. Read-only for the title data.
// ---------------------------------------------------------------------------------------

const ZSLOT: u64 = 0x60;
const ZMAX: u64 = 16384;

struct ZState {
    rb: ID3D12Resource,
    used: u64,
    recs: Vec<(u64, usize, u64, String)>,
}

static ZS: Mutex<Option<ZState>> = Mutex::new(None);
static ZPENDING: Mutex<Vec<(u64, ZState)>> = Mutex::new(Vec::new());

pub unsafe fn zsnap(list: *mut c_void, at: usize, len: u64, text: String) {
    if !fs_active() {
        return;
    }
    let len = len.min(ZSLOT) & !3;
    let Some((res, off)) = super::real::uma_window_buffer(at, len as usize) else { return };
    if super::list_x::tracked_list_type(list).is_none_or(|t| t == D3D12_COMMAND_LIST_TYPE_BUNDLE) {
        return;
    }
    let Some(l) = ID3D12GraphicsCommandList::from_raw_borrowed(&list) else { return };
    let raw = res as *mut c_void;
    let Some(r) = ID3D12Resource::from_raw_borrowed(&raw) else { return };
    let mut g = ZS.lock().unwrap();
    if g.is_none() {
        let dev_raw = super::real::device_raw();
        let Some(dev) = ID3D12Device::from_raw_borrowed(&dev_raw) else { return };
        let bd = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: ZSLOT * ZMAX,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut rb: Option<ID3D12Resource> = None;
        if dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut rb).is_err() {
            return;
        }
        let Some(rb) = rb else { return };
        let mut m: *mut c_void = std::ptr::null_mut();
        if rb.Map(0, None, Some(&mut m)).is_ok() && !m.is_null() {
            std::ptr::write_bytes(m as *mut u8, 0xCD, (ZSLOT * ZMAX) as usize);
            rb.Unmap(0, None);
        }
        *g = Some(ZState { rb, used: 0, recs: Vec::new() });
    }
    let Some(z) = g.as_mut() else { return };
    if z.used >= ZMAX {
        return;
    }
    let slot = z.used;
    z.used += 1;
    let tr = |a: D3D12_RESOURCE_STATES, b: D3D12_RESOURCE_STATES| D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: std::mem::transmute_copy(r),
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: a,
                StateAfter: b,
            }),
        },
    };
    l.ResourceBarrier(&[D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 { UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER { pResource: std::mem::ManuallyDrop::new(None) }) },
    }]);
    l.ResourceBarrier(&[tr(D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
    l.CopyBufferRegion(&z.rb, slot * ZSLOT, r, off, len);
    l.ResourceBarrier(&[tr(D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON)]);
    z.recs.push((slot, list as usize, len, text));
}

/// At each present: close the snapshots of a traced frame that just closed (`closing`), and log
/// those closed two presents ago.
unsafe fn zsnap_present(n: u64, closing: bool) {
    if closing {
        if let Some(z) = ZS.lock().unwrap().take() {
            ZPENDING.lock().unwrap().push((n, z));
        }
    }
    let due: Vec<(u64, ZState)> = {
        let mut g = ZPENDING.lock().unwrap();
        let (d, keep): (Vec<_>, Vec<_>) = std::mem::take(&mut *g).into_iter().partition(|(f, _)| n >= f + 2);
        *g = keep;
        d
    };
    for (f, z) in due {
        let mut m: *mut c_void = std::ptr::null_mut();
        if z.rb.Map(0, None, Some(&mut m)).is_err() || m.is_null() {
            std::mem::forget(z);
            continue;
        }
        let mut last: HashMap<usize, Vec<u32>> = HashMap::new();
        let mut prev_text: HashMap<usize, String> = HashMap::new(); // LOCAL: the call just before on that list
        for (slot, list, len, text) in &z.recs {
            let prev = prev_text.insert(*list, text.clone()).unwrap_or_default();
            let w: Vec<u32> = std::slice::from_raw_parts((m as *const u8).add((slot * ZSLOT) as usize) as *const u32, (*len / 4) as usize).to_vec();
            let ch: Vec<String> = match last.get(list) {
                Some(p) => w.iter().zip(p.iter()).enumerate().filter(|(_, (a, b))| a != b).map(|(k, (a, b))| format!("[{k}] {b:x}->{a:x}")).collect(),
                None => vec!["first on this list".into()],
            };
            // LOCAL: unchanged snapshots are only counted
            if ch.is_empty() && !text.contains("NAMES IT") {
                last.insert(*list, w);
                continue;
            }
            crate::xlog_info!("zsnap: frame {f} list {list:#x} before {text}: {w:x?} | changed {} | since {prev}", ch.join(" "));
            last.insert(*list, w);
        }
        crate::xlog_info!("zsnap: frame {f} done: {} snapshots on {} lists", z.recs.len(), last.len());
        z.rb.Unmap(0, None);
        std::mem::forget(z); // a list run late must not write freed memory
    }
}

/// LOCAL (agent Z): the byte stride of a command signature (0: unknown).
pub fn stride_of(sig: usize) -> u32 {
    STRIDES.lock().unwrap().as_ref().and_then(|m| m.get(&sig).copied()).unwrap_or(0)
}
