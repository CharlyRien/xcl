//! LOCAL (agent H): `XCL_LOCAL_LCAP=<dir>` – capture of one frame's lighting inputs for the
//! offline replay (`lreplay` test). Needs `XCL_LOCAL_RTSUM=1` (bound targets, shader hashes).
//!
//! After `XCL_LOCAL_LCAP_AFTER_S` (log time, default 125 s): the first translated dispatch of a
//! compute program whose hash starts with `XCL_LOCAL_LCAP_CS` (default `520a7d5c`, the material
//! masks) and then the first run of consecutive translated draws on one list whose vertex shader
//! starts with one of `XCL_LOCAL_LCAP_VS` (default the four tiled-lighting VS). At each: the user
//! data (io), the argument copy, every console descriptor heap the io / arguments point into
//! (GPU bytes), the slots of the decoded used table entries (`XCL_LOCAL_LCAP_SLOTS_*`), the
//! textures their T#s name (all subresources, PC footprints), the buffers their V#s name, the
//! bound targets, the pipelines' console containers and descriptions, the indirect argument
//! records; the compute program's UAV textures again after it, the targets again after the run of
//! draws. Copies go into READBACK buffers on the title's list (one null UAV + aliasing barrier
//! around each batch), written to `<dir>` three presents after the end (fence wait first).
//! Budget `XCL_LOCAL_LCAP_MB` (default 900), per texture `XCL_LOCAL_LCAP_TEX_MB` (default 64).

use std::collections::{HashMap, HashSet};
use std::ffi::c_void;
use std::mem::ManuallyDrop;
use std::sync::{Mutex, OnceLock};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

fn env(k: &str) -> Option<String> {
    std::env::var(k).ok().filter(|v| !v.trim().is_empty())
}

pub fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| env("XCL_LOCAL_LCAP").is_some())
}

fn after_s() -> f64 {
    static A: OnceLock<f64> = OnceLock::new();
    *A.get_or_init(|| env("XCL_LOCAL_LCAP_AFTER_S").and_then(|v| v.trim().parse().ok()).unwrap_or(125.0))
}

fn prefixes(k: &str, default: &str) -> Vec<String> {
    env(k).unwrap_or_else(|| default.into()).split(',').map(|s| s.trim().to_lowercase()).filter(|s| !s.is_empty()).collect()
}

fn vs_set() -> &'static [String] {
    static V: OnceLock<Vec<String>> = OnceLock::new();
    V.get_or_init(|| prefixes("XCL_LOCAL_LCAP_VS", "b6395e96,f0c0ace0,3cfac863,43ab7a0c"))
}

fn cs_set() -> &'static [String] {
    static V: OnceLock<Vec<String>> = OnceLock::new();
    V.get_or_init(|| prefixes("XCL_LOCAL_LCAP_CS", "520a7d5c"))
}

// ---------------------------------------------------------------------------------------
// LOCAL (agent Li): `XCL_LOCAL_LCAP_CS_UAV=<i,j,...>` (shared view indices; needs `XCL_LOCAL_LCAP_SPEC2`,
// the stage file of agent R, `S\agentR\spec2.txt`): the compute program the capture takes is, once armed
// (unchanged: `XCL_LOCAL_LCAP_AFTER_S` / `_AFTER_CS_S` after the mask program), the first one whose
// used-resource table names as a UAV image (kind 3, 8 dwords) a T# whose word 0 is one of these indices,
// instead of the first `XCL_LOCAL_LCAP_CS` program; its table slots come from the same file. Meant for
// the R32_UINT textures behind the R9G9B9E5 light views the tiled-lighting draws read: in L4 / R4 each
// light texture has an R32_UINT read view, an R32_UINT UAV slot and the R9G9B9E5 view, (2520, 2522, 2524),
// (2526, 2528, 2530), (2588, 2590, 2592), (2594, 2596, 2598), (2750, 2752, 2754); the UAV slots' T#s (word 0
// = 2522, 2528, 2590, 2596, 2752) are what a writer's table holds (L4 c1: the table of
// indirect_diffuse_lighting CS_ScreenProbes_TemporalResolve at 0x44d6f1460, s5+160..184). Every compute
// pipeline is noted (hashes, containers) while this is set.
// Whatever the arming, each compute bind that names one is logged (`LOCAL lcap uavwatch`: the first 64,
// then each new program / index), so one run also says which programs write those textures and when.
// An ExecuteIndirect dispatch has no groups here (`groups 0 0 0`): its argument records follow
// (`csargs` line, `cs_args`).
// ---------------------------------------------------------------------------------------
fn uav_watch() -> &'static [u32] {
    static V: OnceLock<Vec<u32>> = OnceLock::new();
    V.get_or_init(|| env("XCL_LOCAL_LCAP_CS_UAV").map(|v| v.split(',').filter_map(|s| s.trim().parse().ok()).collect()).unwrap_or_default())
}

/// the watched indices that compute program `hash`'s used-resource table names as UAV images, as
/// (user-data dword, dword offset, index), and the table slots of that program (for `Cap::inputs`)
unsafe fn uav_watch_hits(hash: u64, io: &[u32]) -> (Vec<(u8, u32, u32)>, Vec<(usize, Vec<u32>)>) {
    let mut hits = Vec::new();
    let mut slots: Vec<(usize, Vec<u32>)> = Vec::new();
    let (Some(s), Some(u)) = (spec2(hash, 6), crate::gpu::uma::get()) else { return (hits, slots) };
    let base = u.base() & !0xFFFF_FFFF;
    for &(dw, off, kind, n) in &s.tables {
        match slots.iter_mut().find(|(k, _)| *k == dw as usize) {
            Some((_, v)) => v.push(off),
            None => slots.push((dw as usize, vec![off])),
        }
        let Some(&t) = io.get(dw as usize) else { continue };
        if kind != 3 || n != 8 || t == 0 {
            continue;
        }
        let p = (base | t as usize) + 4 * off as usize;
        if !crate::gpu::shader::readable(p, 32) {
            continue;
        }
        let w: Vec<u32> = (0..8).map(|k| (p as *const u32).add(k).read_unaligned()).collect();
        if w[3] >> 28 >= 8 && uav_watch().contains(&w[0]) {
            hits.push((dw, off, w[0]));
        }
    }
    (hits, slots)
}

// ---------------------------------------------------------------------------------------
// LOCAL (agent R): geometry capture. `XCL_LOCAL_LCAP_GEOM=<n>`: instead of the lighting stage, the
// first run of at most n consecutive two-pass mesh draws on one list whose bound render targets
// include one of format `XCL_LOCAL_LCAP_GEOM_FMT` (default 29, the sRGB G-buffer target), after
// `XCL_LOCAL_LCAP_GEOM_SKIP` such draws were let through. Per draw: both stages' user data (io),
// argument copies, what the stage's used-resource table names (`XCL_LOCAL_LCAP_SPEC2`, a file made by
// `agentR/mkspec2.py`), the root descriptors' targets, the heap slots that words of the constants and
// of the root descriptors' targets name (guesses, bounded), the records the compute pass wrote and
// the rewritten indirect arguments; the bound targets before the run and after it.
// ---------------------------------------------------------------------------------------
fn geom_draws() -> u32 {
    static A: OnceLock<u32> = OnceLock::new();
    *A.get_or_init(|| env("XCL_LOCAL_LCAP_GEOM").and_then(|v| v.trim().parse().ok()).unwrap_or(0))
}

fn geom_num(k: &str, d: u64) -> u64 {
    env(k).and_then(|v| v.trim().parse().ok()).unwrap_or(d)
}

#[derive(Default)]
struct Spec2 {
    /// (user-data dword of the table pointer, dword offset in the table, kind, descriptor dwords)
    tables: Vec<(u8, u32, u8, u8)>,
    /// root descriptors (64-bit address at the dword and the next)
    roots: Vec<u8>,
    /// root constants (first dword, count)
    consts: Vec<(u8, u8)>,
    heap: Option<u8>,
}

fn spec2(hash: u64, hw: u8) -> Option<&'static Spec2> {
    static S: OnceLock<HashMap<(u64, u8), Spec2>> = OnceLock::new();
    S.get_or_init(|| {
        let mut m = HashMap::new();
        let Some(text) = env("XCL_LOCAL_LCAP_SPEC2").and_then(|f| std::fs::read_to_string(f).ok()) else { return m };
        for line in text.lines() {
            let mut it = line.split_whitespace();
            let (Some(h), Some(hw)) = (it.next().and_then(|h| u64::from_str_radix(h, 16).ok()), it.next().and_then(|h| h.parse::<u8>().ok())) else { continue };
            let mut s = Spec2::default();
            for t in it {
                let p: Vec<&str> = t.split(':').collect();
                let n = |k: usize| p.get(k).and_then(|v| v.parse::<u32>().ok()).unwrap_or(0);
                match p[0] {
                    "t" => s.tables.push((n(1) as u8, n(2), n(3) as u8, n(4) as u8)),
                    "r" => s.roots.push(n(1) as u8),
                    "c" => s.consts.push((n(1) as u8, n(2) as u8)),
                    "h" => s.heap = Some(n(1) as u8),
                    _ => {}
                }
            }
            m.insert((h, hw), s);
        }
        crate::xlog_info!("LOCAL lcap (geometry): {} stage specs loaded", m.len());
        m
    })
    .get(&(hash, hw))
}

// ---------------------------------------------------------------------------------------
// LOCAL (agent R): `XCL_LOCAL_OOW_DEMAND=1` (with `XCL_LOCAL_LCAP_SPEC2=<file>`, no capture needed): readable
// copies ON DEMAND of the buffers a two-pass mesh draw's stages name in their descriptor tables when the
// buffer lies in a GPU view of title memory (outside the memory windows, where a translated shader reads
// zero). Found off game (captures R2 / R3): 92 of 168 mesh draws of a frame wrote one vertex per record
// because one buffer of their mesh table (the meshlets) had no copy - the mirrors made at view creation
// (`XCL_LOCAL_ATV_MIRROR`) stop at `XCL_LOCAL_ATV_CAP_MB` (500 MB of the 1030 MB the title makes views of),
// while one frame's draws name 350 such buffers, 2 MB in all. At the bind of each stage: every used table
// entry (the stage's used-resource table, from the spec file) that is a buffer descriptor whose address is
// in a GPU view gets a mirror (`rtsum::mirror_range`, refilled at each present like the others) and the
// descriptor in the title's table is rewritten to point at it, as `real::atv_view` does at creation.
// ---------------------------------------------------------------------------------------
static DEMAND_PSO: Mutex<Option<HashMap<usize, [u64; 2]>>> = Mutex::new(None);
static DEMAND_STATS: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

pub fn demand_on() -> bool {
    static A: OnceLock<bool> = OnceLock::new();
    *A.get_or_init(|| env("XCL_LOCAL_OOW_DEMAND").is_some())
}

pub fn demand_note(pso: usize, ms: &[u8], ps: Option<&[u8]>) {
    if demand_on() && pso != 0 {
        DEMAND_PSO.lock().unwrap().get_or_insert_with(HashMap::new).insert(pso, [crate::gpu::shader::fnv64(ms), ps.map_or(0, crate::gpu::shader::fnv64)]);
    }
}

/// a two-pass mesh stage of `pso` is about to be bound with the user data `io`
pub unsafe fn demand_at_bind(pso: usize, compute: bool, io: &[u32]) {
    use std::sync::atomic::Ordering::Relaxed;
    if !demand_on() {
        return;
    }
    let real = super::pso_async::resolve(pso);
    let Some(h) = DEMAND_PSO.lock().unwrap().as_ref().and_then(|m| [pso, real].iter().find_map(|p| m.get(p).copied())) else { return };
    let (hash, hw) = if compute { (h[0], 7u8) } else { (h[1], 5u8) };
    let Some(sp) = spec2(hash, hw) else { return };
    let Some(u) = crate::gpu::uma::get() else { return };
    let base = u.base() & !0xFFFF_FFFF;
    for &(dw, off, _, _) in &sp.tables {
        let sg = if hw == 7 { dw as usize + super::console_list::first_sgpr_ms() - 8 } else { dw as usize };
        let t = io.get(sg).copied().unwrap_or(0);
        if t == 0 {
            continue;
        }
        let p = (base | t as usize) + 4 * off as usize;
        if !crate::gpu::shader::readable(p, 16) {
            continue;
        }
        let w: [u32; 4] = std::array::from_fn(|k| (p as *const u32).add(k).read_unaligned());
        if w[3] >> 28 >= 8 || w[3] == 0 {
            continue;
        }
        let a = (w[0] as u64 | ((w[1] & 0xFFFF) as u64) << 32) as usize;
        if a >> 32 == 0 || crate::hle::kernelx::mirrors::gpu_view_of(a).is_none() {
            continue;
        }
        let stride = ((w[1] >> 16) & 0x3FFF) as usize;
        let len = if stride == 0 { w[2] as usize } else { w[2] as usize * stride };
        DEMAND_STATS[0].fetch_add(1, Relaxed);
        if len == 0 || len > (geom_num("XCL_LOCAL_OOW_DEMAND_MAX_MB", 64) << 20) as usize {
            DEMAND_STATS[3].fetch_add(1, Relaxed);
            continue;
        }
        match super::rtsum::mirror_range(a, len, false, "ATV demand") {
            Some(dst) => {
                (p as *mut u32).write_unaligned(dst as u32);
                (p as *mut u32).add(1).write_unaligned((w[1] & 0xFFFF_0000) | ((dst >> 32) as u32 & 0xFFFF));
                let n = DEMAND_STATS[1].fetch_add(1, Relaxed) + 1;
                let b = DEMAND_STATS[2].fetch_add(len as u64, Relaxed) + len as u64;
                if n <= 4 || n.is_power_of_two() {
                    crate::xlog_info!("LOCAL oow demand: {n} buffers outside the windows given a copy at a bind ({} KiB in all; last: {a:#x}+{len:#x} of the {} table at {p:#x} -> {dst:#x}); {} refused", b >> 10, if compute { "mesh" } else { "pixel" }, DEMAND_STATS[3].load(Relaxed));
                }
            }
            None => {
                DEMAND_STATS[3].fetch_add(1, Relaxed);
            }
        }
    }
}

/// (user SGPR, dword offsets in its table) of the used table entries (agent S's decoder, union of
/// the TL VS / PS permutations and of CS_GenerateMaterialMasks)
fn slot_spec(gfx: bool) -> &'static [(usize, Vec<u32>)] {
    static G: OnceLock<Vec<(usize, Vec<u32>)>> = OnceLock::new();
    static C: OnceLock<Vec<(usize, Vec<u32>)>> = OnceLock::new();
    let parse = |s: String| -> Vec<(usize, Vec<u32>)> {
        s.split(';')
            .filter_map(|p| {
                let (k, v) = p.split_once('=')?;
                Some((k.trim().parse().ok()?, v.split(',').filter_map(|o| o.trim().parse().ok()).collect()))
            })
            .collect()
    };
    if gfx {
        G.get_or_init(|| parse(env("XCL_LOCAL_LCAP_SLOTS_GFX").unwrap_or_else(|| "8=0,8,16,32,56,72,80,104,120,168,176,192,216,224,232,240,248,264,312,328,336,352,360,368,400,432,440,456,472,488,536,552,576,584,600,624,664,672,744,752,760,784,800,808,856,872,880,920,928,936,968,976,992,1000,1008,1016,1056,1064,1072,1080,1096,1112,1160,1168,1200,1216,1224,1240,1248,1264,1280,1312,1392,1400,1416,1424,1440,1448;9=0,8,16,24,32,40,48,56,64,72,80,88,96".into())))
    } else {
        C.get_or_init(|| parse(env("XCL_LOCAL_LCAP_SLOTS_CS").unwrap_or_else(|| "4=16,104,120,176,192,248,336,368,416,432,488,584,664,752,872,880,992,1000,1016,1168,1264,1312,1376,1392,1416,1424;5=0,8,16,24,32,40,48,56,64".into())))
    }
}

fn hexbytes(p: *const u8, n: usize) -> String {
    if p.is_null() {
        return "none".into();
    }
    (0..n).map(|k| format!("{:02x}", unsafe { *p.add(k) })).collect()
}

// ---------------------------------------------------------------------------------------
// views: which resource / PC view description each shared-heap index holds
// ---------------------------------------------------------------------------------------

/// compact (the map holds every view the title makes from the start: no strings)
#[derive(Clone, Copy)]
struct View {
    res: usize,
    uav: bool,
    has: bool,
    desc: [u8; 40],
}

impl View {
    fn desc_hex(&self) -> String {
        if self.has { hexbytes(self.desc.as_ptr(), 40) } else { "none".into() }
    }
}

#[derive(Default)]
struct Views {
    by_handle: HashMap<u64, View>,
    by_index: HashMap<u32, u64>,
    samp_by_handle: HashMap<u64, String>,
    samp_by_index: HashMap<u32, u64>,
}

static VIEWS: Mutex<Option<Views>> = Mutex::new(None);

/// A view was made at PC CPU handle `handle` (`index`: its shared-heap index, None: the heap is
/// not in the shared heap) of `res` (null: a null view) with `desc` (`len` bytes, null: none).
pub unsafe fn note_view(handle: u64, index: Option<u32>, res: *mut c_void, uav: bool, desc: *const u8, len: usize) {
    if !on() {
        return;
    }
    let mut d = [0u8; 40];
    if !desc.is_null() {
        std::ptr::copy_nonoverlapping(desc, d.as_mut_ptr(), len.min(40));
    }
    let v = View { res: res as usize, uav, has: !desc.is_null(), desc: d };
    let mut g = VIEWS.lock().unwrap();
    let m = g.get_or_insert_with(Views::default);
    m.by_handle.insert(handle, v);
    if let Some(i) = index {
        m.by_index.insert(i, handle);
    }
}

/// Console view slots copied: (dst srv, dst uav, dst shared index, src srv, src uav)
pub fn note_view_copy(d_srv: u64, d_uav: u64, d_index: Option<u32>, s_srv: u64, s_uav: u64) {
    if !on() {
        return;
    }
    let mut g = VIEWS.lock().unwrap();
    let m = g.get_or_insert_with(Views::default);
    for (d, s) in [(d_srv, s_srv), (d_uav, s_uav)] {
        match m.by_handle.get(&s).copied() {
            Some(v) => {
                m.by_handle.insert(d, v);
            }
            None => {
                m.by_handle.remove(&d);
            }
        }
    }
    if let Some(i) = d_index {
        m.by_index.insert(i, d_srv);
        m.by_index.insert(i + 1, d_uav);
    }
}

pub unsafe fn note_sampler(handle: u64, index: u32, desc: *const D3D12_SAMPLER_DESC) {
    if !on() {
        return;
    }
    let mut g = VIEWS.lock().unwrap();
    let m = g.get_or_insert_with(Views::default);
    m.samp_by_handle.insert(handle, hexbytes(desc as *const u8, std::mem::size_of::<D3D12_SAMPLER_DESC>()));
    m.samp_by_index.insert(index & 0xFFF, handle);
}

pub fn note_sampler_copy(d: u64, d_index: u32, s: u64) {
    if !on() {
        return;
    }
    let mut g = VIEWS.lock().unwrap();
    let m = g.get_or_insert_with(Views::default);
    if let Some(v) = m.samp_by_handle.get(&s).cloned() {
        m.samp_by_handle.insert(d, v);
    }
    m.samp_by_index.insert(d_index & 0xFFF, d);
}

/// SRV / UAV-handle (or index + 1) view of a shared-heap index (the UAV is at index + 1)
fn view_of(index: u32) -> Option<View> {
    let g = VIEWS.lock().unwrap();
    let m = g.as_ref()?;
    m.by_handle.get(m.by_index.get(&index)?).copied()
}

// ---------------------------------------------------------------------------------------
// pipelines: the descriptions and containers of the pipelines of interest
// ---------------------------------------------------------------------------------------

struct SavedPso {
    gfx: bool,
    /// LOCAL (agent R): a two-pass mesh pipeline (blobs: MS, PS; desc: its state sub-objects) and its record text
    mesh: bool,
    extra: String,
    hashes: [u64; 2],
    desc: String,
    blobs: Vec<Vec<u8>>,
}

static PSOS: Mutex<Option<HashMap<usize, SavedPso>>> = Mutex::new(None);

unsafe fn blob(b: &D3D12_SHADER_BYTECODE) -> Option<Vec<u8>> {
    let (p, n) = (b.pShaderBytecode as usize, b.BytecodeLength);
    (p != 0 && n != 0 && crate::gpu::shader::readable(p, n)).then(|| std::slice::from_raw_parts(p as *const u8, n).to_vec())
}

fn matches(h: u64, set: &[String]) -> bool {
    let s = format!("{h:016x}");
    set.iter().any(|p| s.starts_with(p.as_str()))
}

pub unsafe fn note_graphics(pso: usize, desc: *const c_void) {
    if !on() || pso == 0 || !crate::gpu::shader::readable(desc as usize, std::mem::size_of::<D3D12_GRAPHICS_PIPELINE_STATE_DESC>()) {
        return;
    }
    let d = &*(desc as *const D3D12_GRAPHICS_PIPELINE_STATE_DESC);
    let (Some(vs), ps) = (blob(&d.VS), blob(&d.PS)) else { return };
    let hv = crate::gpu::shader::fnv64(&vs);
    if !matches(hv, vs_set()) {
        return;
    }
    let hp = ps.as_deref().map_or(0, crate::gpu::shader::fnv64);
    let s = SavedPso { mesh: false, extra: String::new(), gfx: true, hashes: [hv, hp], desc: hexbytes(desc as *const u8, std::mem::size_of::<D3D12_GRAPHICS_PIPELINE_STATE_DESC>()), blobs: [Some(vs), ps].into_iter().flatten().collect() };
    let n = {
        let mut g = PSOS.lock().unwrap();
        let m = g.get_or_insert_with(HashMap::new);
        m.insert(pso, s);
        m.len()
    };
    if n == 1 || n == 10 || n == 100 {
        crate::xlog_info!("LOCAL lcap: {n} pipelines of interest seen (last vs {hv:016x} ps {hp:016x} as {pso:#x})");
    }
}

pub unsafe fn note_compute(pso: usize, desc: *const c_void) {
    if !on() || pso == 0 || !crate::gpu::shader::readable(desc as usize, std::mem::size_of::<D3D12_COMPUTE_PIPELINE_STATE_DESC>()) {
        return;
    }
    let d = &*(desc as *const D3D12_COMPUTE_PIPELINE_STATE_DESC);
    let Some(cs) = blob(&d.CS) else { return };
    let h = crate::gpu::shader::fnv64(&cs);
    // LOCAL (agent Li): with `XCL_LOCAL_LCAP_CS_UAV`, every compute pipeline (the watch needs their hashes;
    // the title's compute containers are about 16 MB in all)
    if !matches(h, cs_set()) && uav_watch().is_empty() {
        return;
    }
    let s = SavedPso { mesh: false, extra: String::new(), gfx: false, hashes: [h, 0], desc: hexbytes(desc as *const u8, std::mem::size_of::<D3D12_COMPUTE_PIPELINE_STATE_DESC>()), blobs: vec![cs] };
    PSOS.lock().unwrap().get_or_insert_with(HashMap::new).insert(pso, s);
}

/// LOCAL (agent R): a two-pass mesh pipeline `pso` was made from the console MS `ms` and PS `ps`;
/// `state`: the stream's other sub-objects as given to the driver (PC formats), `extra`: the record's shape
pub fn note_mesh(pso: usize, ms: &[u8], ps: Option<&[u8]>, state: &[u8], extra: String) {
    if !on() || geom_draws() == 0 || pso == 0 {
        return;
    }
    let hashes = [crate::gpu::shader::fnv64(ms), ps.map_or(0, crate::gpu::shader::fnv64)];
    let s = SavedPso { mesh: true, extra, gfx: true, hashes, desc: hexbytes(state.as_ptr(), state.len()), blobs: [Some(ms.to_vec()), ps.map(|p| p.to_vec())].into_iter().flatten().collect() };
    let n = {
        let mut g = PSOS.lock().unwrap();
        let m = g.get_or_insert_with(HashMap::new);
        m.insert(pso, s);
        m.len()
    };
    if n == 1 || n == 100 || n == 1000 {
        crate::xlog_info!("LOCAL lcap (geometry): {n} mesh pipelines noted (last ms {:016x} ps {:016x} as {pso:#x})", hashes[0], hashes[1]);
    }
}

/// LOCAL (agent R): (MS hash, PS hash) of a noted two-pass mesh pipeline
fn saved_mesh(pso: usize) -> Option<[u64; 2]> {
    let real = super::pso_async::resolve(pso);
    let g = PSOS.lock().unwrap();
    let m = g.as_ref()?;
    [pso, real].iter().find_map(|p| m.get(p)).filter(|s| s.mesh).map(|s| s.hashes)
}

fn saved_hashes(pso: usize) -> Option<(bool, [u64; 2])> {
    let real = super::pso_async::resolve(pso);
    let g = PSOS.lock().unwrap();
    let m = g.as_ref()?;
    [pso, real].iter().find_map(|p| m.get(p)).map(|s| (s.gfx, s.hashes))
}

// ---------------------------------------------------------------------------------------
// the capture
// ---------------------------------------------------------------------------------------

struct Chunk {
    rb: Option<ID3D12Resource>,
    size: u64,
    used: u64,
    /// LOCAL (agent L): 0 = the compute stage's copies, 1 = the draws'; a chunk holds one stage only, so a
    /// stage whose list has run can be written out alone
    stage: u8,
    written: bool,
}

#[derive(Default, PartialEq, Clone, Copy)]
enum St {
    #[default]
    Idle,
    Active,
    Done,
}

#[derive(Default)]
struct Cap {
    chunks: Vec<Chunk>,
    bytes: u64,
    man: Vec<String>,
    cpu: Vec<u8>,
    res_ids: HashMap<usize, u32>,
    tex_done: HashSet<(u32, &'static str)>,
    ranges_done: HashSet<(u64, &'static str)>,
    views_done: HashSet<u32>,
    psos_done: HashMap<usize, u32>,
    cs: St,
    cs_list: usize,
    cs_uav_res: Vec<usize>,
    cs_groups: [u32; 3],
    /// LOCAL (agent Li): the table slots of the compute program chosen by `XCL_LOCAL_LCAP_CS_UAV`
    cs_spec: Option<Vec<(usize, Vec<u32>)>>,
    tl: St,
    tl_list: usize,
    tl_targets: Vec<usize>,
    draws: u32,
    end_presents: u32,
    written: bool,
    skipped: u64,
    note_frame: u64,
    // LOCAL (agent L): staged writes and repeated captures
    stage_now: u8,
    index: u32,
    cs_exec: bool,
    tl_exec: bool,
    cs_flushed: bool,
    cs_presents: u32,
    man_written: usize,
    t_first: f64,
    // LOCAL (agent R): geometry capture
    /// the open draw of each list (its number), between its compute stage and its mesh stage
    geom_pending: HashMap<usize, u32>,
    geom_lists: HashSet<usize>,
    /// a present was seen since the capture was armed (the capture starts at a frame's first draw)
    geom_fresh: bool,
    geom_presents: u32,
    geom_over: u32,
    geom_foreign: u32,
    guess_bytes: u64,
    reserve: u64,
    trace: HashMap<usize, String>,
    /// LOCAL (agent G): root-descriptor ranges whose CPU bytes were taken at record time: read again
    /// once the GPU ran the captured lists (`cpu_after.bin`)
    cpu_roots: Vec<(usize, usize)>,
    /// LOCAL (agent M): bytes of `timg` title images put in `cpu` so far
    title_img_bytes: usize,
}

/// LOCAL (agent G): `XCL_LOCAL_LCAP_ROOT_CPU=0` leaves out the CPU bytes of the root descriptors'
/// targets (taken by default: the GPU's copy of them differed from the CPU's in capture R4)
fn root_cpu() -> bool {
    env("XCL_LOCAL_LCAP_ROOT_CPU").is_none_or(|v| v != "0")
}

/// LOCAL (agent L): capture `k` goes to `<XCL_LOCAL_LCAP>/c<k>`
fn cap_dir(k: u32) -> std::path::PathBuf {
    std::path::PathBuf::from(env("XCL_LOCAL_LCAP").unwrap()).join(format!("c{k}"))
}

/// LOCAL (agent L): `XCL_LOCAL_LCAP_REPEAT=<n>` captures (default 1), the next one armed
/// `XCL_LOCAL_LCAP_GAP_S` (default 6) after the previous one is on disk
fn repeat() -> u32 {
    env("XCL_LOCAL_LCAP_REPEAT").and_then(|v| v.parse().ok()).unwrap_or(1)
}
static NEXT_ARM_MS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);


static CAP: Mutex<Option<Cap>> = Mutex::new(None);
// LOCAL (agent L): per thread (a global was overwritten by another thread's dispatch between the note and the bind: run L3)
thread_local! {
    static GROUPS: std::cell::Cell<[u32; 3]> = const { std::cell::Cell::new([0; 3]) };
}
static PRESENTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn budget() -> u64 {
    env("XCL_LOCAL_LCAP_MB").and_then(|v| v.parse().ok()).unwrap_or(900u64) << 20
}

fn tex_max() -> u64 {
    env("XCL_LOCAL_LCAP_TEX_MB").and_then(|v| v.parse().ok()).unwrap_or(64u64) << 20
}

fn max_draws() -> u32 {
    env("XCL_LOCAL_LCAP_DRAWS").and_then(|v| v.parse().ok()).unwrap_or(400)
}

unsafe fn device() -> Option<ID3D12Device> {
    let raw = super::real::device_raw();
    ID3D12Device::from_raw_borrowed(&raw).cloned()
}

impl Cap {
    /// `n` bytes (aligned 512) in a readback chunk: (chunk, offset)
    unsafe fn alloc(&mut self, n: u64) -> Option<(usize, u64)> {
        // LOCAL (agent R): `reserve` is kept for the targets copied at the end of a geometry capture
        if self.bytes + n > budget().saturating_sub(self.reserve) {
            self.skipped += 1;
            return None;
        }
        let need = (n + 511) & !511;
        let stage = self.stage_now;
        if let Some((k, c)) = self.chunks.iter_mut().enumerate().last().filter(|(_, c)| !c.written && c.stage == stage && c.used + need <= c.size) {
            let at = c.used;
            c.used += need;
            self.bytes += need;
            return Some((k, at));
        }
        let size = need.max(64 << 20);
        let bd = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let dev = device()?;
        let mut rb: Option<ID3D12Resource> = None;
        // VRAM, not process commit (run h1 was killed by the memory guard with READBACK chunks); read back
        // in 64 MB pieces at write time
        let bd = D3D12_RESOURCE_DESC { Flags: D3D12_RESOURCE_FLAG_NONE, ..bd };
        dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut rb).ok()?;
        self.chunks.push(Chunk { rb: Some(rb?), size, used: need, stage, written: false });
        self.bytes += need;
        Some((self.chunks.len() - 1, 0))
    }

    fn res_id(&mut self, r: usize) -> u32 {
        let n = self.res_ids.len() as u32;
        *self.res_ids.entry(r).or_insert(n)
    }

    /// GPU bytes of UMA range [p, p + n) (window buffer copy) as range `name`
    unsafe fn range(&mut self, l: &ID3D12GraphicsCommandList, name: &str, p: usize, n: usize, phase: &'static str, cpu: bool) {
        if n == 0 || !self.ranges_done.insert((p as u64, phase)) {
            return;
        }
        let Some((wbuf, off)) = super::real::uma_window_buffer(p, n) else {
            self.man.push(format!("norange {name} ea {p:#x} bytes {n:#x} (not in a window)"));
            return;
        };
        let k = crate::gpu::uma::get().and_then(|u| u.window_of(p)).unwrap_or(9);
        let raw = wbuf as *mut c_void;
        let Some(w) = ID3D12Resource::from_raw_borrowed(&raw) else { return };
        let Some((c, at)) = self.alloc(n as u64) else {
            self.man.push(format!("skip range {name} ea {p:#x} bytes {n:#x} (budget)"));
            return;
        };
        l.CopyBufferRegion(self.chunks[c].rb.as_ref().unwrap(), at, w, off, n as u64);
        let mut line = format!("range {name} ea {p:#x} window {k} offset {off:#x} bytes {n:#x} phase {phase} file chunk{c}.bin@{at:#x}");
        // (LOCAL agent G: a budget for the CPU copies, `XCL_LOCAL_LCAP_CPU_MB`, 1024 by default)
        if cpu && self.cpu.len() + n <= (geom_num("XCL_LOCAL_LCAP_CPU_MB", 1024) << 20) as usize && crate::gpu::shader::readable(p, n) {
            if name.starts_with("root@") {
                self.cpu_roots.push((p, n));
            }
            let o = self.cpu.len();
            self.cpu.extend_from_slice(std::slice::from_raw_parts(p as *const u8, n));
            line += &format!(" cpu cpu.bin@{o:#x} cpu_bytes {n:#x}");
        }
        self.man.push(line);
    }

    /// all subresources of texture `r` (phase `phase`)
    unsafe fn tex(&mut self, l: &ID3D12GraphicsCommandList, direct: bool, r: usize, phase: &'static str, why: &str) {
        if r == 0 {
            return;
        }
        let id = self.res_id(r);
        if !self.tex_done.insert((id, phase)) {
            return;
        }
        let raw = r as *mut c_void;
        let Some(res) = ID3D12Resource::from_raw_borrowed(&raw) else { return };
        let d = res.GetDesc();
        let dhex = hexbytes(&d as *const _ as *const u8, std::mem::size_of::<D3D12_RESOURCE_DESC>());
        // LOCAL (agent R): the title's own bytes of a texture kept in step (first 64 KiB at its address,
        // through the CPU view when it lies in a GPU view): non-zero bytes / bytes looked at
        let title = if geom_draws() == 0 {
            String::new()
        } else {
            match super::title_textures::address(r) {
                None => " title none".to_string(),
                Some(va) => {
                    let src = crate::hle::kernelx::mirrors::source(va).unwrap_or(va);
                    let m = 0x1_0000usize;
                    // LOCAL (agent G): also the address XCL really reads the texture from
                    // (`title_textures::local_source`: the GPU view itself when a physical page is
                    // mapped there), whether a page is mapped at the GPU view, and its bytes
                    let nz = |p: usize| if crate::gpu::shader::readable(p, m) { format!("{}/{m}", std::slice::from_raw_parts(p as *const u8, m).iter().filter(|x| **x != 0).count()) } else { "unreadable".to_string() };
                    let used = super::title_textures::local_source(r);
                    let more = format!(" xcl_src {} at_va {} phys_at_va {} pending {}", used.map_or("none".to_string(), |u| format!("{u:#x} {}", nz(u))), nz(va), crate::hle::kernelx::physical_page_at(va), super::title_textures::local_pending(r));
                    if crate::gpu::shader::readable(src, m) {
                        let b = std::slice::from_raw_parts(src as *const u8, m);
                        format!(" title {va:#x} cpu_nz {}/{m}{more}", b.iter().filter(|x| **x != 0).count())
                    } else {
                        format!(" title {va:#x} cpu_unreadable{more}")
                    }
                }
            }
        };
        let head = format!("tex {id} {phase} res {r:#x} ({}x{}x{} f{} mips {} flags {:#x} samples {}) why {why} evicted {}{title} desc {dhex}", d.Width, d.Height, d.DepthOrArraySize, d.Format.0, d.MipLevels, d.Flags.0, d.SampleDesc.Count, super::vram::is_evicted(r));
        if d.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER || d.SampleDesc.Count > 1 {
            self.man.push(format!("{head} skipped (buffer or MSAA)"));
            return;
        }
        let depth = d.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL.0 != 0;
        if depth && !direct {
            self.man.push(format!("{head} skipped (depth on a non-direct list)"));
            return;
        }
        let planes = if matches!(d.Format.0, 19 | 20 | 44 | 45) { 2 } else { 1 };
        let arr = if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D { 1 } else { d.DepthOrArraySize as u32 };
        let nsub = d.MipLevels as u32 * arr * planes;
        let Some(dev) = device() else { return };
        let mut fps = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); nsub as usize];
        let mut total = 0u64;
        // (LOCAL agent M: the rows too, for the title's image below)
        let mut rows = vec![0u32; nsub as usize];
        dev.GetCopyableFootprints(&d, 0, nsub, 0, Some(fps.as_mut_ptr()), Some(rows.as_mut_ptr()), None, Some(&mut total));
        if total == 0 || total == u64::MAX || total > tex_max() {
            self.man.push(format!("{head} skipped ({total} bytes)"));
            return;
        }
        let Some((c, at)) = self.alloc(total) else {
            self.man.push(format!("{head} skipped (budget)"));
            return;
        };
        for (k, fp) in fps.iter().enumerate() {
            let mut f = *fp;
            f.Offset += at;
            let dst = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(self.chunks[c].rb.as_ref().unwrap()), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: f } };
            let src = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(res), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: k as u32 } };
            l.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
        }
        self.man.push(format!("{head} nsub {nsub} total {total:#x} file chunk{c}.bin@{at:#x}"));
        // LOCAL (agent M): where XCL reads this texture from and what the page-mapping method's
        // record gives (`tsrc`), and those bytes detiled in the footprints above (`timg`, in
        // cpu.bin; `XCL_LOCAL_LCAP_TITLE_MB`, 512, bounds them): the texture XCL would upload
        // with #1102, for `tex<ID>=file:` in the replay
        if geom_draws() != 0 {
            if let Some((desc, img)) = super::title_textures::local_title_image(r, &fps, &rows, total as usize) {
                self.man.push(format!("tsrc {id} {phase} {desc}"));
                if let Some(img) = img {
                    let used = self.title_img_bytes;
                    if used + img.len() <= (geom_num("XCL_LOCAL_LCAP_TITLE_MB", 512) << 20) as usize {
                        let o = self.cpu.len();
                        self.cpu.extend_from_slice(&img);
                        self.title_img_bytes += img.len();
                        self.man.push(format!("timg {id} {phase} cpu cpu.bin@{o:#x} bytes {:#x}", img.len()));
                    } else {
                        self.man.push(format!("timg {id} {phase} skipped (budget)"));
                    }
                }
            }
        }
    }

    /// the view at shared index `i` (and its UAV at i + 1): lines + texture copies
    unsafe fn image(&mut self, l: &ID3D12GraphicsCommandList, direct: bool, i: u32, phase: &'static str, uavs: &mut Vec<usize>) {
        for (idx, kind) in [(i, "srv"), (i + 1, "uav")] {
            let Some(v) = view_of(idx) else {
                if self.views_done.insert(idx) {
                    self.man.push(format!("view {idx} {kind} unknown"));
                }
                continue;
            };
            if v.res != 0 {
                if v.uav {
                    uavs.push(v.res);
                }
                self.tex(l, direct, v.res, phase, &format!("view{idx}"));
                // LOCAL (agent Li): a format-alias reader: its writers (line + texels), for `alias<W>=<R>` in the replay
                if !v.uav {
                    for (w, what) in super::format_alias::local_writers_of(v.res) {
                        self.tex(l, direct, w, phase, &format!("alias-writer-of-view{idx}"));
                        let line = format!("alias reader {} view {idx} writer {} {what}", self.res_id(v.res), self.res_id(w));
                        if !self.man.contains(&line) {
                            self.man.push(line);
                        }
                    }
                }
            }
            if self.views_done.insert(idx) {
                let id = if v.res == 0 { u32::MAX } else { self.res_id(v.res) };
                self.man.push(format!("view {idx} {} res {} desc {}", if v.uav { "uav" } else { "srv" }, if v.res == 0 { "null".to_string() } else { format!("{id}") }, v.desc_hex()));
            }
        }
    }

    unsafe fn pso(&mut self, pso: usize) -> u32 {
        if let Some(&id) = self.psos_done.get(&pso) {
            return id;
        }
        let id = self.psos_done.len() as u32;
        self.psos_done.insert(pso, id);
        let real = super::pso_async::resolve(pso);
        let g = PSOS.lock().unwrap();
        let s = g.as_ref().and_then(|m| [pso, real].iter().find_map(|p| m.get(p)));
        match s {
            Some(s) => {
                let dir = cap_dir(self.index);
                let _ = std::fs::create_dir_all(&dir);
                let names: Vec<String> = s.blobs.iter().enumerate().map(|(k, b)| {
                    let n = format!("pso{id}_{}.bin", if !s.gfx { "cs" } else if k == 0 { if s.mesh { "ms" } else { "vs" } } else { "ps" });
                    let _ = std::fs::write(dir.join(&n), b);
                    n
                }).collect();
                self.man.push(format!("pso {id} {} {:016x} {:016x} files {} desc {}{}", if s.mesh { "mesh" } else if s.gfx { "gfx" } else { "cs" }, s.hashes[0], s.hashes[1], names.join(","), if s.desc.is_empty() { "none" } else { &s.desc }, s.extra));
            }
            None => self.man.push(format!("pso {id} unknown {pso:#x}")),
        }
        id
    }

    /// everything one bind reads: heaps, argument copy, used table slots and what they name
    unsafe fn inputs(&mut self, l: &ID3D12GraphicsCommandList, direct: bool, gfx: bool, io: &[u32], table_at: usize, args: &[u8], phase: &'static str, uavs: &mut Vec<usize>) {
        let Some(u) = crate::gpu::uma::get() else { return };
        let base = u.base() & !0xFFFF_FFFF;
        self.range(l, &format!("args@{table_at:#x}"), table_at, args.len().max(16), phase, true);
        // heaps the io words / argument words point into (tables, view-heap base)
        let mut words: Vec<u32> = io.iter().take(32).copied().collect();
        words.extend(args.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())));
        for w in words {
            if w < 0x10000 {
                continue;
            }
            if let Some((start, bytes)) = super::console_heap::heap_extent(base | w as usize) {
                self.range(l, &format!("heap@{start:#x}"), start, bytes.min(64 << 20), phase, false);
            } else if let Some(k) = u.window_of(base | w as usize) {
                // LOCAL (agent L): a pointer into window memory outside the console heaps: 16 KiB from it
                let p = base | w as usize;
                let (wb, ws) = u.window(k);
                let n = (16usize << 10).min(wb + ws - p);
                if p != table_at {
                    self.range(l, &format!("ptr@{p:#x}"), p, n, phase, false);
                }
            }
        }
        // used table slots (LOCAL agent Li: those of the program `XCL_LOCAL_LCAP_CS_UAV` chose)
        let spec: Vec<(usize, Vec<u32>)> = match (&self.cs_spec, gfx) {
            (Some(s), false) => s.clone(),
            _ => slot_spec(gfx).to_vec(),
        };
        for (sgpr, offs) in &spec {
            let Some(&t) = io.get(*sgpr) else { continue };
            if t == 0 {
                continue;
            }
            let tbl = base | t as usize;
            // LOCAL (agent L): the table itself (GPU and CPU bytes), wherever it lives
            let span = 4 * offs.iter().copied().max().unwrap_or(0) as usize + 32;
            self.range(l, &format!("table{sgpr}@{tbl:#x}"), tbl, span, phase, true);
            for &o in offs {
                let p = tbl + 4 * o as usize;
                if !crate::gpu::shader::readable(p, 32) {
                    continue;
                }
                let w: Vec<u32> = (0..8).map(|k| (p as *const u32).add(k).read_unaligned()).collect();
                self.man.push(format!("slot s{sgpr}+{o} at {p:#x} phase {phase} words {}", w.iter().map(|x| format!("{x:08x}")).collect::<Vec<_>>().join(" ")));
                if w[3] >> 28 >= 8 {
                    if w[0] < 0x10_0000 && w[0] >= 4 {
                        self.image(l, direct, w[0], phase, uavs);
                    }
                } else if w[3] != 0 {
                    let mut a = (w[0] as u64 | ((w[1] & 0xFFFF) as u64) << 32) as usize;
                    if a >> 32 == 0 && a != 0 {
                        a |= base;
                    }
                    let stride = (w[1] >> 16) & 0x3FFF;
                    let n = if stride == 0 { w[2] as u64 } else { w[2] as u64 * stride as u64 };
                    let n = n.min(16 << 20) as usize;
                    if a != 0 && n > 0 {
                        self.range(l, &format!("buf@{a:#x}"), a, n, phase, false);
                    }
                }
            }
        }
    }
}

impl Cap {
    /// LOCAL (agent R): what a stage of a geometry draw reads, from its used-resource table (`spec2`):
    /// the argument copy, the heaps and pointers the io words name (as `inputs`), its table entries
    /// and what they name, its root descriptors' targets, and the heap slots that constants and the
    /// first words of the root descriptors' targets name (guesses, within `XCL_LOCAL_LCAP_GUESS_MB`)
    #[allow(clippy::too_many_arguments)]
    unsafe fn inputs2(&mut self, l: &ID3D12GraphicsCommandList, direct: bool, draw: u32, hash: u64, hw: u8, io: &[u32], table_at: usize, args: &[u8], phase: &'static str, uavs: &mut Vec<usize>) {
        let Some(u) = crate::gpu::uma::get() else { return };
        let base = u.base() & !0xFFFF_FFFF;
        self.range(l, &format!("args@{table_at:#x}"), table_at, args.len().max(16), phase, true);
        // the first 64 KiB of every window (null and near-null addresses are read: run R2's notes)
        for k in 0..4 {
            let (wb, ws) = u.window(k);
            if ws >= 0x1_0000 {
                self.range(l, &format!("page0@{wb:#x}"), wb, 0x1_0000, phase, false);
            }
        }
        // LOCAL (agent G): `XCL_LOCAL_LCAP_PAGES=<file>`: 64 KiB pages of the windows (one hex address a line, e.g.
        // the `not captured` pages the replay's notes list) taken at the first draw that needs a page there
        static PAGES: std::sync::OnceLock<Vec<usize>> = std::sync::OnceLock::new();
        let pages = PAGES.get_or_init(|| {
            env("XCL_LOCAL_LCAP_PAGES").and_then(|f| std::fs::read_to_string(f).ok()).map_or_else(Vec::new, |t| t.lines().filter_map(|x| usize::from_str_radix(x.trim().trim_start_matches("0x"), 16).ok()).map(|a| a & !0xFFFF).collect())
        });
        for &p in pages {
            if u.window_of(p).is_some() {
                self.range(l, &format!("page@{p:#x}"), p, 0x1_0000, phase, false);
            }
        }
        let Some(sp) = spec2(hash, hw) else {
            self.man.push(format!("nospec {hash:016x} hw {hw}"));
            return;
        };
        let sg = |dw: u8| if hw == 7 { dw as usize + super::console_list::first_sgpr_ms() - 8 } else { dw as usize };
        let word = |dw: u8| io.get(sg(dw)).copied().unwrap_or(0);
        // the tables: their used span, then each used entry
        let mut spans: HashMap<u8, (u32, u32)> = HashMap::new();
        for &(dw, off, _, dsz) in &sp.tables {
            let e = spans.entry(dw).or_insert((off, off + dsz as u32));
            e.0 = e.0.min(off);
            e.1 = e.1.max(off + dsz as u32);
        }
        for (&dw, &(lo, hi)) in &spans {
            let t = word(dw);
            if t != 0 {
                self.range(l, &format!("table{dw}@{:#x}", base | t as usize), (base | t as usize) + 4 * lo as usize, 4 * (hi - lo) as usize, phase, true);
            }
        }
        // the big raw buffers of the tables (1 MiB and more): a stage's constants are byte offsets of its
        // material record in one of them (run R3: the record begins with the heap slots of its textures)
        let mut big: Vec<(usize, usize)> = Vec::new();
        for &(dw, off, kind, _) in &sp.tables {
            let t = word(dw);
            if t == 0 {
                continue;
            }
            let p = (base | t as usize) + 4 * off as usize;
            if crate::gpu::shader::readable(p, 16) {
                let w: Vec<u32> = (0..4).map(|k| (p as *const u32).add(k).read_unaligned()).collect();
                if w[3] >> 28 < 8 && w[3] != 0 {
                    let mut a = (w[0] as u64 | ((w[1] & 0xFFFF) as u64) << 32) as usize;
                    if a >> 32 == 0 && a != 0 {
                        a |= base;
                    }
                    let stride = ((w[1] >> 16) & 0x3FFF) as usize;
                    let n = if stride == 0 { w[2] as usize } else { w[2] as usize * stride };
                    if n >= 1 << 20 && u.window_of(a).is_some() && !big.contains(&(a, n)) {
                        big.push((a, n));
                    }
                }
            }
            self.slot(l, direct, &format!("hw{hw} t{dw}+{off} kind {kind}"), p, base, phase, uavs, geom_num("XCL_LOCAL_LCAP_BUF_MB", 16) << 20);
        }
        // the root descriptors' targets
        let root_n = (geom_num("XCL_LOCAL_LCAP_ROOT_KB", 64) << 10) as usize;
        let mut small: Vec<(String, Vec<u32>)> = Vec::new();
        for &dw in &sp.roots {
            let mut a = (word(dw) as u64 | (word(dw + 1) as u64) << 32) as usize;
            if a == 0 {
                self.man.push(format!("root hw{hw} dw{dw} null"));
                continue;
            }
            if a >> 32 == 0 {
                a |= base;
            }
            let inw = u.window_of(a).map(|k| {
                let (wb, ws) = u.window(k);
                root_n.min(wb + ws - a)
            });
            self.man.push(format!("root hw{hw} dw{dw} at {a:#x} {}", inw.map_or("NOT IN A WINDOW".to_string(), |n| format!("bytes {n:#x}"))));
            if inw.is_none() {
                self.man.push(format!("oowio {draw} {} {} ea {a:#x}", if hw == 7 { "c" } else { "m" }, sg(dw)));
                if let Some(src) = crate::hle::kernelx::mirrors::source(a) {
                    if self.ranges_done.insert((a as u64, "oow")) {
                        let n = (0..=root_n / 0x1000).rev().map(|k| k * 0x1000).find(|&n| n == 0 || crate::gpu::shader::readable(src, n)).unwrap_or(0);
                        if n > 0 && self.cpu.len() + n <= (geom_num("XCL_LOCAL_LCAP_OOW_MB", 300) << 20) as usize {
                            let o = self.cpu.len();
                            self.cpu.extend_from_slice(std::slice::from_raw_parts(src as *const u8, n));
                            self.man.push(format!("oow ea {a:#x} bytes {n:#x} source {src:#x} cpu cpu.bin@{o:#x}"));
                        } else {
                            self.man.push(format!("oow ea {a:#x} source {src:#x} not copied (budget or not readable)"));
                        }
                    }
                } else {
                    self.man.push(format!("oow ea {a:#x} no CPU view"));
                }
            }
            if let Some(n) = inw {
                self.range(l, &format!("root@{a:#x}"), a, n, phase, root_cpu());
                let m = 4096.min(n);
                if crate::gpu::shader::readable(a, m) {
                    small.push((format!("root{dw}"), (0..m / 4).map(|k| (a as *const u32).add(k).read_unaligned()).collect()));
                }
            }
        }
        for &(dw, n) in &sp.consts {
            small.push((format!("const{dw}"), (0..n).map(|k| word(dw + k)).collect()));
            for k in 0..n {
                let c = word(dw + k) as usize;
                if c == 0 || c % 4 != 0 {
                    continue;
                }
                for &(a, len) in &big {
                    if c + 128 <= len && crate::gpu::shader::readable(a + c, 128) {
                        small.push((format!("mat{}@{:#x}", dw + k, a + c), (0..32).map(|j| ((a + c) as *const u32).add(j).read_unaligned()).collect()));
                    }
                }
            }
        }
        // 4 KiB at every other io word that is an address in window 0 (as agent L's `inputs`: the sampler
        // heap, ...); after the tables and the roots: a range is taken once per address
        for &w in io.iter().take(40) {
            if w >= 0x1_0000 {
                let p = base | w as usize;
                if super::console_heap::heap_extent(p).is_none() && u.window_of(p).is_some() {
                    let (wb, ws) = u.window(u.window_of(p).unwrap());
                    self.range(l, &format!("ptr@{p:#x}"), p, (4usize << 10).min(wb + ws - p), phase, false);
                }
            }
        }
        // heap slots named by those words (the stage indexes the view heap itself)
        let Some(hdw) = sp.heap else { return };
        let hb = word(hdw);
        if hb == 0 {
            self.man.push(format!("heapbase hw{hw} dw{hdw} null"));
            return;
        }
        let hb = base | hb as usize;
        let ext = super::console_heap::heap_extent(hb);
        let slots = ext.map_or(0, |(start, bytes)| (start + bytes).saturating_sub(hb) / 32);
        if let Some((start, bytes)) = ext {
            self.range(l, &format!("heap@{start:#x}"), start, bytes.min(64 << 20), phase, false);
        }
        self.man.push(format!("heapbase hw{hw} dw{hdw} at {hb:#x} slots {slots}"));
        let cap = geom_num("XCL_LOCAL_LCAP_GUESS_MB", 400) << 20;
        let mut done: HashSet<usize> = HashSet::new();
        for (src, words) in small {
            let is_const = src.starts_with("const");
            for (k, &w) in words.iter().enumerate() {
                let cands: &[usize] = if is_const { &[w as usize, (w & 0x3FF) as usize, (w & 0xFFFF) as usize, (w >> 16) as usize] } else { &[w as usize] };
                for &i in cands {
                    if i == 0 || i >= slots || !done.insert(i) {
                        continue;
                    }
                    let p = hb + 32 * i;
                    if !crate::gpu::shader::readable(p, 32) {
                        continue;
                    }
                    let w3 = (p as *const u32).add(3).read_unaligned();
                    let w0 = (p as *const u32).read_unaligned();
                    let image = w3 >> 28 >= 8;
                    // an image slot: only one whose view is known; a buffer slot: only from a constant
                    if image && !(4..0x10_0000).contains(&w0) || image && view_of(w0).is_none_or(|v| v.res == 0 || super::title_textures::address(v.res).is_none()) || !image && (!is_const || w3 == 0) {
                        continue;
                    }
                    if self.guess_bytes > cap {
                        self.man.push(format!("guess budget reached at {src}[{k}]"));
                        return;
                    }
                    let before = self.bytes;
                    self.slot(l, direct, &format!("hw{hw} guess {src}[{k}] heap {i}"), p, base, phase, uavs, 4 << 20);
                    self.guess_bytes += self.bytes - before;
                }
            }
        }
    }

    /// LOCAL (agent R): one console descriptor at `p` (as `inputs` reads a table entry): its line and what it names
    #[allow(clippy::too_many_arguments)]
    unsafe fn slot(&mut self, l: &ID3D12GraphicsCommandList, direct: bool, what: &str, p: usize, base: usize, phase: &'static str, uavs: &mut Vec<usize>, buf_max: u64) {
        if !crate::gpu::shader::readable(p, 32) {
            self.man.push(format!("gslot {what} at {p:#x} not readable"));
            return;
        }
        let w: Vec<u32> = (0..8).map(|k| (p as *const u32).add(k).read_unaligned()).collect();
        let mut line = format!("gslot {what} at {p:#x} words {}", w.iter().map(|x| format!("{x:08x}")).collect::<Vec<_>>().join(" "));
        if w[3] >> 28 >= 8 {
            if w[0] < 0x10_0000 && w[0] >= 4 {
                line += &format!(" image view {}", w[0]);
                self.man.push(line);
                self.image(l, direct, w[0], phase, uavs);
                return;
            }
        } else if w[3] != 0 {
            let mut a = (w[0] as u64 | ((w[1] & 0xFFFF) as u64) << 32) as usize;
            if a >> 32 == 0 && a != 0 {
                a |= base;
            }
            let stride = (w[1] >> 16) & 0x3FFF;
            let n = if stride == 0 { w[2] as u64 } else { w[2] as u64 * stride as u64 };
            let inw = crate::gpu::uma::get().and_then(|u| u.window_of(a)).is_some();
            line += &format!(" buffer {a:#x} bytes {n:#x} stride {stride} {}", if inw { "in a window" } else { "NOT IN A WINDOW" });
            self.man.push(line);
            let n = n.min(buf_max) as usize;
            if a != 0 && n > 0 && inw {
                self.range(l, &format!("buf@{a:#x}"), a, n, phase, false);
            } else if a != 0 && n > 0 {
                // a buffer view outside the memory windows (a shader reads zero there): the bytes the
                // title has at its CPU view, for the replay's "what if it were readable" (`oow=map`)
                self.man.push(format!("oowslot at {p:#x} ea {a:#x} w0 {:08x} w1 {:08x}", w[0], w[1]));
                let cap = (geom_num("XCL_LOCAL_LCAP_OOW_MB", 300) << 20) as usize;
                if let Some(src) = crate::hle::kernelx::mirrors::source(a) {
                    if self.ranges_done.insert((a as u64, "oow")) {
                        if self.cpu.len() + n <= cap && crate::gpu::shader::readable(src, n) {
                            let o = self.cpu.len();
                            self.cpu.extend_from_slice(std::slice::from_raw_parts(src as *const u8, n));
                            self.man.push(format!("oow ea {a:#x} bytes {n:#x} source {src:#x} cpu cpu.bin@{o:#x}"));
                        } else {
                            self.man.push(format!("oow ea {a:#x} bytes {n:#x} source {src:#x} not copied (budget or not readable)"));
                        }
                    }
                } else {
                    self.man.push(format!("oow ea {a:#x} bytes {n:#x} no CPU view"));
                }
            }
            return;
        }
        self.man.push(line);
    }
}

unsafe fn barrier_batch(l: &ID3D12GraphicsCommandList) {
    let uav = D3D12_RESOURCE_BARRIER { Type: D3D12_RESOURCE_BARRIER_TYPE_UAV, Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE, Anonymous: D3D12_RESOURCE_BARRIER_0 { UAV: ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER { pResource: ManuallyDrop::new(None) }) } };
    let alias = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_ALIASING,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 { Aliasing: ManuallyDrop::new(D3D12_RESOURCE_ALIASING_BARRIER { pResourceBefore: ManuallyDrop::new(None), pResourceAfter: ManuallyDrop::new(None) }) },
    };
    l.ResourceBarrier(&[uav, alias]);
}

fn list_ok(list: *mut c_void) -> Option<bool> {
    let t = super::list_x::tracked_list_type(list)?;
    if t == D3D12_COMMAND_LIST_TYPE_BUNDLE || t == D3D12_COMMAND_LIST_TYPE_COPY {
        return None;
    }
    Some(t == D3D12_COMMAND_LIST_TYPE_DIRECT)
}

/// translated Dispatch(x, y, z) about to bind
pub fn note_groups(g: [u32; 3]) {
    if on() {
        GROUPS.with(|c| c.set(g));
    }
}

static FIRST_CS: OnceLock<f64> = OnceLock::new();

/// LOCAL (agent R, from agent H's / L's conditions in `on_bind`): whether a capture may start now
fn armed(now: f64) -> bool {
    let rel: Option<f64> = env("XCL_LOCAL_LCAP_AFTER_CS_S").and_then(|v| v.parse().ok());
    if now < after_s() || rel.is_some_and(|r| FIRST_CS.get().is_none_or(|t| now < t + r)) {
        return false;
    }
    // LOCAL (agent L): not before the gap after the previous capture
    ((now * 1000.0) as u64) >= NEXT_ARM_MS.load(std::sync::atomic::Ordering::Relaxed)
}

/// LOCAL (agent R): whether `list`'s bound render targets include the geometry capture's format; its targets
unsafe fn geom_targets(list: *mut c_void) -> (bool, Vec<usize>, usize) {
    let (rts, ds) = super::rtsum::bound(list).unwrap_or_default();
    let fmt_want = geom_num("XCL_LOCAL_LCAP_GEOM_FMT", 29) as i32;
    let is_geom = rts.iter().any(|&r| {
        let raw = r as *mut c_void;
        r != 0 && ID3D12Resource::from_raw_borrowed(&raw).is_some_and(|x| x.GetDesc().Format.0 == fmt_want)
    });
    (is_geom, rts, ds)
}

/// LOCAL (agent R): a two-pass mesh stage is about to be bound on `list` (`compute`: its compute pass,
/// else its mesh pass) with the user data `io`, the argument copy at `table_at` and the heap bases.
/// The capture takes every such draw on the geometry targets, on every list, from the first one after
/// a present until the lighting stage begins (`on_bind`).
pub unsafe fn on_two_pass(list: *mut c_void, pso: usize, compute: bool, io: &[u32], table_at: usize, args: &[u8], heaps: [u32; 2]) {
    if !on() || geom_draws() == 0 || !armed(crate::log::uptime()) {
        return;
    }
    let mut g = CAP.lock().unwrap();
    let c = g.get_or_insert_with(Cap::default);
    if c.written || c.tl == St::Done || (c.tl == St::Idle && !c.geom_fresh) {
        return;
    }
    let Some(direct) = list_ok(list) else { return };
    let Some(l) = ID3D12GraphicsCommandList::from_raw_borrowed(&list) else { return };
    if !compute {
        // the mesh stage of a draw whose compute stage was taken
        let Some(k) = c.geom_pending.remove(&(list as usize)) else { return };
        let Some(hashes) = saved_mesh(pso) else { return };
        c.stage_now = 1;
        c.man.push(format!("gmesh {k} table {table_at:#x} io {}", io.iter().map(|w| format!("{w:x}")).collect::<Vec<_>>().join(" ")));
        barrier_batch(l);
        let mut uavs = Vec::new();
        c.inputs2(l, direct, k, hashes[1], 5, io, table_at, args, "before", &mut uavs);
        barrier_batch(l);
        return;
    }
    let (is_geom, rts, ds) = geom_targets(list);
    if !is_geom {
        return;
    }
    if c.draws >= geom_draws() {
        c.geom_over += 1;
        return;
    }
    let Some(hashes) = saved_mesh(pso) else {
        c.man.push(format!("unnoted mesh pipeline {pso:#x} on list {:#x}: draw not captured", list as usize));
        return;
    };
    c.stage_now = 1;
    if c.tl == St::Idle {
        c.note_frame = PRESENTS.load(std::sync::atomic::Ordering::Relaxed);
        c.t_first = crate::log::uptime();
        c.tl = St::Active;
        c.tl_list = 0;
        c.reserve = geom_num("XCL_LOCAL_LCAP_RESERVE_MB", 240) << 20;
        crate::xlog_info!("LOCAL lcap (geometry): frame capture starts on list {:#x} at present {}", list as usize, c.note_frame);
    }
    let t = c.trace.entry(list as usize).or_default();
    if t.len() < 6000 {
        t.push('M');
    }
    c.geom_lists.insert(list as usize);
    barrier_batch(l);
    // every target met, before: copied once, at its first use in the capture
    for r in rts.iter().copied().chain((ds != 0).then_some(ds)) {
        if r != 0 && !c.tl_targets.contains(&r) {
            c.tl_targets.push(r);
            let id = c.res_id(r);
            c.man.push(format!("gtarget {id} {}", if r == ds { "ds" } else { "rt" }));
            c.tex(l, direct, r, "before", "target");
        }
    }
    let k = c.draws;
    c.draws += 1;
    let id = c.pso(pso);
    let ids: Vec<String> = rts.iter().chain(std::iter::once(&ds)).map(|&r| if r == 0 { "-".to_string() } else { c.res_id(r).to_string() }).collect();
    c.man.push(format!("gdraw {k} list {:#x} pso {id} targets {} heaps {:x} {:x} table {table_at:#x} state {} io {}", list as usize, ids.join(","), heaps[0], heaps[1], super::list_x::local_draw_state(list), io.iter().map(|w| format!("{w:x}")).collect::<Vec<_>>().join(" ")));
    let mut uavs = Vec::new();
    c.inputs2(l, direct, k, hashes[0], 7, io, table_at, args, "before", &mut uavs);
    c.geom_pending.insert(list as usize, k);
    barrier_batch(l);
}

/// LOCAL (agent R): right after the compute pass of a two-pass mesh draw on `list`: the records it wrote
/// (the first `XCL_LOCAL_LCAP_REC_KB` of `res`, default 256) and, for an ExecuteIndirect, the rewritten
/// arguments of its two passes (`scaled`: buffer, compute command at, mesh command at, stride, offset
/// of the DispatchMesh argument); `groups`: of a direct DispatchMesh.
pub unsafe fn after_ms_compute(list: *mut c_void, res: usize, record_bytes: u64, split: u32, scaled: Option<(usize, u64, u64, u32, u32)>, groups: u32) {
    if !on() || geom_draws() == 0 {
        return;
    }
    let mut g = CAP.lock().unwrap();
    let Some(c) = g.as_mut() else { return };
    if c.tl != St::Active {
        return;
    }
    let Some(&k) = c.geom_pending.get(&(list as usize)) else { return };
    let Some(l) = ID3D12GraphicsCommandList::from_raw_borrowed(&list) else { return };
    let raw = res as *mut c_void;
    let Some(r) = ID3D12Resource::from_raw_borrowed(&raw) else { return };
    c.stage_now = 1;
    // whole records
    let n = (geom_num("XCL_LOCAL_LCAP_REC_KB", 256) << 10).min(r.GetDesc().Width) / record_bytes.max(1) * record_bytes.max(1);
    let mut line = format!("grec {k} record_bytes {record_bytes} split {split} groups {groups}");
    barrier_batch(l);
    if n > 0 {
        if let Some((ch, at)) = c.alloc(n) {
            l.CopyBufferRegion(c.chunks[ch].rb.as_ref().unwrap(), at, r, 0, n);
            line += &format!(" bytes {n:#x} file chunk{ch}.bin@{at:#x}");
        }
    }
    if let Some((sb, at_c, at_m, stride, off)) = scaled {
        let sraw = sb as *mut c_void;
        if let Some(sr) = ID3D12Resource::from_raw_borrowed(&sraw) {
            let st = (stride as u64 + 3) & !3;
            if let Some((ch, at)) = c.alloc(2 * st) {
                l.CopyBufferRegion(c.chunks[ch].rb.as_ref().unwrap(), at, sr, at_c, st);
                l.CopyBufferRegion(c.chunks[ch].rb.as_ref().unwrap(), at + st, sr, at_m, st);
                line += &format!(" args chunk{ch}.bin@{at:#x} stride {st} offset {off}");
            }
        }
    }
    barrier_batch(l);
    c.man.push(line);
}

/// `bind_abi` after the io words are made (before the pipeline is set).
pub unsafe fn on_bind(list: *mut c_void, pso: usize, gfx: bool, io: &[u32], table_at: usize, args: &[u8]) {
    if !on() {
        return;
    }
    // `XCL_LOCAL_LCAP_AFTER_CS_S=<s>`: armed that long after the first dispatch of the mask program
    // (the world's first lighting frame), and not before `XCL_LOCAL_LCAP_AFTER_S` (runs h4 / h5: the
    // world arrived at 98-118 s and the device was lost 25-42 s later)
    let now = crate::log::uptime();
    if !gfx && FIRST_CS.get().is_none() && saved_hashes(pso).is_some_and(|(gx, h)| !gx && matches(h[0], cs_set())) {
        let _ = FIRST_CS.set(now);
        crate::xlog_info!("LOCAL lcap: first dispatch of the mask program at {now:.3} s");
    }
    // LOCAL (agent Li): compute programs that name a watched UAV (`XCL_LOCAL_LCAP_CS_UAV`), whatever the arming
    let mut li_slots: Option<Vec<(usize, Vec<u32>)>> = None;
    if !gfx && !uav_watch().is_empty() {
        if let Some((false, h)) = saved_hashes(pso) {
            let (hits, slots) = uav_watch_hits(h[0], io);
            if !hits.is_empty() {
                static SEEN: Mutex<Option<HashSet<(u64, u32)>>> = Mutex::new(None);
                static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                let fresh = hits.iter().any(|&(_, _, i)| SEEN.lock().unwrap().get_or_insert_with(HashSet::new).insert((h[0], i)));
                let k = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if k < 64 || fresh {
                    let at: Vec<String> = hits.iter().map(|(dw, off, i)| format!("{i} at s{dw}+{off}")).collect();
                    crate::xlog_info!(
                        "LOCAL lcap uavwatch #{k}: program {:016x} names as UAV {} on list {:#x} type {:?}, groups {:?} (0 0 0: indirect), present {}",
                        h[0], at.join(", "), list as usize, super::list_x::tracked_list_type(list).map(|t| t.0), GROUPS.with(|c| c.get()), PRESENTS.load(std::sync::atomic::Ordering::Relaxed)
                    );
                }
                li_slots = Some(slots);
            }
        }
    }
    if !armed(now) {
        return;
    }
    // LOCAL (agent R): the geometry capture takes two-pass mesh draws (`on_two_pass`); here: it ends at the
    // first bind of the lighting stage (the mask program or a tiled-lighting draw), where the targets are
    // copied again; other translated draws on the geometry targets are counted (not captured)
    if geom_draws() > 0 {
        let mut g = CAP.lock().unwrap();
        let Some(c) = g.as_mut() else { return };
        if c.written || c.tl != St::Active {
            return;
        }
        let lighting = saved_hashes(pso).is_some_and(|(gx, h)| if gx { gfx && matches(h[0], vs_set()) } else { !gfx && matches(h[0], cs_set()) });
        if lighting {
            if let (Some(direct), Some(l)) = (list_ok(list), ID3D12GraphicsCommandList::from_raw_borrowed(&list)) {
                c.tl_list = list as usize;
                c.man.push(format!("gend list {:#x} foreign {} over {}", list as usize, c.geom_foreign, c.geom_over));
                end_tl(c, l, direct, if gfx { "the lighting draws begin" } else { "the mask program is bound" });
            }
            return;
        }
        if gfx && geom_targets(list).0 {
            c.geom_foreign += 1;
            let t = c.trace.entry(list as usize).or_default();
            if t.len() < 6000 {
                t.push('G');
            }
        }
        return;
    }
    let mut g = CAP.lock().unwrap();
    let c = g.get_or_insert_with(Cap::default);
    if c.written {
        return;
    }
    // LOCAL diagnostic (run h4: no TL run started): the first graphics binds after the compute capture
    if gfx && c.cs == St::Done && c.tl == St::Idle {
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let k = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if k < 6 || (k < 2000 && super::rtsum::shaders(pso).contains("vs b6395e96")) && k % 50 == 0 {
            crate::xlog_info!("LOCAL lcap: gfx bind {k} after the compute: {} saved {:?} list type {:?} pso {pso:#x} real {:#x}", super::rtsum::shaders(pso), saved_hashes(pso).map(|h| format!("{:016x}", h.1[0])), super::list_x::tracked_list_type(list).map(|t| t.0), super::pso_async::resolve(pso));
        }
    }
    let Some(direct) = list_ok(list) else { return };
    let Some(l) = ID3D12GraphicsCommandList::from_raw_borrowed(&list) else { return };
    let hashes = saved_hashes(pso);
    let need_cs = env("XCL_LOCAL_LCAP_NO_CS").is_none();
    // the run of tiled-lighting draws ends at the first other bind on its list
    let is_tl = gfx && hashes.is_some_and(|(gx, h)| gx && matches(h[0], vs_set()));
    if c.tl == St::Active && list as usize == c.tl_list && (!is_tl || c.draws >= max_draws()) {
        end_tl(c, l, direct, "next bind");
    }
    // LOCAL (agent Li): with `XCL_LOCAL_LCAP_CS_UAV`, the first program that names a watched UAV
    let cs_wanted = if uav_watch().is_empty() { hashes.is_some_and(|(gx, h)| !gx && matches(h[0], cs_set())) } else { li_slots.is_some() };
    if !gfx && c.cs == St::Idle && cs_wanted {
        c.cs_spec = li_slots;
        c.note_frame = PRESENTS.load(std::sync::atomic::Ordering::Relaxed);
        c.stage_now = 0;
        c.t_first = now;
        let id = c.pso(pso);
        let groups = GROUPS.with(|c| c.get());
        c.cs_groups = groups;
        c.man.push(format!("dispatch 0 list {:#x} pso {id} groups {} {} {} present {} at {:.3} s io {}", list as usize, groups[0], groups[1], groups[2], c.note_frame, crate::log::uptime(), io.iter().map(|w| format!("{w:x}")).collect::<Vec<_>>().join(" ")));
        barrier_batch(l);
        let mut uavs = Vec::new();
        c.inputs(l, direct, false, io, table_at, args, "csbefore", &mut uavs);
        barrier_batch(l);
        c.cs_uav_res = uavs;
        c.cs = St::Active;
        c.cs_list = list as usize;
        crate::xlog_info!("LOCAL lcap: compute program captured before its dispatch at present {} ({} MB so far)", c.note_frame, c.bytes >> 20);
        return;
    }
    if is_tl && (c.cs == St::Done || !need_cs) && (c.tl == St::Idle || (c.tl == St::Active && list as usize == c.tl_list)) {
        let first = c.tl == St::Idle;
        c.stage_now = 1;
        if first {
            if c.t_first == 0.0 {
                c.t_first = now;
            }
            c.tl = St::Active;
            c.tl_list = list as usize;
            let (rts, ds) = super::rtsum::bound(list).unwrap_or_default();
            c.tl_targets = rts.iter().copied().chain((ds != 0).then_some(ds)).collect();
            let ids: Vec<String> = c.tl_targets.clone().iter().map(|&r| c.res_id(r).to_string()).collect();
            c.man.push(format!("targets {} rt {} ds {}", ids.join(","), rts.len(), (ds != 0) as u32));
            crate::xlog_info!("LOCAL lcap: tiled-lighting run starts on list {:#x} at present {}", list as usize, PRESENTS.load(std::sync::atomic::Ordering::Relaxed));
        }
        let id = c.pso(pso);
        let k = c.draws;
        c.draws += 1;
        let (rts, ds) = super::rtsum::bound(list).unwrap_or_default();
        let ids: Vec<String> = rts.iter().chain(std::iter::once(&ds)).map(|&r| if r == 0 { "-".to_string() } else { c.res_id(r).to_string() }).collect();
        c.man.push(format!("draw {k} list {:#x} pso {id} targets {} state {} io {}", list as usize, ids.join(","), super::list_x::local_draw_state(list), io.iter().map(|w| format!("{w:x}")).collect::<Vec<_>>().join(" ")));
        barrier_batch(l);
        let mut uavs = Vec::new();
        if first {
            for r in c.tl_targets.clone() {
                c.tex(l, direct, r, "before", "target");
            }
        }
        c.inputs(l, direct, true, io, table_at, args, "before", &mut uavs);
        barrier_batch(l);
    }
}

unsafe fn end_tl(c: &mut Cap, l: &ID3D12GraphicsCommandList, direct: bool, why: &str) {
    c.stage_now = 1;
    c.reserve = 0; // LOCAL (agent R)
    barrier_batch(l);
    for r in c.tl_targets.clone() {
        c.tex(l, direct, r, "after", "target");
    }
    barrier_batch(l);
    c.tl = St::Done;
    c.man.push(format!("end {} draws ({why})", c.draws));
    crate::xlog_info!("LOCAL lcap: tiled-lighting run ends after {} draws ({why}); {} MB in readback buffers, {} skipped for the budget", c.draws, c.bytes >> 20, c.skipped);
}

/// after the translated Dispatch (or ExecuteIndirect dispatch) on `list`
pub unsafe fn after_dispatch(list: *mut c_void) {
    if !on() {
        return;
    }
    let mut g = CAP.lock().unwrap();
    let Some(c) = g.as_mut() else { return };
    if c.cs != St::Active || c.cs_list != list as usize {
        return;
    }
    let Some(direct) = list_ok(list) else { return };
    let Some(l) = ID3D12GraphicsCommandList::from_raw_borrowed(&list) else { return };
    barrier_batch(l);
    c.stage_now = 0;
    for r in c.cs_uav_res.clone() {
        c.tex(l, direct, r, "csafter", "uav");
    }
    barrier_batch(l);
    c.cs = St::Done;
    crate::xlog_info!("LOCAL lcap: compute program's UAV textures captured after it ({} MB so far)", c.bytes >> 20);
}

/// LOCAL (agent Li): after a translated ExecuteIndirect dispatch on `list`, before `after_dispatch`: the
/// argument records of the captured compute program (its groups: 12 bytes per record)
pub unsafe fn cs_args(list: *mut c_void, args: usize, args_off: u64, max: u32) {
    if !on() {
        return;
    }
    let mut g = CAP.lock().unwrap();
    let Some(c) = g.as_mut() else { return };
    if c.cs != St::Active || c.cs_list != list as usize {
        return;
    }
    let Some(l) = ID3D12GraphicsCommandList::from_raw_borrowed(&list) else { return };
    let raw = args as *mut c_void;
    let Some(a) = ID3D12Resource::from_raw_borrowed(&raw) else { return };
    let n = (12 * max.clamp(1, 16) as u64).min(a.GetDesc().Width.saturating_sub(args_off)) & !3;
    if n == 0 {
        return;
    }
    c.stage_now = 0;
    let Some((k, at)) = c.alloc(n) else { return };
    l.CopyBufferRegion(c.chunks[k].rb.as_ref().unwrap(), at, a, args_off, n);
    c.man.push(format!("csargs max {max} args {args:#x}+{args_off:#x} bytes {n:#x} file chunk{k}.bin@{at:#x}"));
}

/// after a translated ExecuteIndirect draw on `list`: its argument records
pub unsafe fn after_ei(list: *mut c_void, args: usize, args_off: u64, max: u32, stride: u32, count_buf: usize, count_off: u64) {
    if !on() {
        return;
    }
    let mut g = CAP.lock().unwrap();
    let Some(c) = g.as_mut() else { return };
    if c.tl != St::Active || c.tl_list != list as usize {
        return;
    }
    let Some(l) = ID3D12GraphicsCommandList::from_raw_borrowed(&list) else { return };
    let raw = args as *mut c_void;
    let Some(a) = ID3D12Resource::from_raw_borrowed(&raw) else { return };
    let n = (stride.max(16) as u64 * max.clamp(1, 64) as u64).min(a.GetDesc().Width.saturating_sub(args_off)) & !3;
    if n == 0 {
        return;
    }
    c.stage_now = 1;
    let Some((k, at)) = c.alloc(n) else { return };
    // the records as the GPU has them right before the draw ran (copied after it on the list:
    // nothing between writes them)
    l.CopyBufferRegion(c.chunks[k].rb.as_ref().unwrap(), at, a, args_off, n);
    let mut line = format!("records {} args {args:#x}+{args_off:#x} stride {stride} max {max} bytes {n:#x} file chunk{k}.bin@{at:#x}", c.draws.saturating_sub(1));
    // LOCAL (agent L): the count the GPU used, when the title passes a count buffer
    let craw = count_buf as *mut c_void;
    if let (true, Some(cb)) = (count_buf != 0, ID3D12Resource::from_raw_borrowed(&craw)) {
        if count_off + 4 <= cb.GetDesc().Width {
            if let Some((k2, at2)) = c.alloc(4) {
                l.CopyBufferRegion(c.chunks[k2].rb.as_ref().unwrap(), at2, cb, count_off, 4);
                line += &format!(" count chunk{k2}.bin@{at2:#x}");
            }
        }
    }
    c.man.push(line);
}

/// before Close of `list`
pub unsafe fn on_close(list: *mut c_void) {
    if !on() {
        return;
    }
    let mut g = CAP.lock().unwrap();
    let Some(c) = g.as_mut() else { return };
    if c.tl == St::Active && c.tl_list == list as usize {
        if let (Some(direct), Some(l)) = (list_ok(list), ID3D12GraphicsCommandList::from_raw_borrowed(&list)) {
            end_tl(c, l, direct, "list closed");
        }
    }
}

/// LOCAL (agent L): `lists` (the title's, and the PC lists they map to) ran on `queue`: a stage whose
/// list is among them is written out at once
pub unsafe fn on_execute(queue: *mut c_void, raw: &[*mut c_void], mapped: &[*mut c_void]) {
    if !on() {
        return;
    }
    let mut g = CAP.lock().unwrap();
    let Some(c) = g.as_mut() else { return };
    if c.written {
        return;
    }
    let has = |l: usize| l != 0 && (raw.iter().any(|x| *x as usize == l) || mapped.iter().any(|x| *x as usize == l));
    // LOCAL (agent R): the order the lists with captured geometry draws are submitted in
    if geom_draws() > 0 && c.tl != St::Idle {
        for x in raw.iter().chain(mapped.iter()) {
            if c.geom_lists.remove(&(*x as usize)) {
                c.man.push(format!("gexec {:#x}", *x as usize));
            }
        }
    }
    if c.cs == St::Done && !c.cs_exec && has(c.cs_list) {
        c.cs_exec = true;
    }
    if c.tl == St::Done && !c.tl_exec && has(c.tl_list) {
        c.tl_exec = true;
    }
    let qraw = queue;
    let Some(q) = ID3D12CommandQueue::from_raw_borrowed(&qraw) else { return };
    if q.GetDesc().Type != D3D12_COMMAND_LIST_TYPE_DIRECT {
        return;
    }
    try_flush(c, q, "execute");
}

/// LOCAL (agent L): writes what is ready. The manifest goes first (every line so far), then the chunks
/// of the finished stages, then a `complete` line: a write cut by a device loss leaves a manifest
/// whose entries the replay checks against the chunk files' lengths.
/// LOCAL (agent L): one write of finished chunks, run on its own thread with no capture lock held and
/// through the queue's original ExecuteCommandLists (runs L1 / L2: a write made on the title's thread
/// under the capture lock and through the hooked queue never returned).
struct Job {
    seq: u64,
    index: u32,
    dir: std::path::PathBuf,
    lines: Vec<String>,
    cpu: Vec<u8>,
    chunks: Vec<(usize, ID3D12Resource, u64)>,
    fin: bool,
    draws: u32,
    skipped: u64,
    fence: Option<ID3D12Fence>,
    q: ID3D12CommandQueue,
    dev: ID3D12Device,
    from: String,
    /// LOCAL (agent G): the root ranges to read again on the CPU once the GPU is done
    cpu_roots: Vec<(usize, usize)>,
}
struct SendJob(Job);
unsafe impl Send for SendJob {}
static JOB_LOCK: Mutex<()> = Mutex::new(());
static JOB_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
static LAST_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// LOCAL (agent L): what is ready leaves the capture (under the lock, no GPU wait) and is written by a thread.
/// The manifest goes first (every line so far), then the chunks, then the manifest again with a
/// `complete` line: a write cut by a device loss leaves a manifest whose entries the replay checks
/// against the chunk files' lengths.
unsafe fn try_flush(c: &mut Cap, q: &ID3D12CommandQueue, from: &str) {
    use std::sync::atomic::Ordering::Relaxed;
    let tl_ready = c.tl == St::Done && (c.tl_exec || c.end_presents >= 3);
    let stale = c.cs == St::Done && c.tl == St::Idle && c.cs_presents > 6;
    let cs_ready = c.cs == St::Done && !c.cs_flushed && (c.cs_exec || c.cs_presents >= 3);
    let fin = tl_ready || stale;
    if !fin && !cs_ready {
        return;
    }
    let Some(dev) = device() else { return };
    let mut lines = vec![
        format!("# LOCAL agent H/L lighting capture {}: first copy at {:.3} s, taken for writing at {:.3} s (from {from}), present {}", c.index, c.t_first, crate::log::uptime(), PRESENTS.load(Relaxed)),
        format!("uma_base_hi {:#x}", crate::gpu::uma::get().map_or(4, |u| (u.base() >> 32) as u32)),
        format!("windows {}", super::real::uma_window_vas().iter().map(|v| format!("{v:#x}")).collect::<Vec<_>>().join(" ")),
    ];
    lines.extend(c.man.iter().cloned());
    // LOCAL (agent R): the binds seen on the lists with the geometry targets (M mesh draw, G other draw, C dispatch)
    for (lst, t) in &c.trace {
        lines.push(format!("trace list {lst:#x} {t}"));
    }
    if geom_draws() > 0 && fin {
        if let Some(v) = VIEWS.lock().unwrap().as_ref() {
            let mut all: Vec<(u32, View)> = v.by_index.iter().filter_map(|(i, h)| Some((*i, *v.by_handle.get(h)?))).filter(|x| x.1.res != 0 && !x.1.uav).collect();
            all.sort_by_key(|x| x.0);
            let (mut ev, mut text) = (0, String::new());
            for (i, w) in &all {
                let e = super::vram::is_evicted(w.res);
                ev += e as u32;
                text += &format!("{i} {:#x} f{} {}\n", w.res, u32::from_le_bytes(w.desc[..4].try_into().unwrap()), e as u8);
            }
            lines.push(format!("allviews {} srv views with a resource, {ev} of an evicted texture (file views.txt: index, resource, view format, evicted)", all.len()));
            let dir = cap_dir(c.index);
            let _ = std::fs::create_dir_all(&dir);
            let _ = std::fs::write(dir.join("views.txt"), text);
        }
    }
    if let Some(v) = VIEWS.lock().unwrap().as_ref() {
        let mut s: Vec<(u32, String)> = v.samp_by_index.iter().filter_map(|(i, h)| Some((*i, v.samp_by_handle.get(h)?.clone()))).collect();
        s.sort();
        for (i, d) in s {
            lines.push(format!("sampler {i} desc {d}"));
        }
    }
    let stages: &[u8] = if fin { &[0, 1] } else { &[0] };
    let mut chunks = Vec::new();
    for (k, ch) in c.chunks.iter_mut().enumerate() {
        if ch.written || !stages.contains(&ch.stage) {
            continue;
        }
        if let Some(rb) = ch.rb.take() {
            chunks.push((k, rb, ch.used));
        }
        ch.written = true;
    }
    // everything submitted to this queue so far (the stage's list is among it) is done when this fence is
    let fence = dev.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE).ok();
    if let Some(f) = &fence {
        let _ = q.Signal(f, 1);
    }
    let job = Job { seq: JOB_SEQ.fetch_add(1, Relaxed), index: c.index, dir: cap_dir(c.index), lines, cpu: c.cpu.clone(), chunks, fin, draws: c.draws, skipped: c.skipped, fence, q: q.clone(), dev, from: from.to_string(), cpu_roots: if fin { c.cpu_roots.clone() } else { Vec::new() } };
    crate::xlog_info!("LOCAL lcap: capture {} {} handed to the writer ({} chunks, from {from})", c.index, if fin { "(all)" } else { "(cs stage)" }, job.chunks.len());
    if !fin {
        c.cs_flushed = true;
    } else {
        let k = c.index + 1;
        if k < repeat() {
            // armed again by the writer when this capture is on disk
            NEXT_ARM_MS.store(u64::MAX, Relaxed);
            *c = Cap { index: k, ..Default::default() };
        } else {
            c.written = true;
            c.chunks.clear();
            c.cpu = Vec::new();
        }
    }
    let sj = SendJob(job);
    let _ = std::thread::Builder::new().name("lcap-writer".into()).spawn(move || {
        let sj = sj;
        run_job(sj.0)
    });
}

unsafe fn run_job(job: Job) {
    use std::sync::atomic::Ordering::Relaxed;
    let _g = JOB_LOCK.lock().unwrap();
    let t0 = std::time::Instant::now();
    let _ = std::fs::create_dir_all(&job.dir);
    let newest = LAST_SEQ.fetch_max(job.seq, Relaxed) <= job.seq;
    let write_man = |extra: &[String]| {
        let all: Vec<&str> = job.lines.iter().chain(extra.iter()).map(|s| s.as_str()).collect();
        let _ = std::fs::write(job.dir.join("manifest.txt"), all.join("\n") + "\n");
    };
    if newest {
        write_man(&[]);
        let _ = std::fs::write(job.dir.join("cpu.bin"), &job.cpu);
    }
    let fence_ok = match &job.fence {
        Some(f) => match crate::gpu::fence::wait(f, 1, &job.dev, None, 20000) {
            Ok(()) => true,
            Err(e) => {
                crate::xlog_warn!("LOCAL lcap: fence wait failed ({e}) - writing what is there");
                false
            }
        },
        None => false,
    };
    // LOCAL (agent G): the CPU's view of the root ranges now that the GPU ran the lists: next to the
    // CPU bytes taken at record time and the GPU's (`range ... cpu`), it tells a CPU write the GPU
    // never got (record = after != GPU) from data the GPU wrote (GPU = after != record)
    let mut extra: Vec<String> = Vec::new();
    if !job.cpu_roots.is_empty() {
        let mut after: Vec<u8> = Vec::new();
        for &(p, n) in &job.cpu_roots {
            if crate::gpu::shader::readable(p, n) {
                let o = after.len();
                after.extend_from_slice(std::slice::from_raw_parts(p as *const u8, n));
                extra.push(format!("cpuafter ea {p:#x} bytes {n:#x} file cpu_after.bin@{o:#x} at {:.3} s", crate::log::uptime()));
            } else {
                extra.push(format!("cpuafter ea {p:#x} bytes {n:#x} not readable"));
            }
        }
        let _ = std::fs::write(job.dir.join("cpu_after.bin"), &after);
    }
    let (total, done) = read_chunks(&job.dev, &job.q, &job.chunks, &job.dir);
    if LAST_SEQ.load(Relaxed) <= job.seq {
        extra.push(format!("complete {} chunks {done:?} of {} fence {fence_ok} bytes {total:#x} draws {} skipped {}", if job.fin { "all" } else { "cs" }, job.chunks.len(), job.draws, job.skipped));
        write_man(&extra);
    }
    crate::xlog_info!("LOCAL lcap: written {} ({}, from {}): chunks {done:?} of {}, {} MB, {} manifest lines, {} draws, in {:.1} s", job.dir.display(), if job.fin { "all" } else { "cs stage" }, job.from, job.chunks.len(), total >> 20, job.lines.len(), job.draws, t0.elapsed().as_secs_f64());
    if job.fin && job.index + 1 < repeat() {
        let gap: f64 = env("XCL_LOCAL_LCAP_GAP_S").and_then(|v| v.parse().ok()).unwrap_or(6.0);
        NEXT_ARM_MS.store(((crate::log::uptime() + gap) * 1000.0) as u64, Relaxed);
    }
}

/// at each present: three presents after the end, the readback buffers are written out
pub unsafe fn on_present(queue: *mut c_void) {
    if !on() {
        return;
    }
    let n = PRESENTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let mut g = CAP.lock().unwrap();
    if geom_draws() > 0 && g.is_none() && armed(crate::log::uptime()) {
        *g = Some(Cap::default()); // LOCAL (agent R)
    }
    let Some(c) = g.as_mut() else { return };
    if c.written {
        return;
    }
    // LOCAL (agent L): the present count is only the fallback for a list whose submission was not seen
    let _ = n;
    // LOCAL (agent R)
    if geom_draws() > 0 {
        if c.tl == St::Idle && armed(crate::log::uptime()) {
            c.geom_fresh = true;
        } else if c.tl == St::Active {
            c.geom_presents += 1;
            if c.geom_presents >= 2 {
                c.man.push(format!("gend present foreign {} over {} (no lighting bind seen: targets not copied after)", c.geom_foreign, c.geom_over));
                c.tl = St::Done;
                c.tl_exec = true;
                crate::xlog_info!("LOCAL lcap (geometry): frame capture ends at a present after {} draws; {} MB", c.draws, c.bytes >> 20);
            }
        }
    }
    if c.cs == St::Done {
        c.cs_presents += 1;
    }
    if c.tl == St::Done {
        c.end_presents += 1;
    }
    let qraw = queue;
    let Some(q) = ID3D12CommandQueue::from_raw_borrowed(&qraw) else { return };
    try_flush(c, q, "present");
}

/// the VRAM chunks to `<dir>/chunk<k>.bin` through one 64 MB readback buffer on `q` (its original
/// ExecuteCommandLists: no XCL hook, no XCL lock)
unsafe fn read_chunks(dev: &ID3D12Device, q: &ID3D12CommandQueue, chunks: &[(usize, ID3D12Resource, u64)], dir: &std::path::Path) -> (u64, Vec<usize>) {
    use std::io::Write;
    // LOCAL (agent R): `XCL_LOCAL_LCAP_PIECE_MB` (default 64): each piece waits its turn behind the title's own
    // work on the queue (run R3: 0,75 s a piece), so bigger pieces write a capture sooner
    #[allow(non_snake_case)]
    let PIECE: u64 = geom_num("XCL_LOCAL_LCAP_PIECE_MB", 64).clamp(16, 1024) << 20;
    let mut done: Vec<usize> = Vec::new();
    let mut total = 0u64;
    let Ok(alloc) = dev.CreateCommandAllocator::<ID3D12CommandAllocator>(D3D12_COMMAND_LIST_TYPE_DIRECT) else { return (0, done) };
    let Ok(fence) = dev.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE) else { return (0, done) };
    let bd = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: PIECE,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    let mut rb: Option<ID3D12Resource> = None;
    if dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut rb).is_err() {
        return (0, done);
    }
    let Some(rb) = rb else { return (0, done) };
    let qraw = q.as_raw();
    type Exec = unsafe extern "system" fn(*mut c_void, u32, *const *mut c_void);
    let exec: Exec = std::mem::transmute(super::real::orig_fn(qraw, 10));
    let mut fv = 0u64;
    for (k, chrb, used) in chunks.iter() {
        let Ok(mut f) = std::fs::File::create(dir.join(format!("chunk{k}.bin"))) else { continue };
        let mut at = 0u64;
        let mut first = true;
        while at < *used {
            let n = PIECE.min(*used - at);
            let _ = alloc.Reset();
            let Ok(l) = dev.CreateCommandList::<_, _, ID3D12GraphicsCommandList>(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None) else { return (total, done) };
            if first {
                let b = D3D12_RESOURCE_BARRIER {
                    Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                    Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                    Anonymous: D3D12_RESOURCE_BARRIER_0 {
                        Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                            pResource: std::mem::transmute_copy(chrb),
                            Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                            StateBefore: D3D12_RESOURCE_STATE_COPY_DEST,
                            StateAfter: D3D12_RESOURCE_STATE_COPY_SOURCE,
                        }),
                    },
                };
                l.ResourceBarrier(&[b]);
                first = false;
            }
            l.CopyBufferRegion(&rb, 0, chrb, at, n);
            if l.Close().is_err() {
                return (total, done);
            }
            let lraw = l.as_raw();
            exec(qraw, 1, &lraw);
            fv += 1;
            let _ = q.Signal(&fence, fv);
            if crate::gpu::fence::wait(&fence, fv, dev, None, 20000).is_err() {
                crate::xlog_warn!("LOCAL lcap: readback of chunk {k} at {at:#x} did not finish");
                return (total, done);
            }
            let mut m: *mut c_void = std::ptr::null_mut();
            if rb.Map(0, None, Some(&mut m)).is_ok() && !m.is_null() {
                let _ = f.write_all(std::slice::from_raw_parts(m as *const u8, n as usize));
                rb.Unmap(0, None);
            }
            at += n;
            total += n;
        }
        let _ = f.flush();
        done.push(*k);
    }
    (total, done)
}
