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
    let s = SavedPso { gfx: true, hashes: [hv, hp], desc: hexbytes(desc as *const u8, std::mem::size_of::<D3D12_GRAPHICS_PIPELINE_STATE_DESC>()), blobs: [Some(vs), ps].into_iter().flatten().collect() };
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
    if !matches(h, cs_set()) {
        return;
    }
    let s = SavedPso { gfx: false, hashes: [h, 0], desc: hexbytes(desc as *const u8, std::mem::size_of::<D3D12_COMPUTE_PIPELINE_STATE_DESC>()), blobs: vec![cs] };
    PSOS.lock().unwrap().get_or_insert_with(HashMap::new).insert(pso, s);
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
        if self.bytes + n > budget() {
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
        if cpu && crate::gpu::shader::readable(p, n) {
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
        let head = format!("tex {id} {phase} res {r:#x} ({}x{}x{} f{} mips {} flags {:#x} samples {}) why {why} desc {dhex}", d.Width, d.Height, d.DepthOrArraySize, d.Format.0, d.MipLevels, d.Flags.0, d.SampleDesc.Count);
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
        dev.GetCopyableFootprints(&d, 0, nsub, 0, Some(fps.as_mut_ptr()), None, None, Some(&mut total));
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
                    let n = format!("pso{id}_{}.bin", if !s.gfx { "cs" } else if k == 0 { "vs" } else { "ps" });
                    let _ = std::fs::write(dir.join(&n), b);
                    n
                }).collect();
                self.man.push(format!("pso {id} {} {:016x} {:016x} files {} desc {}", if s.gfx { "gfx" } else { "cs" }, s.hashes[0], s.hashes[1], names.join(","), s.desc));
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
        // used table slots
        for (sgpr, offs) in slot_spec(gfx) {
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

/// `bind_abi` after the io words are made (before the pipeline is set).
pub unsafe fn on_bind(list: *mut c_void, pso: usize, gfx: bool, io: &[u32], table_at: usize, args: &[u8]) {
    if !on() {
        return;
    }
    // `XCL_LOCAL_LCAP_AFTER_CS_S=<s>`: armed that long after the first dispatch of the mask program
    // (the world's first lighting frame), and not before `XCL_LOCAL_LCAP_AFTER_S` (runs h4 / h5: the
    // world arrived at 98-118 s and the device was lost 25-42 s later)
    static FIRST_CS: OnceLock<f64> = OnceLock::new();
    let now = crate::log::uptime();
    if !gfx && FIRST_CS.get().is_none() && saved_hashes(pso).is_some_and(|(gx, h)| !gx && matches(h[0], cs_set())) {
        let _ = FIRST_CS.set(now);
        crate::xlog_info!("LOCAL lcap: first dispatch of the mask program at {now:.3} s");
    }
    let rel: Option<f64> = env("XCL_LOCAL_LCAP_AFTER_CS_S").and_then(|v| v.parse().ok());
    if now < after_s() || rel.is_some_and(|r| FIRST_CS.get().is_none_or(|t| now < t + r)) {
        return;
    }
    // LOCAL (agent L): not before the gap after the previous capture
    if ((now * 1000.0) as u64) < NEXT_ARM_MS.load(std::sync::atomic::Ordering::Relaxed) {
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
    if !gfx && c.cs == St::Idle && hashes.is_some_and(|(gx, h)| !gx && matches(h[0], cs_set())) {
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
    let job = Job { seq: JOB_SEQ.fetch_add(1, Relaxed), index: c.index, dir: cap_dir(c.index), lines, cpu: c.cpu.clone(), chunks, fin, draws: c.draws, skipped: c.skipped, fence, q: q.clone(), dev, from: from.to_string() };
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
    let (total, done) = read_chunks(&job.dev, &job.q, &job.chunks, &job.dir);
    if LAST_SEQ.load(Relaxed) <= job.seq {
        write_man(&[format!("complete {} chunks {done:?} of {} fence {fence_ok} bytes {total:#x} draws {} skipped {}", if job.fin { "all" } else { "cs" }, job.chunks.len(), job.draws, job.skipped)]);
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
    let Some(c) = g.as_mut() else { return };
    if c.written {
        return;
    }
    // LOCAL (agent L): the present count is only the fallback for a list whose submission was not seen
    let _ = n;
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
    const PIECE: u64 = 64 << 20;
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
