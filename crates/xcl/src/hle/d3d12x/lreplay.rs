//! LOCAL (agent H): offline replay of a lighting capture (`lcap.rs`, `XCL_LOCAL_LCAP`).
//!
//! `XCL_HEAP_WINDOWS=1 XCL_LREPLAY=<capture dir> cargo test --release -p xcl --lib lreplay -- --nocapture`
//! - `XCL_LREPLAY_MODE=cs|tl|both` (default both): the material-mask dispatch, the tiled-lighting
//!   draws, or the dispatch then the draws on the same resources (the draws then see what the
//!   replayed dispatch wrote into the twins, and `XCL_LREPLAY_DEPTH` decides f44)
//! - `XCL_LREPLAY_DEPTH=capture|twin|twin16` the second depth before the draws: as captured (default),
//!   the depth twin's R32 values written into it as depth (`twin`), or as k/65535 of the twin's
//!   value read as an integer (`twin16`)
//! - `XCL_LREPLAY_INSTANCES=<n>`: every draw's instance count forced to n (default: the record's)
//! - `XCL_LREPLAY_OUT=<dir>` (default `<capture>/replay`): outputs (raw targets + BMPs + stats)
//! - `XCL_LREPLAY_DRAWS=a..b` only those draws
//!
//! The memory windows are rebuilt as in agent R's `iso_cull`: 4 GiB reserved buffers with the
//! captured ranges' tiles mapped at their own offsets (no relocation). Textures are recreated
//! with their PC descriptions and filled with the captured bytes; every captured view is made
//! at its shared-heap index with its PC description; samplers at their sampler-heap index.

use std::collections::HashMap;
use std::mem::{transmute_copy, ManuallyDrop};
use std::path::{Path, PathBuf};
use std::time::Instant;

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D::*;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::*;

use crate::gpu::recomp;

mod geom; // LOCAL (agent R): the geometry pass (two-pass mesh draws)

fn env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.trim().is_empty())
}

fn hex(s: &str) -> u64 {
    u64::from_str_radix(s.trim_start_matches("0x"), 16).unwrap()
}

fn unhex(s: &str) -> Option<Vec<u8>> {
    if s == "none" {
        return None;
    }
    Some((0..s.len() / 2).map(|k| u8::from_str_radix(&s[2 * k..2 * k + 2], 16).unwrap()).collect())
}

fn tok<'a>(l: &'a str, k: &str) -> Option<&'a str> {
    let mut it = l.split_whitespace();
    while let Some(t) = it.next() {
        if t == k {
            return it.next();
        }
    }
    None
}

unsafe fn pod<T: Copy>(b: &[u8]) -> T {
    assert!(b.len() >= std::mem::size_of::<T>());
    std::ptr::read_unaligned(b.as_ptr() as *const T)
}

struct Files {
    dir: PathBuf,
    cache: HashMap<String, Vec<u8>>,
    missing: usize,
}

impl Files {
    /// LOCAL (agent L): None when the bytes are not on disk (a capture cut while it was written)
    fn try_get(&mut self, spec: &str, n: usize) -> Option<Vec<u8>> {
        let (f, o) = spec.split_once('@').unwrap();
        let o = hex(o) as usize;
        let dir = self.dir.clone();
        let b = self.cache.entry(f.to_string()).or_insert_with(|| std::fs::read(dir.join(f)).unwrap_or_default());
        let r = b.get(o..o + n).map(|x| x.to_vec());
        if r.is_none() {
            self.missing += 1;
        }
        r
    }
}

#[derive(Clone)]
struct Range {
    w: u32,
    off: u64,
    phase: String,
    bytes: Vec<u8>,
}

struct Tex {
    id: u32,
    phase: String,
    desc: D3D12_RESOURCE_DESC,
    nsub: u32,
    bytes: Vec<u8>,
}

struct ViewL {
    idx: u32,
    uav: bool,
    res: Option<u32>,
    desc: Option<Vec<u8>>,
}

struct Pso {
    gfx: bool,
    /// LOCAL (agent R): a two-pass mesh pipeline (files: MS, PS; desc: its state sub-objects)
    #[allow(dead_code)]
    mesh: bool,
    hashes: [u64; 2],
    files: Vec<String>,
    desc: Vec<u8>,
}

struct Draw {
    k: u32,
    pso: u32,
    targets: Vec<Option<u32>>,
    state: String,
    io: Vec<u32>,
    rec: Option<[u32; 4]>,
    recs: Vec<[u32; 4]>,
}

struct Capture {
    uma_hi: u32,
    ranges: Vec<Range>,
    texs: Vec<Tex>,
    descs: HashMap<u32, D3D12_RESOURCE_DESC>,
    views: Vec<ViewL>,
    samplers: Vec<(u32, D3D12_SAMPLER_DESC)>,
    psos: HashMap<u32, Pso>,
    dispatch: Option<(u32, [u32; 3], Vec<u32>)>,
    targets: (Vec<u32>, usize),
    draws: Vec<Draw>,
    missing: usize,
    complete: Vec<String>,
    rec_info: Vec<String>,
    // LOCAL (agent R): geometry capture
    gdraws: Vec<geom::GDraw>,
    evicted: HashMap<u32, bool>,
    /// (texture, is the depth target) of every target the geometry draws had bound
    gtargets: Vec<(u32, bool)>,
    /// the lists with geometry draws, in the order they were submitted
    gexec: Vec<u64>,
    /// buffers outside the memory windows: (address, the title's bytes at its CPU view)
    oow: Vec<(u64, Vec<u8>)>,
    /// descriptors naming them: (descriptor address, buffer address, descriptor word 1)
    oowslots: Vec<(u64, u64, u32)>,
    /// root descriptors naming them: (draw, compute stage, first io word, address)
    oowio: Vec<(u32, bool, usize, u64)>,
}

/// LOCAL (agent R): with `oow=map`: where each buffer outside the windows was put (old address -> new)
static OOW_MAP: std::sync::Mutex<Option<HashMap<u64, u64>>> = std::sync::Mutex::new(None);

fn words(s: &str) -> Vec<u32> {
    s.split_whitespace().map(|w| u32::from_str_radix(w, 16).unwrap_or(0)).collect()
}

fn load(dir: &Path) -> Capture {
    let man = std::fs::read_to_string(dir.join("manifest.txt")).expect("manifest.txt");
    let mut f = Files { dir: dir.to_path_buf(), cache: HashMap::new(), missing: 0 };
    let mut c = Capture { uma_hi: 4, ranges: vec![], texs: vec![], descs: HashMap::new(), views: vec![], samplers: vec![], psos: HashMap::new(), dispatch: None, targets: (vec![], 0), draws: vec![], missing: 0, complete: vec![], rec_info: vec![], gdraws: vec![], evicted: HashMap::new(), gtargets: vec![], gexec: vec![], oow: vec![], oowslots: vec![], oowio: vec![] };
    let mut recs: HashMap<u32, Vec<[u32; 4]>> = HashMap::new();
    for l in man.lines() {
        let first = l.split_whitespace().next().unwrap_or("");
        match first {
            "uma_base_hi" => c.uma_hi = hex(tok(l, "uma_base_hi").unwrap()) as u32,
            "range" => {
                let n = hex(tok(l, "bytes").unwrap()) as usize;
                let bytes = if env("XCL_LREPLAY_SRC").is_some_and(|v| v == "cpu") && tok(l, "cpu").is_some() { f.try_get(tok(l, "cpu").unwrap(), n) } else { f.try_get(tok(l, "file").unwrap(), n) };
                let Some(bytes) = bytes else { continue };
                c.ranges.push(Range { w: tok(l, "window").unwrap().parse().unwrap(), off: hex(tok(l, "offset").unwrap()), phase: tok(l, "phase").unwrap().to_string(), bytes });
            }
            "tex" => {
                let it: Vec<&str> = l.split_whitespace().collect();
                let id: u32 = it[1].parse().unwrap();
                let desc: D3D12_RESOURCE_DESC = unsafe { pod(&unhex(tok(l, "desc").unwrap()).unwrap()) };
                c.descs.insert(id, desc);
                if let Some(e) = tok(l, "evicted") {
                    c.evicted.insert(id, e == "true");
                }
                if let (Some(file), Some(total)) = (tok(l, "file"), tok(l, "total")) {
                    let total = hex(total) as usize;
                    if let Some(bytes) = f.try_get(file, total) {
                        c.texs.push(Tex { id, phase: it[2].to_string(), desc, nsub: tok(l, "nsub").unwrap().parse().unwrap(), bytes });
                    }
                }
            }
            "view" => {
                let it: Vec<&str> = l.split_whitespace().collect();
                if it.len() < 6 || it[3] == "unknown" {
                    continue;
                }
                c.views.push(ViewL { idx: it[1].parse().unwrap(), uav: it[2] == "uav", res: it[4].parse().ok(), desc: unhex(tok(l, "desc").unwrap()) });
            }
            "sampler" => {
                let it: Vec<&str> = l.split_whitespace().collect();
                c.samplers.push((it[1].parse().unwrap(), unsafe { pod(&unhex(it[3]).unwrap()) }));
            }
            "pso" => {
                let it: Vec<&str> = l.split_whitespace().collect();
                if it[2] == "unknown" {
                    continue;
                }
                c.psos.insert(it[1].parse().unwrap(), Pso { mesh: it[2] == "mesh", gfx: it[2] == "gfx", hashes: [hex(it[3]), hex(it[4])], files: tok(l, "files").unwrap().split(',').map(String::from).collect(), desc: unhex(tok(l, "desc").unwrap()).unwrap() });
            }
            "dispatch" => {
                let it: Vec<&str> = l.split_whitespace().collect();
                let mut g = [it[7].parse().unwrap(), it[8].parse().unwrap(), it[9].parse().unwrap()];
                // LOCAL (agent L): `XCL_LREPLAY_GROUPS=x,y,z` overrides (capture L3 recorded another thread's groups)
                if let Some(v) = env("XCL_LREPLAY_GROUPS") {
                    let p: Vec<u32> = v.split(',').filter_map(|x| x.trim().parse().ok()).collect();
                    if p.len() == 3 {
                        g = [p[0], p[1], p[2]];
                    }
                }
                c.dispatch = Some((tok(l, "pso").unwrap().parse().unwrap(), g, words(l.split_once(" io ").unwrap().1)));
            }
            "targets" => {
                let it: Vec<&str> = l.split_whitespace().collect();
                c.targets = (it[1].split(',').filter_map(|v| v.parse().ok()).collect(), tok(l, "rt").unwrap().parse().unwrap());
            }
            "draw" => {
                let it: Vec<&str> = l.split_whitespace().collect();
                let (head, io) = l.split_once(" io ").unwrap();
                c.draws.push(Draw {
                    k: it[1].parse().unwrap(),
                    pso: tok(l, "pso").unwrap().parse().unwrap(),
                    targets: tok(l, "targets").unwrap().split(',').map(|v| v.parse().ok()).collect(),
                    state: head.split_once(" state ").map(|x| x.1.to_string()).unwrap_or_default(),
                    io: words(io),
                    rec: None,
                    recs: vec![],
                });
            }
            "gdraw" | "gmesh" | "grec" => geom::parse_line(first, l, &mut f, &mut c.gdraws),
            "gtarget" => {
                let it: Vec<&str> = l.split_whitespace().collect();
                c.gtargets.push((it[1].parse().unwrap(), it[2] == "ds"));
            }
            "gexec" => c.gexec.push(hex(l.split_whitespace().nth(1).unwrap())),
            "oow" => {
                if let (Some(ea), Some(n), Some(file)) = (tok(l, "ea"), tok(l, "bytes"), tok(l, "cpu")) {
                    if let Some(b) = f.try_get(file, hex(n) as usize) {
                        c.oow.push((hex(ea), b));
                    }
                }
            }
            "oowio" => {
                let it: Vec<&str> = l.split_whitespace().collect();
                c.oowio.push((it[1].parse().unwrap(), it[2] == "c", it[3].parse().unwrap(), hex(it[5])));
            }
            "oowslot" => c.oowslots.push((hex(tok(l, "at").unwrap()), hex(tok(l, "ea").unwrap()), hex(tok(l, "w1").unwrap()) as u32)),
            "records" => {
                let it: Vec<&str> = l.split_whitespace().collect();
                // LOCAL (agent L): every record of the call (stride from the line), and the GPU's count
                let n = hex(tok(l, "bytes").unwrap()) as usize;
                let stride: usize = tok(l, "stride").and_then(|v| v.parse().ok()).unwrap_or(16).max(16);
                let max: usize = tok(l, "max").and_then(|v| v.parse().ok()).unwrap_or(1);
                let Some(b) = f.try_get(tok(l, "file").unwrap(), n) else { continue };
                let count = tok(l, "count").and_then(|s| f.try_get(s, 4)).map(|c| u32::from_le_bytes(c[..4].try_into().unwrap()) as usize);
                let nrec = (n / stride).min(max).min(count.unwrap_or(usize::MAX));
                let v: Vec<[u32; 4]> = (0..nrec).map(|r| std::array::from_fn(|k| u32::from_le_bytes(b[r * stride + 4 * k..r * stride + 4 * k + 4].try_into().unwrap()))).collect();
                c.rec_info.push(format!("draw {} stride {stride} max {max} count {count:?} records {nrec}", it[1]));
                recs.insert(it[1].parse().unwrap(), v);
            }
            _ => {}
        }
    }
    for d in c.draws.iter_mut() {
        d.recs = recs.get(&d.k).cloned().unwrap_or_default();
        d.rec = d.recs.first().copied();
    }
    // LOCAL (agent R): a first-format geometry capture (run R1) names its targets in one line
    if c.gtargets.is_empty() && !c.gdraws.is_empty() {
        c.gtargets = c.targets.0.iter().enumerate().map(|(k, id)| (*id, k >= c.targets.1)).collect();
    }
    c.missing = f.missing;
    c.complete = man.lines().filter(|l| l.starts_with("complete ")).map(String::from).collect();
    c
}

fn device() -> Option<(ID3D12Device, String)> {
    unsafe {
        let factory: IDXGIFactory6 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).ok()?;
        let sel = crate::gpu::adapter::Selector::from_env_or("auto");
        let (a, desc) = crate::gpu::adapter::candidates(&factory, &sel).into_iter().next()?;
        let dev = crate::gpu::agility::create_device(&a, D3D_FEATURE_LEVEL_12_0).ok()?;
        Some((dev, crate::gpu::adapter::name(&desc)))
    }
}

fn buf(dev: &ID3D12Device, size: u64, heap: D3D12_HEAP_TYPE, state: D3D12_RESOURCE_STATES, uav: bool) -> ID3D12Resource {
    let d = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: size.max(256),
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: if uav { D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS } else { D3D12_RESOURCE_FLAG_NONE },
        ..Default::default()
    };
    let mut r: Option<ID3D12Resource> = None;
    unsafe { dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: heap, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &d, state, None, &mut r) }.unwrap();
    r.unwrap()
}

fn transition(r: &ID3D12Resource, before: D3D12_RESOURCE_STATES, after: D3D12_RESOURCE_STATES) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: unsafe { transmute_copy(r) },
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    }
}

fn uav_barrier() -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER { Type: D3D12_RESOURCE_BARRIER_TYPE_UAV, Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE, Anonymous: D3D12_RESOURCE_BARRIER_0 { UAV: ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER { pResource: ManuallyDrop::new(None) }) } }
}

struct Gpu {
    dev: ID3D12Device,
    queue: ID3D12CommandQueue,
    fence: ID3D12Fence,
    fv: u64,
    alloc: ID3D12CommandAllocator,
    keep: Vec<windows::core::IUnknown>,
}

impl Gpu {
    unsafe fn list(&self) -> ID3D12GraphicsCommandList {
        self.alloc.Reset().unwrap();
        self.dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &self.alloc, None).unwrap()
    }
    unsafe fn run(&mut self, l: ID3D12GraphicsCommandList) {
        l.Close().unwrap();
        self.queue.ExecuteCommandLists(&[Some(l.cast().unwrap())]);
        self.fv += 1;
        self.queue.Signal(&self.fence, self.fv).unwrap();
        let t0 = Instant::now();
        while self.fence.GetCompletedValue() < self.fv {
            if t0.elapsed().as_secs() > 30 {
                panic!("GPU did not finish in 30 s (device removed: {:?})", self.dev.GetDeviceRemovedReason());
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        self.keep.push(l.cast().unwrap());
    }
}

/// LOCAL (agent R): DXGI formats that can view one another (same typeless family)
fn format_family(f: i32) -> i32 {
    match f {
        1..=4 => 1,
        5..=8 => 5,
        9..=14 => 9,
        15..=18 => 15,
        19..=22 => 19,
        23..=25 => 23,
        27..=32 => 27,
        33..=38 => 33,
        39..=43 => 39,
        44..=47 => 44,
        48..=52 => 48,
        53..=59 => 53,
        60..=64 => 60,
        70..=72 => 70,
        73..=75 => 73,
        76..=78 => 76,
        79..=81 => 79,
        82..=84 => 82,
        87 | 90 | 91 => 87,
        88 | 92 | 93 => 88,
        94..=96 => 94,
        97..=99 => 97,
        x => x,
    }
}

/// LOCAL (agent R): whether a captured view description can be a view of a resource of description `d`
fn view_fits(b: &[u8], uav: bool, d: &D3D12_RESOURCE_DESC) -> bool {
    let w = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
    let (fmt, dim) = (w(0) as i32, w(4));
    let res_dim = d.Dimension.0; // 1 buffer, 2 1D, 3 2D, 4 3D
    if fmt != 0 && res_dim != 1 && format_family(fmt) != format_family(d.Format.0) {
        return false;
    }
    let want = if uav {
        match dim {
            1 => 1,
            2 | 3 => 2,
            4 | 5 => 3,
            8 => 4,
            _ => return false,
        }
    } else {
        match dim {
            1 => 1,
            2 | 3 => 2,
            4..=6 | 8 | 9 => 3,
            7 => 4,
            _ => return false,
        }
    };
    if want != res_dim {
        return false;
    }
    // mip range of a texture SRV (union at 16: most detailed mip, mip levels)
    if !uav && res_dim != 1 && matches!(dim, 4 | 5 | 7 | 8 | 9) {
        let (first, n) = (w(16), w(20));
        if first >= d.MipLevels as u32 || (n != u32::MAX && first + n > d.MipLevels as u32) {
            return false;
        }
    }
    if uav && res_dim != 1 && w(8) >= d.MipLevels as u32 {
        return false;
    }
    // a cube view needs 6 layers
    if !uav && matches!(dim, 8 | 9) && d.DepthOrArraySize < 6 {
        return false;
    }
    true
}

fn planes(f: i32) -> u32 {
    if matches!(f, 19 | 20 | 44 | 45) { 2 } else { 1 }
}

fn nsub(d: &D3D12_RESOURCE_DESC) -> u32 {
    let arr = if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D { 1 } else { d.DepthOrArraySize as u32 };
    d.MipLevels as u32 * arr * planes(d.Format.0)
}

/// read back all subresources of `r` (`state` now): bytes in footprint layout from base 0
unsafe fn read_tex(g: &mut Gpu, r: &ID3D12Resource, state: D3D12_RESOURCE_STATES) -> (Vec<u8>, Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>) {
    let d = r.GetDesc();
    let n = nsub(&d);
    let mut fps = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); n as usize];
    let mut total = 0u64;
    g.dev.GetCopyableFootprints(&d, 0, n, 0, Some(fps.as_mut_ptr()), None, None, Some(&mut total));
    let rb = buf(&g.dev, total, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST, false);
    let l = g.list();
    l.ResourceBarrier(&[transition(r, state, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
    for (k, fp) in fps.iter().enumerate() {
        let dst = D3D12_TEXTURE_COPY_LOCATION { pResource: transmute_copy(&rb), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: *fp } };
        let src = D3D12_TEXTURE_COPY_LOCATION { pResource: transmute_copy(r), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: k as u32 } };
        l.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
    }
    l.ResourceBarrier(&[transition(r, D3D12_RESOURCE_STATE_COPY_SOURCE, state)]);
    g.run(l);
    let mut q: *mut std::ffi::c_void = std::ptr::null_mut();
    rb.Map(0, None, Some(&mut q)).unwrap();
    let v = std::slice::from_raw_parts(q as *const u8, total as usize).to_vec();
    rb.Unmap(0, None);
    (v, fps)
}

fn f16(h: u16) -> f32 {
    let (s, e, m) = ((h >> 15) as u32, ((h >> 10) & 0x1F) as i32, (h & 0x3FF) as f32);
    let v = match e {
        0 => m / 1024.0 * 2f32.powi(-14),
        31 => f32::NAN,
        _ => (1.0 + m / 1024.0) * 2f32.powi(e - 15),
    };
    if s == 1 { -v } else { v }
}

/// subresource 0 of a 2D texture as rows of texel words (u32 / u64 by texel size)
fn texels(bytes: &[u8], fp: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT, bpt: usize) -> Vec<u64> {
    let (w, h, p) = (fp.Footprint.Width as usize, fp.Footprint.Height as usize, fp.Footprint.RowPitch as usize);
    let mut v = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            let o = fp.Offset as usize + y * p + x * bpt;
            let mut t = [0u8; 8];
            if o + bpt > bytes.len() {
                v.push(0);
                continue;
            }
            t[..bpt].copy_from_slice(&bytes[o..o + bpt]);
            v.push(u64::from_le_bytes(t));
        }
    }
    v
}

fn histogram(v: &[u64], fmt: &dyn Fn(u64) -> String) -> String {
    let mut m: HashMap<u64, usize> = HashMap::new();
    for &x in v {
        *m.entry(x).or_default() += 1;
    }
    let mut t: Vec<(u64, usize)> = m.into_iter().collect();
    t.sort_by_key(|x| std::cmp::Reverse(x.1));
    format!("{} distinct; top {}", t.len(), t.iter().take(8).map(|(x, c)| format!("{}×{c} ({:.1}%)", fmt(*x), 100.0 * *c as f64 / v.len() as f64)).collect::<Vec<_>>().join(" "))
}

/// RGBA16F texels -> BMP (linear, scaled so the 99th percentile of the brightest channel is 255)
fn write_hdr_bmp(path: &Path, t: &[u64], w: usize, h: usize) -> String {
    let rgb: Vec<[f32; 3]> = t.iter().map(|&x| [f16(x as u16), f16((x >> 16) as u16), f16((x >> 32) as u16)]).collect();
    let mut lum: Vec<f32> = rgb.iter().map(|c| c[0].max(c[1]).max(c[2])).filter(|v| v.is_finite() && *v > 0.0).collect();
    lum.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |x: f64| if lum.is_empty() { 0.0 } else { lum[((lum.len() - 1) as f64 * x) as usize] };
    let (p50, p99, mx) = (q(0.5), q(0.99), q(1.0));
    let scale = match env("XCL_LREPLAY_EXPOSE").and_then(|v| v.parse::<f32>().ok()) {
        Some(x) if x > 0.0 => 1.0 / x,
        _ => if p99 > 0.0 { p99 } else { 1.0 },
    };
    let row = (w * 3).div_ceil(4) * 4;
    let mut bmp = Vec::with_capacity(54 + row * h);
    bmp.extend_from_slice(b"BM");
    for v in [(54 + row * h) as u32, 0, 54, 40, w as u32, h as u32] {
        bmp.extend_from_slice(&v.to_le_bytes());
    }
    bmp.extend_from_slice(&1u16.to_le_bytes());
    bmp.extend_from_slice(&24u16.to_le_bytes());
    for _ in 0..6 {
        bmp.extend_from_slice(&0u32.to_le_bytes());
    }
    for y in (0..h).rev() {
        let start = bmp.len();
        for x in 0..w {
            let c = rgb[y * w + x];
            let g = |v: f32| if v.is_finite() { ((v / scale).clamp(0.0, 1.0) * 255.0) as u8 } else { 255 };
            bmp.extend_from_slice(&[g(c[2]), g(c[1]), g(c[0])]);
        }
        bmp.resize(start + row, 0);
    }
    let _ = std::fs::write(path, &bmp);
    format!("non-zero texels {} of {}; brightest channel p50 {p50} p99 {p99} max {mx} (BMP scaled to p99)", lum.len(), rgb.len())
}

// ---------------------------------------------------------------------------------------
// LOCAL (agent L): replay with different values, without recompiling.
// `XCL_LREPLAY_SET="<directive>;<directive>;..."` (or one per line in the file `XCL_LREPLAY_SETFILE`):
//   tex<ID>=zero | const:<hex bytes, repeated> | file:<raw file in footprint layout, e.g. a replay's *_replay.bin>
//   view<INDEX>=null | fmt:<DXGI format number>
//   mem:<window>:<hex offset>=<hex bytes>        bytes of a memory window (tables, constants, argument copies)
//   io<K>=<hex>                                  user-data word K of every draw and of the dispatch
// `XCL_LREPLAY_EXPOSE=<x>`: HDR previews multiplied by x (default: scaled so that the 99th percentile is white)
// Every run writes `<out>/summary.txt` (same lines as printed with the prefix SUMMARY).
// ---------------------------------------------------------------------------------------
fn directives() -> Vec<(String, String)> {
    let mut text = env("XCL_LREPLAY_SET").unwrap_or_default();
    if let Some(f) = env("XCL_LREPLAY_SETFILE") {
        text.push(';');
        text.push_str(&std::fs::read_to_string(f).unwrap_or_default().replace('\n', ";"));
    }
    text.split(';').filter_map(|d| d.trim().split_once('=').map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))).collect()
}

fn directive(key: &str) -> Option<String> {
    directives().into_iter().find(|(k, _)| k == key).map(|x| x.1)
}

/// the bytes a `tex<ID>=...` directive gives a texture of `n` bytes
fn tex_override(id: u32, n: usize) -> Option<Vec<u8>> {
    let v = directive(&format!("tex{id}"))?;
    let out = if v == "zero" {
        vec![0u8; n]
    } else if let Some(h) = v.strip_prefix("const:") {
        let pat = unhex(h).filter(|p| !p.is_empty())?;
        (0..n).map(|k| pat[k % pat.len()]).collect()
    } else if let Some(f) = v.strip_prefix("file:") {
        let mut b = std::fs::read(f).unwrap_or_else(|e| panic!("tex{id}=file:{f}: {e}"));
        b.resize(n, 0);
        b
    } else {
        panic!("tex{id}={v}: unknown directive value");
    };
    summary(format!("override tex{id} = {} ({n} bytes)", &v[..v.len().min(60)]));
    Some(out)
}

static SUMMARY: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

fn summary(line: String) {
    println!("SUMMARY {line}");
    SUMMARY.lock().unwrap().push(line);
}

/// RGBA16F texels: non-zero counts and percentiles of the brightest colour channel
fn hdr_stats(t: &[u64]) -> String {
    let mut lum: Vec<f32> = Vec::new();
    let (mut nz_rgb, mut nz_a, mut nan) = (0usize, 0usize, 0usize);
    for &x in t {
        let c = [f16(x as u16), f16((x >> 16) as u16), f16((x >> 32) as u16)];
        if x & 0xFFFF_FFFF_FFFF != 0 {
            nz_rgb += 1;
        }
        if x >> 48 != 0 {
            nz_a += 1;
        }
        let m = c[0].max(c[1]).max(c[2]);
        if !m.is_finite() {
            nan += 1;
        } else {
            lum.push(m.max(0.0));
        }
    }
    lum.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |x: f64| if lum.is_empty() { 0.0 } else { lum[((lum.len() - 1) as f64 * x) as usize] };
    format!("texels {} rgb-nonzero {nz_rgb} alpha-nonzero {nz_a} nan/inf {nan} p50 {:e} p90 {:e} p99 {:e} max {:e}", t.len(), q(0.5), q(0.9), q(0.99), q(1.0))
}

#[test]
fn lreplay() {
    let Some(dir) = env("XCL_LREPLAY").map(PathBuf::from) else { return };
    let t_start = Instant::now();
    let cap = load(&dir);
    let mode = env("XCL_LREPLAY_MODE").unwrap_or_else(|| "both".into());
    let out = env("XCL_LREPLAY_OUT").map(PathBuf::from).unwrap_or_else(|| dir.join("replay"));
    let _ = std::fs::create_dir_all(&out);
    println!("lreplay: {} | {} ranges, {} texture copies, {} views, {} samplers, {} pipelines, dispatch {}, {} draws; loaded in {:.1} s", dir.display(), cap.ranges.len(), cap.texs.len(), cap.views.len(), cap.samplers.len(), cap.psos.len(), cap.dispatch.is_some(), cap.draws.len(), t_start.elapsed().as_secs_f64());
    println!("lreplay: {} geometry draws (two-pass mesh)", cap.gdraws.len());
    println!("lreplay: {} entries without bytes on disk (skipped); complete lines: {:?}; records: {}", cap.missing, cap.complete, cap.rec_info.iter().take(3).cloned().collect::<Vec<_>>().join(" | "));
    let Some((dev, name)) = device() else {
        recomp::runner::skip_test("no GPU");
        return;
    };
    summary(format!("capture {} mode {mode} depth {} set [{}] draws {:?} instances {:?}", dir.display(), env("XCL_LREPLAY_DEPTH").unwrap_or_else(|| "capture".into()), directives().iter().map(|(k, v)| format!("{k}={}", &v[..v.len().min(40)])).collect::<Vec<_>>().join("; "), env("XCL_LREPLAY_DRAWS"), env("XCL_LREPLAY_INSTANCES")));
    // the inputs: every captured view, null / all-zero ones named
    let (mut nulls, mut empty, mut filled) = (Vec::new(), Vec::new(), 0);
    let pref: &[&str] = if mode == "cs" { &["csbefore", "before"] } else { &["before", "csafter", "csbefore"] };
    for v in &cap.views {
        match v.res {
            None => nulls.push(v.idx),
            Some(id) => {
                let t = pref.iter().find_map(|p| cap.texs.iter().find(|t| t.id == id && t.phase == *p));
                let over = directive(&format!("tex{id}")).is_some();
                match t {
                    Some(t) if !over && t.bytes.iter().all(|b| *b == 0) => empty.push(format!("view{}=tex{id}({}x{} f{})", v.idx, t.desc.Width, t.desc.Height, t.desc.Format.0)),
                    Some(_) => filled += 1,
                    None => empty.push(format!("view{}=tex{id}(not captured)", v.idx)),
                }
            }
        }
    }
    summary(format!("inputs: {} views, {filled} with content, {} null {:?}, {} all-zero or missing: {}", cap.views.len(), nulls.len(), nulls, empty.len(), empty.join(" ")));
    unsafe { replay(&cap, &dev, &name, &mode, &out) };
    summary(format!("total {:.1} s", t_start.elapsed().as_secs_f64()));
    let _ = std::fs::write(out.join("summary.txt"), SUMMARY.lock().unwrap().join("\n") + "\n");
    println!("LREPLAY total {:.1} s", t_start.elapsed().as_secs_f64());
}

unsafe fn replay(cap: &Capture, dev: &ID3D12Device, name: &str, mode: &str, out: &Path) {
    // LOCAL (agent R): a geometry capture is replayed by `geom::run` (no compute stage, writable targets)
    let geom = !cap.gdraws.is_empty();
    let do_cs = !geom && (mode == "cs" || mode == "both");
    let do_tl = geom || mode == "tl" || mode == "both";
    let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
    let mut g = Gpu { dev: dev.clone(), queue, fence: dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap(), fv: 0, alloc: dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap(), keep: vec![] };
    // ---- windows: the phase's ranges (the compute's first, the draws' over them)
    let phases: Vec<&str> = if do_cs { vec!["csbefore"] } else { vec!["before"] };
    let mut placed: Vec<&Range> = cap.ranges.iter().filter(|r| phases.contains(&r.phase.as_str())).collect();
    if do_cs && do_tl {
        // the draws' own argument copies and tables too (they do not overlap the compute's argument copy)
        placed.extend(cap.ranges.iter().filter(|r| r.phase == "before"));
    }
    // LOCAL (agent R): directive `oow=map`: what if the buffers outside the memory windows were readable?
    // Their bytes (the title's CPU view at the capture) go into free space of window 2 and the descriptors
    // that name them are rewritten to point there (what a mirror of the range does in the title).
    let mut extra: Vec<Range> = Vec::new();
    if geom && directive("oow").as_deref() == Some("map") {
        let w = 2u32;
        let mut at = placed.iter().filter(|r| r.w == w).map(|r| r.off + r.bytes.len() as u64).max().unwrap_or(0);
        at = ((at + 0xFFFF) & !0xFFFF) + (16 << 20);
        let mut map: HashMap<u64, u64> = HashMap::new();
        for (ea, bytes) in &cap.oow {
            if map.contains_key(ea) || at + bytes.len() as u64 >= (1 << 32) - (1 << 20) {
                continue;
            }
            map.insert(*ea, ((cap.uma_hi + w) as u64) << 32 | at);
            extra.push(Range { w, off: at, phase: "before".into(), bytes: bytes.clone() });
            at = (at + bytes.len() as u64 + 0xFFFF) & !0xFFFF;
        }
        let mut patched = 0;
        for (p, ea, w1) in &cap.oowslots {
            let Some(n) = map.get(ea) else { continue };
            let pw = ((p >> 32) as u32).wrapping_sub(cap.uma_hi);
            if pw >= 4 {
                continue;
            }
            let mut b = (*n as u32).to_le_bytes().to_vec();
            b.extend_from_slice(&((w1 & 0xFFFF_0000) | ((n >> 32) as u32 & 0xFFFF)).to_le_bytes());
            extra.push(Range { w: pw, off: p & 0xFFFF_FFFF, phase: "before".into(), bytes: b });
            patched += 1;
        }
        *OOW_MAP.lock().unwrap() = Some(map.clone());
        summary(format!("override oow=map: {} buffers outside the windows made readable ({} MB), {patched} of {} descriptors rewritten", map.len(), extra.iter().map(|r| r.bytes.len()).sum::<usize>() >> 20, cap.oowslots.len()));
    }
    let placed: Vec<&Range> = placed.iter().copied().chain(extra.iter()).collect();
    const TILE: u64 = 65536;
    const WIN: u64 = 1 << 32;
    let mut wins: Vec<ID3D12Resource> = Vec::new();
    for k in 0..4u32 {
        let d = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: WIN,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let mut r: Option<ID3D12Resource> = None;
        dev.CreateReservedResource(&d, D3D12_RESOURCE_STATE_COMMON, None, &mut r).unwrap();
        let r = r.unwrap();
        let mut tiles: Vec<u64> = Vec::new();
        for rg in placed.iter().filter(|r| r.w == k) {
            for t in rg.off / TILE..=(rg.off + rg.bytes.len() as u64).saturating_sub(1) / TILE {
                tiles.push(t);
            }
        }
        tiles.sort();
        tiles.dedup();
        if !tiles.is_empty() {
            let hd = D3D12_HEAP_DESC { SizeInBytes: tiles.len() as u64 * TILE, Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, Alignment: 0, Flags: D3D12_HEAP_FLAG_DENY_RT_DS_TEXTURES | D3D12_HEAP_FLAG_DENY_NON_RT_DS_TEXTURES };
            let mut h: Option<ID3D12Heap> = None;
            dev.CreateHeap(&hd, &mut h).unwrap();
            let h = h.unwrap();
            let coords: Vec<D3D12_TILED_RESOURCE_COORDINATE> = tiles.iter().map(|&t| D3D12_TILED_RESOURCE_COORDINATE { X: t as u32, Y: 0, Z: 0, Subresource: 0 }).collect();
            let sizes: Vec<D3D12_TILE_REGION_SIZE> = tiles.iter().map(|_| D3D12_TILE_REGION_SIZE { NumTiles: 1, UseBox: false.into(), Width: 1, Height: 1, Depth: 1 }).collect();
            let flags: Vec<D3D12_TILE_RANGE_FLAGS> = tiles.iter().map(|_| D3D12_TILE_RANGE_FLAG_NONE).collect();
            let starts: Vec<u32> = (0..tiles.len() as u32).collect();
            let counts: Vec<u32> = tiles.iter().map(|_| 1).collect();
            g.queue.UpdateTileMappings(&r, tiles.len() as u32, Some(coords.as_ptr()), Some(sizes.as_ptr()), &h, tiles.len() as u32, Some(flags.as_ptr()), Some(starts.as_ptr()), Some(counts.as_ptr()), D3D12_TILE_MAPPING_FLAG_NONE);
            g.keep.push(h.cast().unwrap());
        }
        println!("  window {k}: {} tiles mapped", tiles.len());
        wins.push(r);
    }
    let total: u64 = placed.iter().map(|r| (r.bytes.len() as u64 + 255) & !255).sum::<u64>() + 4096;
    let up = buf(dev, total, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, false);
    let mut q: *mut std::ffi::c_void = std::ptr::null_mut();
    up.Map(0, None, Some(&mut q)).unwrap();
    let l = g.list();
    for w in &wins {
        l.ResourceBarrier(&[transition(w, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST)]);
    }
    let mut at = 0u64;
    for rg in &placed {
        std::ptr::copy_nonoverlapping(rg.bytes.as_ptr(), (q as *mut u8).add(at as usize), rg.bytes.len());
        l.CopyBufferRegion(&wins[rg.w as usize], rg.off, &up, at, rg.bytes.len() as u64);
        at = (at + rg.bytes.len() as u64 + 255) & !255;
    }
    up.Unmap(0, None);
    for w in &wins {
        l.ResourceBarrier(&[transition(w, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)]);
    }
    g.run(l);
    g.keep.push(up.cast().unwrap());
    println!("  {} ranges placed ({} MB)", placed.len(), total >> 20);
    for (k, v) in directives() {
        let Some(rest) = k.strip_prefix("mem:") else { continue };
        let Some((w, off)) = rest.split_once(':') else { continue };
        let (w, off): (usize, u64) = (w.parse().unwrap(), hex(off));
        let bytes = unhex(&v).unwrap_or_default();
        let mut words = vec![0u32; bytes.len().div_ceil(4)];
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), words.as_mut_ptr() as *mut u8, bytes.len());
        let b = upload_words(&mut g, &words);
        let l = g.list();
        l.ResourceBarrier(&[transition(&b, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE), transition(&wins[w], D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST)]);
        l.CopyBufferRegion(&wins[w], off, &b, 0, bytes.len() as u64);
        l.ResourceBarrier(&[transition(&wins[w], D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)]);
        g.run(l);
        g.keep.push(b.cast().unwrap());
        summary(format!("override mem window {w} offset {off:#x}: {} bytes", bytes.len()));
    }
    // ---- textures: every resource a view or a target names; data of the first phase that has it
    let tex_pref: Vec<&str> = if do_cs { vec!["csbefore", "before", "csafter", "after"] } else { vec!["before", "csafter", "csbefore", "after"] };
    let mut ids: Vec<u32> = cap.views.iter().filter_map(|v| v.res).collect();
    ids.extend(cap.targets.0.iter().copied());
    ids.extend(cap.gtargets.iter().map(|t| t.0));
    ids.sort();
    ids.dedup();
    let (rt_ids, ds_id) = (cap.targets.0[..cap.targets.1.min(cap.targets.0.len())].to_vec(), cap.targets.0.get(cap.targets.1).copied());
    // LOCAL (agent R): a geometry capture names its targets one by one
    let (rt_ids, ds_id) = if geom { (cap.gtargets.iter().filter(|t| !t.1).map(|t| t.0).collect(), cap.gtargets.iter().find(|t| t.1).map(|t| t.0)) } else { (rt_ids, ds_id) };
    let mut res: HashMap<u32, (ID3D12Resource, D3D12_RESOURCE_STATES)> = HashMap::new();
    let mut missing = Vec::new();
    for id in ids {
        let Some(d) = cap.descs.get(&id) else {
            missing.push(id);
            continue;
        };
        let data = tex_pref.iter().find_map(|p| cap.texs.iter().find(|t| t.id == id && t.phase == *p));
        // LOCAL (agent L): `tex<ID>=...` replaces the captured content
        let over: Option<Tex> = {
            let n = nsub(d);
            let mut tot = 0u64;
            dev.GetCopyableFootprints(d, 0, n, 0, None, None, None, Some(&mut tot));
            tex_override(id, tot as usize).map(|bytes| Tex { id, phase: "override".into(), desc: *d, nsub: n, bytes })
        };
        let data = over.as_ref().or(data);
        let ds = d.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL.0 != 0;
        let mut r: Option<ID3D12Resource> = None;
        if dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, D3D12_HEAP_FLAG_NONE, d, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut r).is_err() {
            println!("  tex {id}: not created ({}x{} f{} flags {:#x})", d.Width, d.Height, d.Format.0, d.Flags.0);
            continue;
        }
        let r = r.unwrap();
        let state = if ds && geom && cap.gtargets.iter().any(|t| t.1 && t.0 == id) {
            D3D12_RESOURCE_STATE_DEPTH_WRITE
        } else if ds {
            if do_cs && !do_tl { D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE } else { D3D12_RESOURCE_STATE_DEPTH_READ | D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE }
        } else if rt_ids.contains(&id) && do_tl && !do_cs {
            D3D12_RESOURCE_STATE_RENDER_TARGET
        } else if d.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 != 0 {
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS
        } else {
            D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE
        };
        let l = g.list();
        let mut keep_up = None;
        if let Some(t) = data {
            let n = t.nsub;
            let mut fps = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); n as usize];
            let mut tot = 0u64;
            dev.GetCopyableFootprints(&t.desc, 0, n, 0, Some(fps.as_mut_ptr()), None, None, Some(&mut tot));
            let up = buf(dev, tot, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, false);
            let mut q: *mut std::ffi::c_void = std::ptr::null_mut();
            up.Map(0, None, Some(&mut q)).unwrap();
            std::ptr::copy_nonoverlapping(t.bytes.as_ptr(), q as *mut u8, t.bytes.len().min(tot as usize));
            up.Unmap(0, None);
            for (k, fp) in fps.iter().enumerate() {
                let src = D3D12_TEXTURE_COPY_LOCATION { pResource: transmute_copy(&up), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: *fp } };
                let dst = D3D12_TEXTURE_COPY_LOCATION { pResource: transmute_copy(&r), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: k as u32 } };
                l.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
            }
            keep_up = Some(up);
        } else {
            println!("  tex {id}: no captured content ({}x{} f{}): zero", d.Width, d.Height, d.Format.0);
        }
        l.ResourceBarrier(&[transition(&r, D3D12_RESOURCE_STATE_COPY_DEST, state)]);
        g.run(l);
        drop(keep_up);
        // LOCAL (agent R): name the texture whose upload loses the device
        if let Err(e) = dev.GetDeviceRemovedReason() {
            panic!("device removed after the upload of tex {id} ({}x{}x{} f{} mips {} flags {:#x} dim {}; data {:?} bytes): {e}", d.Width, d.Height, d.DepthOrArraySize, d.Format.0, d.MipLevels, d.Flags.0, d.Dimension.0, data.map(|t| t.bytes.len()));
        }
        res.insert(id, (r, state));
    }
    if !missing.is_empty() {
        println!("  resources named by views but never described (not captured): {missing:?}");
    }
    // ---- descriptor heaps
    let max_idx = cap.views.iter().map(|v| v.idx).max().unwrap_or(0).max(8) + 2;
    // LOCAL (agent R): the geometry shaders index the heap themselves: every index the title's shared heap has
    let max_idx = if geom { max_idx.max(1_000_000) } else { max_idx };
    let dh: ID3D12DescriptorHeap = dev.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, NumDescriptors: max_idx, Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, NodeMask: 0 }).unwrap();
    let inc = dev.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) as usize;
    let h_at = |i: u32| {
        let mut h = dh.GetCPUDescriptorHandleForHeapStart();
        h.ptr += i as usize * inc;
        h
    };
    for k in 0..4u32 {
        dev.CreateUnorderedAccessView(&wins[k as usize], None::<&ID3D12Resource>, Some(&recomp::runner::raw_window_uav(WIN)), h_at(recomp::MEM_HEAP_BASE + k));
    }
    let null_srv = D3D12_SHADER_RESOURCE_VIEW_DESC { Format: DXGI_FORMAT_R32_FLOAT, ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D, Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, ..Default::default() };
    let null_uav = D3D12_UNORDERED_ACCESS_VIEW_DESC { Format: DXGI_FORMAT_R32_FLOAT, ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D, ..Default::default() };
    let (mut made, mut nulls) = (0, 0);
    let mut stale: Vec<u32> = Vec::new();
    if geom {
        // a null view wherever the capture has none (an index read without a captured view reads zero)
        let t0 = Instant::now();
        for i in 4..max_idx {
            dev.CreateShaderResourceView(None::<&ID3D12Resource>, Some(&null_srv), h_at(i));
        }
        println!("  {} null views in {:.2} s", max_idx - 4, t0.elapsed().as_secs_f64());
    }
    for v in &cap.views {
        if v.idx < 4 {
            continue;
        }
        let vdir = directive(&format!("view{}", v.idx));
        let r = v.res.and_then(|id| res.get(&id)).map(|x| &x.0).filter(|_| vdir.as_deref() != Some("null"));
        // LOCAL (agent R): a view the capture's map pairs with a resource it cannot be a view of (the slot is
        // stale: its resource went and another one took its address) is a null view, counted
        let r = r.filter(|r| {
            let ok = v.desc.as_ref().is_none_or(|b| view_fits(b, v.uav, &r.GetDesc()));
            if !ok {
                stale.push(v.idx);
            }
            ok
        });
        let mut vdesc = v.desc.clone();
        if let (Some(f), Some(b)) = (vdir.as_deref().and_then(|x| x.strip_prefix("fmt:")).and_then(|x| x.parse::<u32>().ok()), vdesc.as_mut()) {
            b[..4].copy_from_slice(&f.to_le_bytes());
        }
        if let Some(x) = &vdir {
            summary(format!("override view{} = {x}", v.idx));
        }
        let v = &ViewL { idx: v.idx, uav: v.uav, res: v.res, desc: vdesc };
        if v.uav {
            let d: Option<D3D12_UNORDERED_ACCESS_VIEW_DESC> = v.desc.as_ref().map(|b| pod(b));
            match r {
                Some(r) => {
                    dev.CreateUnorderedAccessView(r, None::<&ID3D12Resource>, d.as_ref().map(|d| d as *const _), h_at(v.idx));
                    made += 1;
                }
                None => {
                    dev.CreateUnorderedAccessView(None::<&ID3D12Resource>, None::<&ID3D12Resource>, Some(&null_uav), h_at(v.idx));
                    nulls += 1;
                }
            }
        } else {
            let d: Option<D3D12_SHADER_RESOURCE_VIEW_DESC> = v.desc.as_ref().map(|b| pod(b));
            match r {
                Some(r) => {
                    dev.CreateShaderResourceView(r, d.as_ref().map(|d| d as *const _), h_at(v.idx));
                    made += 1;
                }
                None => {
                    dev.CreateShaderResourceView(None::<&ID3D12Resource>, Some(&null_srv), h_at(v.idx));
                    nulls += 1;
                }
            }
        }
        // LOCAL (agent R): name the view whose creation loses the device
        if let Err(e) = dev.GetDeviceRemovedReason() {
            let rd = v.res.and_then(|id| cap.descs.get(&id));
            panic!("device removed at view {} ({}) of tex {:?} ({:?}) desc {:?}: {e}", v.idx, if v.uav { "uav" } else { "srv" }, v.res, rd.map(|d| (d.Dimension.0, d.Width, d.Height, d.DepthOrArraySize, d.MipLevels, d.Format.0, d.Flags.0)), v.desc.as_ref().map(|b| b.iter().map(|x| format!("{x:02x}")).collect::<String>()));
        }
    }
    println!("  view heap {max_idx} descriptors: {made} views made, {nulls} null");
    if !stale.is_empty() {
        summary(format!("stale views (description does not fit the resource now at that address; made null): {} {:?}", stale.len(), &stale[..stale.len().min(40)]));
    }
    if let Err(e) = dev.GetDeviceRemovedReason() {
        panic!("device removed while the views were made: {e}");
    }
    let smax = cap.samplers.iter().map(|s| s.0).max().unwrap_or(0) + 1;
    let sh: ID3D12DescriptorHeap = dev.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, NumDescriptors: smax.min(2048), Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, NodeMask: 0 }).unwrap();
    let sinc = dev.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER) as usize;
    for (i, s) in &cap.samplers {
        if *i < 2048 {
            let mut h = sh.GetCPUDescriptorHandleForHeapStart();
            h.ptr += *i as usize * sinc;
            dev.CreateSampler(s, h);
        }
    }
    let res_buf = buf(dev, if geom { geom::res_bytes() } else { 1 << 20 }, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, true);
    let windows = 4 | recomp::HEAP_WINDOWS;
    if geom {
        geom::run(&mut g, cap, &res, &wins, &dh, &sh, &res_buf, windows, out);
        return;
    }
    // ---- the compute program
    if do_cs {
        if let Some((pid, groups, io)) = &cap.dispatch {
            let p = &cap.psos[pid];
            let blob = std::fs::read(dir_of(out).join(&p.files[0])).unwrap();
            let split = recomp::pso::wave32_only_for_device(dev);
            let t_c = Instant::now();
            let c = shader_tagged(&format!("pso{pid}_cs"), out, &blob, recomp::pso::Stage::Compute, 0, split, windows);
            let rs = recomp::pso::abi_root_signature_with(dev, false, &c.samplers).unwrap();
            let pso: ID3D12PipelineState = dev.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC { pRootSignature: transmute_copy(&rs), CS: D3D12_SHADER_BYTECODE { pShaderBytecode: c.dxil.as_ptr() as _, BytecodeLength: c.dxil.len() }, ..Default::default() }).unwrap();
            println!("  cs {:016x}: DXIL {} B, pipeline in {:.2} s, groups {groups:?}", p.hashes[0], c.dxil.len(), t_c.elapsed().as_secs_f64());
            let mut iow = io.clone();
            iow.resize(recomp::KERNEL_SGPRS, 0);
            iow[recomp::IO_UMA_BASE_HI] = cap.uma_hi;
            for (k, v) in directives() {
                if let Some(i) = k.strip_prefix("io").and_then(|x| x.parse::<usize>().ok()) {
                    iow[i] = hex(&v) as u32;
                }
            }
            let io_buf = upload_words(&mut g, &iow);
            let l = g.list();
            l.SetDescriptorHeaps(&[Some(dh.clone()), Some(sh.clone())]);
            l.SetComputeRootSignature(&rs);
            l.SetPipelineState(&pso);
            l.SetComputeRootUnorderedAccessView(0, io_buf.GetGPUVirtualAddress());
            l.SetComputeRootUnorderedAccessView(1, res_buf.GetGPUVirtualAddress());
            for k in 0..4u32 {
                l.SetComputeRootUnorderedAccessView(2 + k, wins[k as usize].GetGPUVirtualAddress());
            }
            l.Dispatch(groups[0], groups[1], groups[2]);
            l.ResourceBarrier(&[uav_barrier()]);
            let t0 = Instant::now();
            g.run(l);
            println!("  dispatched on {name} in {:.3} s", t0.elapsed().as_secs_f64());
            g.keep.push(io_buf.cast().unwrap());
            g.keep.push(pso.cast().unwrap());
            noted_indices(&mut g, &res_buf, cap, "compute");
            // the UAV textures after: compared with the game's
            let after: Vec<&Tex> = cap.texs.iter().filter(|t| t.phase == "csafter").collect();
            for t in after {
                let Some((r, st)) = res.get(&t.id) else { continue };
                let (bytes, fps) = read_tex(&mut g, r, *st);
                let _ = std::fs::write(out.join(format!("cs_tex{}_replay.bin", t.id)), &bytes);
                describe(&format!("tex {} ({}x{} f{}) after the dispatch", t.id, t.desc.Width, t.desc.Height, t.desc.Format.0), &bytes, &t.bytes, &fps, t.desc.Format.0);
                let n = bytes.len().min(t.bytes.len());
                summary(format!("cs output tex{} ({}x{} f{}): {} of {} bytes differ from the game", t.id, t.desc.Width, t.desc.Height, t.desc.Format.0, bytes[..n].iter().zip(&t.bytes[..n]).filter(|(a, b)| a != b).count(), n));
                let before = cap.texs.iter().find(|x| x.id == t.id && x.phase == "csbefore");
                if let Some(b) = before {
                    let same = b.bytes == t.bytes;
                    println!("    (game: csbefore == csafter: {same})");
                }
            }
        }
    }
    // ---- the depth before the draws
    if do_tl {
        if let Some(ds) = ds_id.and_then(|id| res.get(&id).map(|r| (id, r.0.clone(), r.1))) {
            let how = env("XCL_LREPLAY_DEPTH").unwrap_or_else(|| "capture".into());
            if how != "capture" {
                depth_from_twin(&mut g, cap, &res, ds, &how, &dh);
            }
        }
        draws(&mut g, cap, &res, &rt_ids, ds_id, &wins, &dh, &sh, &res_buf, windows, out, do_cs);
    }
}

/// `XCL_LREPLAY_NOTE=1`: the shader from its HLSL with every descriptor-heap index it reads noted
/// in a bitmap in `res` (+0x1000); else the runtime's (cached) DXIL
/// LOCAL (agent L): `tag` = "pso<ID>_vs|ps|cs". `XCL_LREPLAY_HLSL=1` writes the translated HLSL to
/// `<out>/<tag>.hlsl`; the directive `<tag>=hlsl:<file>` compiles that (edited) HLSL instead of the translation.
fn shader_tagged(tag: &str, out: &Path, blob: &[u8], stage: recomp::pso::Stage, sb: u32, split: bool, windows: u32) -> recomp::pso::Compiled {
    let over = directive(tag).and_then(|v| v.strip_prefix("hlsl:").map(String::from));
    if env("XCL_LREPLAY_HLSL").is_some() || over.is_some() {
        let (t, profile, _) = recomp::pso::console_translation(blob, stage, sb, windows, split, recomp::CompactCs::Never).unwrap();
        if env("XCL_LREPLAY_HLSL").is_some() {
            let _ = std::fs::write(out.join(format!("{tag}.hlsl")), &t.hlsl);
        }
        if let Some(f) = over {
            let h = std::fs::read_to_string(&f).unwrap_or_else(|e| panic!("{tag}=hlsl:{f}: {e}"));
            let dxil = recomp::dxc::compile(&h, "main", profile, &[]).unwrap_or_else(|e| panic!("{tag}: the given HLSL does not compile: {e}"));
            summary(format!("override {tag} = hlsl:{f} ({} B DXIL)", dxil.len()));
            return recomp::pso::Compiled { dxil, samplers: t.static_samplers.clone() };
        }
    }
    shader(blob, stage, sb, split, windows)
}

fn shader(blob: &[u8], stage: recomp::pso::Stage, sb: u32, split: bool, windows: u32) -> recomp::pso::Compiled {
    if env("XCL_LREPLAY_NOTE").is_none() {
        return recomp::pso::console_to_dxil_in(blob, stage, sb, split, windows).unwrap();
    }
    let (t, profile, _) = recomp::pso::console_translation(blob, stage, sb, windows, split, recomp::CompactCs::Never).unwrap();
    let mut h = String::new();
    let pat = "ResourceDescriptorHeap[NonUniformResourceIndex(";
    let mut rest = t.hlsl.as_str();
    while let Some(i) = rest.find(pat) {
        h += &rest[..i + pat.len()];
        rest = &rest[i + pat.len()..];
        // the argument up to its closing parenthesis
        let mut depth = 1;
        let mut end = 0;
        for (k, ch) in rest.char_indices() {
            match ch {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        end = k;
                        break;
                    }
                }
                _ => {}
            }
        }
        h += &format!("xcl_note({})", &rest[..end]);
        rest = &rest[end..];
    }
    h += rest;
    let note = "uint xcl_note(uint i) { if (i >= 4u && i < 0x100000u) { res.InterlockedOr(0x1000u + (i >> 5) * 4u, 1u << (i & 31u)); } return i; }
";
    let at = h.find("
void main(").or_else(|| h.find("
[")).map(|k| k + 1).unwrap_or(0);
    // after the declarations of io / res
    let decl = h.find("RWByteAddressBuffer res : register(u1);").map(|k| k + "RWByteAddressBuffer res : register(u1);".len() + 1).unwrap_or(at);
    h.insert_str(decl, note);
    let dxil = recomp::dxc::compile(&h, "main", profile, &[]).unwrap_or_else(|e| panic!("noted HLSL does not compile: {e}"));
    recomp::pso::Compiled { dxil, samplers: t.static_samplers.clone() }
}

unsafe fn noted_indices(g: &mut Gpu, res_buf: &ID3D12Resource, cap: &Capture, what: &str) {
    if env("XCL_LREPLAY_NOTE").is_none() {
        return;
    }
    let rb = buf(&g.dev, 1 << 20, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST, false);
    let l = g.list();
    l.ResourceBarrier(&[transition(res_buf, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
    l.CopyBufferRegion(&rb, 0, res_buf, 0, 1 << 20);
    l.ResourceBarrier(&[transition(res_buf, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)]);
    g.run(l);
    let mut q: *mut std::ffi::c_void = std::ptr::null_mut();
    rb.Map(0, None, Some(&mut q)).unwrap();
    let w = std::slice::from_raw_parts((q as *const u8).add(0x1000) as *const u32, 0x8000);
    let mut used = Vec::new();
    for (k, &b) in w.iter().enumerate() {
        for i in 0..32 {
            if b >> i & 1 != 0 {
                used.push((k * 32 + i) as u32);
            }
        }
    }
    rb.Unmap(0, None);
    let known: Vec<u32> = used.iter().copied().filter(|i| cap.views.iter().any(|v| v.idx == *i && v.res.is_some())).collect();
    let unknown: Vec<u32> = used.iter().copied().filter(|i| !known.contains(i)).collect();
    println!("  {what}: descriptor-heap indices read: {} ({} with a captured view; without: {:?})", used.len(), known.len(), &unknown[..unknown.len().min(40)]);
    for i in known.iter().take(80) {
        let v = cap.views.iter().find(|v| v.idx == *i).unwrap();
        let d = v.res.and_then(|id| cap.descs.get(&id));
        print!(" {i}:{}{}", v.res.unwrap(), d.map(|d| format!("({}x{} f{})", d.Width, d.Height, d.Format.0)).unwrap_or_default());
    }
    println!();
    // cleared for the next stage
    let z = vec![0u32; 0x8000];
    let up = upload_words(g, &z);
    let l = g.list();
    l.ResourceBarrier(&[transition(&up, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
    l.CopyBufferRegion(res_buf, 0x1000, &up, 0, 0x20000);
    l.ResourceBarrier(&[uav_barrier()]);
    g.run(l);
    g.keep.push(up.cast().unwrap());
}

fn dir_of(out: &Path) -> PathBuf {
    env("XCL_LREPLAY").map(PathBuf::from).unwrap_or_else(|| out.to_path_buf())
}

unsafe fn upload_words(g: &mut Gpu, w: &[u32]) -> ID3D12Resource {
    let n = (w.len() * 4) as u64;
    let up = buf(&g.dev, n, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, false);
    let b = buf(&g.dev, n, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON, true);
    let mut q: *mut std::ffi::c_void = std::ptr::null_mut();
    up.Map(0, None, Some(&mut q)).unwrap();
    std::ptr::copy_nonoverlapping(w.as_ptr(), q as *mut u32, w.len());
    up.Unmap(0, None);
    let l = g.list();
    l.CopyBufferRegion(&b, 0, &up, 0, n);
    l.ResourceBarrier(&[transition(&b, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)]);
    g.run(l);
    g.keep.push(up.cast().unwrap());
    b
}

/// stats of a replayed texture vs the game's bytes
fn describe(what: &str, replay: &[u8], game: &[u8], fps: &[D3D12_PLACED_SUBRESOURCE_FOOTPRINT], fmt: i32) {
    let bpt = match fmt {
        2 | 1 | 3 | 4 => 16,
        10 | 9 | 11..=14 | 16 => 8,
        49..=59 => 2,
        60..=65 => 1,
        _ => 4,
    }
    .min(8);
    let fp = &fps[0];
    let r = texels(replay, fp, bpt);
    let gm = if game.len() >= replay.len() { texels(game, fp, bpt) } else { vec![] };
    let differ = if gm.is_empty() { usize::MAX } else { r.iter().zip(&gm).filter(|(a, b)| a != b).count() };
    let fmtv = |x: u64| -> String {
        match fmt {
            39 | 41 => format!("{:08x}({})", x as u32, f32::from_bits(x as u32)),
            _ => format!("{x:x}"),
        }
    };
    println!("    {what}: replay {}", histogram(&r, &fmtv));
    if !gm.is_empty() {
        println!("    {what}: game   {}", histogram(&gm, &fmtv));
        println!("    {what}: {differ} of {} texels differ (plane 0, mip 0)", r.len());
    }
    if fps.len() > 1 {
        let fp1 = &fps[1];
        let bpt1 = if fp1.Footprint.Format.0 == 62 || fp1.Footprint.Format.0 == 61 { 1 } else { 4 };
        let r1 = texels(replay, fp1, bpt1);
        println!("    {what}: plane/sub 1 (f{}) replay {}", fp1.Footprint.Format.0, histogram(&r1, &|x| format!("{x:x}")));
        if game.len() >= replay.len() {
            println!("    {what}: plane/sub 1 game   {}", histogram(&texels(game, fp1, bpt1), &|x| format!("{x:x}")));
        }
    }
}

/// writes the second depth (D24S8 `ds`) from its depth twin (the R32 twin the mask compute wrote):
/// `twin` = its float as depth, `twin16` = its bits as an integer k, depth k / 65535
unsafe fn depth_from_twin(g: &mut Gpu, cap: &Capture, res: &HashMap<u32, (ID3D12Resource, D3D12_RESOURCE_STATES)>, ds: (u32, ID3D12Resource, D3D12_RESOURCE_STATES), how: &str, _dh: &ID3D12DescriptorHeap) {
    let (ds_id, ds_r, ds_state) = ds;
    let d = ds_r.GetDesc();
    // the depth twin: an R32 texture of the same size reached by a UAV view of the compute
    let twin = cap.views.iter().filter(|v| v.uav).filter_map(|v| v.res).find(|id| *id != ds_id && cap.descs.get(id).is_some_and(|t| t.Width == d.Width && t.Height == d.Height && matches!(t.Format.0, 53 | 54 | 55 | 56)))
        // LOCAL (agent L): the 16-bit twin first (capture L3: the depth twin is R16 typeless; the R32_UINT
        // texture of the same size is the lighting's second render target), then an R32 one
        .or_else(|| cap.views.iter().filter(|v| v.uav).filter_map(|v| v.res).find(|id| *id != ds_id && cap.descs.get(id).is_some_and(|t| t.Width == d.Width && t.Height == d.Height && matches!(t.Format.0, 39 | 41))));
    let twin16 = twin.and_then(|id| cap.descs.get(&id)).is_some_and(|t| matches!(t.Format.0, 53 | 54 | 55 | 56));
    let Some(twin) = twin.and_then(|id| res.get(&id).map(|r| (id, r.0.clone(), r.1))) else {
        println!("  depth from twin: no R32 twin found");
        return;
    };
    let (twin_bytes, fps) = read_tex(g, &twin.1, twin.2);
    let t = texels(&twin_bytes, &fps[0], if twin16 { 2 } else { 4 });
    let (w, h) = (d.Width as usize, d.Height as usize);
    // the D24S8 plane 0 bytes (24 bit depth in the low bits of each 32-bit word)
    let (mut dsb, dfps) = read_tex(g, &ds_r, ds_state);
    let fp = &dfps[0];
    let mut hist: HashMap<u32, usize> = HashMap::new();
    for y in 0..h {
        for x in 0..w {
            let v = t[y * w + x] as u32;
            let depth = match how {
                _ if twin16 => (v as f64 / 65535.0).clamp(0.0, 1.0),
                "twin16" => (v as f64 / 65535.0).clamp(0.0, 1.0),
                _ => (f32::from_bits(v) as f64).clamp(0.0, 1.0),
            };
            let u24 = (depth * 16777215.0).round() as u32;
            *hist.entry(u24).or_default() += 1;
            let o = fp.Offset as usize + y * fp.Footprint.RowPitch as usize + 4 * x;
            let old = u32::from_le_bytes(dsb[o..o + 4].try_into().unwrap());
            dsb[o..o + 4].copy_from_slice(&((old & 0xFF00_0000) | u24).to_le_bytes());
        }
    }
    let mut top: Vec<(u32, usize)> = hist.into_iter().collect();
    top.sort_by_key(|x| std::cmp::Reverse(x.1));
    println!("  depth from twin {} ({how}): {} distinct depths, top {:?}", twin.0, top.len(), top.iter().take(6).map(|(u, c)| (format!("{:.6}", *u as f64 / 16777215.0), *c)).collect::<Vec<_>>());
    // upload
    let mut tot = 0u64;
    let n = dfps.len() as u32;
    let mut fps2 = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); n as usize];
    g.dev.GetCopyableFootprints(&d, 0, n, 0, Some(fps2.as_mut_ptr()), None, None, Some(&mut tot));
    let up = buf(&g.dev, tot, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, false);
    let mut q: *mut std::ffi::c_void = std::ptr::null_mut();
    up.Map(0, None, Some(&mut q)).unwrap();
    std::ptr::copy_nonoverlapping(dsb.as_ptr(), q as *mut u8, dsb.len().min(tot as usize));
    up.Unmap(0, None);
    let l = g.list();
    l.ResourceBarrier(&[transition(&ds_r, ds_state, D3D12_RESOURCE_STATE_COPY_DEST)]);
    for (k, fp) in fps2.iter().enumerate() {
        let src = D3D12_TEXTURE_COPY_LOCATION { pResource: transmute_copy(&up), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: *fp } };
        let dst = D3D12_TEXTURE_COPY_LOCATION { pResource: transmute_copy(&ds_r), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: k as u32 } };
        l.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
    }
    l.ResourceBarrier(&[transition(&ds_r, D3D12_RESOURCE_STATE_COPY_DEST, ds_state)]);
    g.run(l);
    g.keep.push(up.cast().unwrap());
}

fn parse_vp(state: &str) -> Option<(f32, f32, f32, f32, f32, f32)> {
    // "vp (x,y WxH z a..b)"
    let s = state.split_once("vp (")?.1;
    let s = s.split_once(')')?.0;
    let (xy, rest) = s.split_once(' ')?;
    let (x, y) = xy.split_once(',')?;
    let (wh, z) = rest.split_once(" z ")?;
    let (w, h) = wh.split_once('x')?;
    let (a, b) = z.split_once("..")?;
    Some((x.parse().ok()?, y.parse().ok()?, w.parse().ok()?, h.parse().ok()?, a.parse().ok()?, b.parse().ok()?))
}

/// LOCAL (agent L): "sc (l,t)-(r,b)", "topo Some(n)", "sref Some(n)", "bf Some([a, b, c, d])" of the state text
fn parse_sc(state: &str) -> Option<[i32; 4]> {
    let s = state.split_once(" sc (")?.1;
    let (a, rest) = s.split_once(")-(")?;
    let b = rest.split_once(')')?.0;
    let (l, t) = a.split_once(',')?;
    let (r, bt) = b.split_once(',')?;
    Some([l.trim().parse().ok()?, t.trim().parse().ok()?, r.trim().parse().ok()?, bt.trim().parse().ok()?])
}

fn parse_some_u32(state: &str, key: &str) -> Option<u32> {
    let s = state.split_once(key)?.1;
    let s = s.strip_prefix("Some(")?;
    s.split_once(')')?.0.trim().parse().ok()
}

fn parse_bf(state: &str) -> Option<[f32; 4]> {
    let s = state.split_once(" bf Some([")?.1;
    let s = s.split_once(']')?.0;
    let v: Vec<f32> = s.split(',').filter_map(|x| x.trim().parse().ok()).collect();
    (v.len() == 4).then(|| [v[0], v[1], v[2], v[3]])
}

#[allow(clippy::too_many_arguments)]
unsafe fn draws(
    g: &mut Gpu,
    cap: &Capture,
    res: &HashMap<u32, (ID3D12Resource, D3D12_RESOURCE_STATES)>,
    rt_ids: &[u32],
    ds_id: Option<u32>,
    wins: &[ID3D12Resource],
    dh: &ID3D12DescriptorHeap,
    sh: &ID3D12DescriptorHeap,
    res_buf: &ID3D12Resource,
    windows: u32,
    out: &Path,
    after_cs: bool,
) {
    let dev = g.dev.clone();
    let split = recomp::pso::graphics_wave32_for_device(&dev);
    let range = env("XCL_LREPLAY_DRAWS").and_then(|r| {
        let (a, b) = r.split_once("..")?;
        Some((a.parse::<u32>().ok()?, b.parse::<u32>().ok()?))
    });
    let force_inst: Option<u32> = env("XCL_LREPLAY_INSTANCES").and_then(|v| v.parse().ok());
    // targets: RTVs / DSV
    let rtvh: ID3D12DescriptorHeap = dev.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV, NumDescriptors: 8, ..Default::default() }).unwrap();
    let dsvh: ID3D12DescriptorHeap = dev.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_DSV, NumDescriptors: 1, ..Default::default() }).unwrap();
    let rinc = dev.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV) as usize;
    // pipelines (by capture id), the first draw's PSO gives the target formats
    let mut psos: HashMap<u32, (ID3D12RootSignature, ID3D12PipelineState, D3D12_GRAPHICS_PIPELINE_STATE_DESC)> = HashMap::new();
    let t_p = Instant::now();
    for d in &cap.draws {
        if psos.contains_key(&d.pso) {
            continue;
        }
        // LOCAL (agent L): only the pipelines of the draws that will draw (range, instances), and always the first
        let will = !range.is_some_and(|(a, b)| d.k < a || d.k > b) && d.recs.iter().any(|r| r[0] > 0 && force_inst.unwrap_or(r[1]) > 0);
        if !will && !psos.is_empty() && env("XCL_LREPLAY_ALL_PSOS").is_none() {
            continue;
        }
        let Some(p) = cap.psos.get(&d.pso) else { continue };
        let dir = dir_of(out);
        let vs = std::fs::read(dir.join(&p.files[0])).unwrap();
        let ps = p.files.get(1).map(|f| std::fs::read(dir.join(f)).unwrap());
        let vc = shader_tagged(&format!("pso{}_vs", d.pso), out, &vs, recomp::pso::Stage::Vertex, 0, split, windows);
        let mut samplers = vc.samplers.clone();
        let pc = ps.as_ref().map(|ps| shader_tagged(&format!("pso{}_ps", d.pso), out, ps, recomp::pso::Stage::Pixel, samplers.len() as u32, split, windows));
        if let Some(pc) = &pc {
            samplers.extend(pc.samplers.iter().cloned());
        }
        let rs = recomp::pso::abi_root_signature_with(&dev, true, &samplers).unwrap();
        assert!(p.desc.len() >= std::mem::size_of::<D3D12_GRAPHICS_PIPELINE_STATE_DESC>());
        let mut desc: D3D12_GRAPHICS_PIPELINE_STATE_DESC = std::ptr::read_unaligned(p.desc.as_ptr() as *const D3D12_GRAPHICS_PIPELINE_STATE_DESC);
        std::ptr::write(&mut desc.pRootSignature, transmute_copy(&rs));
        desc.VS = D3D12_SHADER_BYTECODE { pShaderBytecode: vc.dxil.as_ptr() as _, BytecodeLength: vc.dxil.len() };
        desc.PS = pc.as_ref().map_or(D3D12_SHADER_BYTECODE::default(), |c| D3D12_SHADER_BYTECODE { pShaderBytecode: c.dxil.as_ptr() as _, BytecodeLength: c.dxil.len() });
        // LOCAL (agent L): as `real::translated_graphics_pso`: a rect list pipeline (console topology type 19)
        // gets the generated GS that completes each rect (without it: the triangle of its three vertices)
        let rect_gs = if desc.PrimitiveTopologyType.0 == 19 && env("XCL_LREPLAY_NO_RECT_GS").is_none() { recomp::pso::rect_list_gs(&vs).ok() } else { None };
        desc.GS = rect_gs.as_ref().map_or(D3D12_SHADER_BYTECODE::default(), |g| D3D12_SHADER_BYTECODE { pShaderBytecode: g.as_ptr() as _, BytecodeLength: g.len() });
        if psos.is_empty() {
            println!("  first pipeline: console topology type {}, rect list GS {}", desc.PrimitiveTopologyType.0, rect_gs.as_ref().map_or("none".to_string(), |g| format!("{} B", g.len())));
        }
        desc.HS = D3D12_SHADER_BYTECODE::default();
        desc.DS = D3D12_SHADER_BYTECODE::default();
        std::ptr::write(&mut desc.StreamOutput, std::mem::zeroed());
        desc.InputLayout = D3D12_INPUT_LAYOUT_DESC::default();
        desc.CachedPSO = D3D12_CACHED_PIPELINE_STATE::default();
        desc.NodeMask = 0;
        super::real::pc_topology_type(&mut desc);
        let pso: ID3D12PipelineState = match dev.CreateGraphicsPipelineState(&desc) {
            Ok(p) => p,
            Err(e) => {
                println!("  pso {}: not built: {e}", d.pso);
                continue;
            }
        };
        if psos.is_empty() {
            let z = &desc.DepthStencilState;
            println!(
                "  first pipeline vs {:016x} ps {:016x}: topology type {} RTs {:?} DSV {} depth (enable {}, write {}, func {}) stencil {} blend0 {} cull {}",
                p.hashes[0],
                p.hashes[1],
                desc.PrimitiveTopologyType.0,
                &desc.RTVFormats[..desc.NumRenderTargets as usize].iter().map(|f| f.0).collect::<Vec<_>>(),
                desc.DSVFormat.0,
                z.DepthEnable.as_bool(),
                z.DepthWriteMask.0,
                z.DepthFunc.0,
                z.StencilEnable.as_bool(),
                desc.BlendState.RenderTarget[0].BlendEnable.as_bool(),
                desc.RasterizerState.CullMode.0
            );
        }
        g.keep.push(rs.cast().unwrap());
        psos.insert(d.pso, (rs, pso, desc));
    }
    println!("  {} graphics pipelines in {:.1} s", psos.len(), t_p.elapsed().as_secs_f64());
    let Some(first) = cap.draws.first().and_then(|d| psos.get(&d.pso)) else {
        println!("  no draw pipeline");
        return;
    };
    let fdesc = &first.2;
    let mut rtvs = Vec::new();
    for (k, id) in rt_ids.iter().enumerate() {
        let (r, _) = &res[id];
        let mut h = rtvh.GetCPUDescriptorHandleForHeapStart();
        h.ptr += k * rinc;
        let v = D3D12_RENDER_TARGET_VIEW_DESC { Format: fdesc.RTVFormats[k], ViewDimension: D3D12_RTV_DIMENSION_TEXTURE2D, ..Default::default() };
        dev.CreateRenderTargetView(r, Some(&v), h);
        rtvs.push(h);
    }
    let dsv = ds_id.map(|id| {
        let h = dsvh.GetCPUDescriptorHandleForHeapStart();
        let v = D3D12_DEPTH_STENCIL_VIEW_DESC { Format: fdesc.DSVFormat, ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2D, Flags: D3D12_DSV_FLAG_READ_ONLY_DEPTH | D3D12_DSV_FLAG_READ_ONLY_STENCIL, ..Default::default() };
        dev.CreateDepthStencilView(&res[&id].0, Some(&v), h);
        h
    });
    // io of every draw in one buffer
    let n = cap.draws.len();
    let mut all = vec![0u32; n * recomp::KERNEL_SGPRS];
    for (k, d) in cap.draws.iter().enumerate() {
        let w = &mut all[k * recomp::KERNEL_SGPRS..(k + 1) * recomp::KERNEL_SGPRS];
        for (i, v) in d.io.iter().take(recomp::KERNEL_SGPRS).enumerate() {
            w[i] = *v;
        }
        w[recomp::IO_UMA_BASE_HI] = cap.uma_hi;
        for (dk, v) in directives() {
            if let Some(i) = dk.strip_prefix("io").and_then(|x| x.parse::<usize>().ok()) {
                w[i] = hex(&v) as u32;
            }
        }
    }
    let io_buf = upload_words(g, &all);
    let l = g.list();
    // the targets to RENDER_TARGET (from what the setup gave them)
    for id in rt_ids {
        let (r, st) = &res[id];
        if *st != D3D12_RESOURCE_STATE_RENDER_TARGET {
            l.ResourceBarrier(&[transition(r, *st, D3D12_RESOURCE_STATE_RENDER_TARGET)]);
        }
    }
    l.SetDescriptorHeaps(&[Some(dh.clone()), Some(sh.clone())]);
    l.OMSetRenderTargets(rtvs.len() as u32, Some(rtvs.as_ptr()), false, dsv.as_ref().map(|h| h as *const _));
    let (mut drawn, mut inst_sum, mut skipped) = (0, 0u64, 0);
    let mut vp_text = String::new();
    for (k, d) in cap.draws.iter().enumerate() {
        if range.is_some_and(|(a, b)| d.k < a || d.k > b) {
            continue;
        }
        let Some((rs, pso, _)) = psos.get(&d.pso) else {
            if d.recs.iter().any(|r| r[0] > 0 && force_inst.unwrap_or(r[1]) > 0) {
                skipped += 1;
            }
            continue;
        };
        if d.recs.iter().any(|r| r[0] > 0 && force_inst.unwrap_or(r[1]) > 0) {
            let p = &cap.psos[&d.pso];
            summary(format!("draw {} pso{} vs {:016x} ps {:016x} record {:x?}", d.k, d.pso, p.hashes[0], p.hashes[1], d.recs.first()));
        }
        let Some(rec) = d.rec else {
            skipped += 1;
            continue;
        };
        let inst = force_inst.unwrap_or(rec[1]);
        let vp = parse_vp(&d.state).unwrap_or((0.0, 0.0, 2560.0, 1440.0, 0.0, 1.0));
        if k == 0 {
            vp_text = format!("viewport {vp:?}, state '{}', record {rec:x?}", d.state);
        }
        l.SetGraphicsRootSignature(rs);
        l.SetPipelineState(pso);
        l.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY(parse_some_u32(&d.state, " topo ").unwrap_or(4) as i32));
        l.RSSetViewports(&[D3D12_VIEWPORT { TopLeftX: vp.0, TopLeftY: vp.1, Width: vp.2, Height: vp.3, MinDepth: vp.4, MaxDepth: vp.5 }]);
        let sc = parse_sc(&d.state).unwrap_or([0, 0, 16384, 16384]);
        l.RSSetScissorRects(&[windows::Win32::Foundation::RECT { left: sc[0], top: sc[1], right: sc[2], bottom: sc[3] }]);
        if let Some(r) = parse_some_u32(&d.state, " sref ") {
            l.OMSetStencilRef(r);
        }
        if let Some(b) = parse_bf(&d.state) {
            l.OMSetBlendFactor(Some(&b));
        }
        l.SetGraphicsRootUnorderedAccessView(0, io_buf.GetGPUVirtualAddress() + (k * recomp::KERNEL_SGPRS * 4) as u64);
        l.SetGraphicsRootUnorderedAccessView(1, res_buf.GetGPUVirtualAddress());
        for w in 0..4u32 {
            l.SetGraphicsRootUnorderedAccessView(2 + w, wins[w as usize].GetGPUVirtualAddress());
        }
        let _ = (rec, inst);
        for rec in &d.recs {
            let inst = force_inst.unwrap_or(rec[1]);
            if inst > 0 && rec[0] > 0 {
                l.DrawInstanced(rec[0], inst, rec[2], rec[3]);
                drawn += 1;
                inst_sum += inst as u64;
            }
        }
    }
    println!("  first draw: {vp_text}");
    let t0 = Instant::now();
    g.run(l);
    noted_indices(g, res_buf, cap, "draws");
    println!("  {drawn} draws with instances ({inst_sum} instances), {skipped} without a pipeline or record, of {n}; GPU {:.3} s (after the dispatch: {after_cs})", t0.elapsed().as_secs_f64());
    summary(format!("draws: {drawn} with instances ({inst_sum} instances) of {n} captured, {skipped} without a pipeline or record"));
    // outputs: each target vs the game's 'after'
    for id in rt_ids.iter().chain(ds_id.iter()) {
        let (r, st) = &res[id];
        let st = if rt_ids.contains(id) { D3D12_RESOURCE_STATE_RENDER_TARGET } else { *st };
        let (bytes, fps) = read_tex(g, r, st);
        let d = r.GetDesc();
        let _ = std::fs::write(out.join(format!("tl_tex{id}_replay.bin"), ), &bytes);
        let game = cap.texs.iter().find(|t| t.id == *id && t.phase == "after").map(|t| t.bytes.clone()).unwrap_or_default();
        let before = cap.texs.iter().find(|t| t.id == *id && t.phase == "before").map(|t| t.bytes.clone()).unwrap_or_default();
        describe(&format!("target {id} ({}x{} f{})", d.Width, d.Height, d.Format.0), &bytes, &game, &fps, d.Format.0);
        if game.len() >= bytes.len() {
            summary(format!("target tex{id} ({}x{} f{}): {} of {} bytes differ from the game", d.Width, d.Height, d.Format.0, bytes.iter().zip(&game).filter(|(a, b)| a != b).count(), bytes.len()));
        } else {
            summary(format!("target tex{id} ({}x{} f{}): no game bytes after the draws in this capture", d.Width, d.Height, d.Format.0));
        }
        if d.Format.0 == 10 {
            summary(format!("HDR tex{id} replay: {}", hdr_stats(&texels(&bytes, &fps[0], 8))));
            if game.len() >= bytes.len() {
                summary(format!("HDR tex{id} game:   {}", hdr_stats(&texels(&game, &fps[0], 8))));
            }
            if before.len() >= bytes.len() {
                summary(format!("HDR tex{id} before: {}", hdr_stats(&texels(&before, &fps[0], 8))));
            }
        }
        if d.Format.0 == 10 {
            let (w, h) = (fps[0].Footprint.Width as usize, fps[0].Footprint.Height as usize);
            let s = write_hdr_bmp(&out.join(format!("tl_tex{id}_replay.bmp")), &texels(&bytes, &fps[0], 8), w, h);
            println!("    HDR replay: {s} -> tl_tex{id}_replay.bmp");
            for (tag, b) in [("game_after", &game), ("game_before", &before)] {
                if b.len() >= bytes.len() {
                    let s = write_hdr_bmp(&out.join(format!("tl_tex{id}_{tag}.bmp")), &texels(b, &fps[0], 8), w, h);
                    println!("    HDR {tag}: {s} -> tl_tex{id}_{tag}.bmp");
                }
            }
        }
    }
    g.keep.push(io_buf.cast().unwrap());
}
