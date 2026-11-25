//! LOCAL (agents H, L, V): off-game replay of a lighting capture (`lcap.rs`, `XCL_LOCAL_LCAP`), as an
//! ordinary binary (a cargo example of the `xcl` package: it links the `xcl` library, so editing
//! this file never recompiles XCL, and it is not part of the crate's test binary).
//!
//! Build: `cargo build --profile replay -p xcl --example lreplay` (profile in `.cargo/config.toml`).
//! Run:   `XCL_HEAP_WINDOWS=1 XCL_LREPLAY=<capture dir> lreplay.exe`
//! - `XCL_LREPLAY_MODE=cs|tl|both` (default both): the material-mask dispatch, the tiled-lighting
//!   draws, or the dispatch then the draws on the same resources (the draws then see what the
//!   replayed dispatch wrote into the twins, and `XCL_LREPLAY_DEPTH` decides f44)
//! - `XCL_LREPLAY_DEPTH=capture|twin|twin16` the second depth before the draws: as captured (default),
//!   the depth twin's R32 values written into it as depth (`twin`), or as k/65535 of the twin's
//!   value read as an integer (`twin16`)
//! - `XCL_LREPLAY_INSTANCES=<n>`: every draw's instance count forced to n (default: the record's)
//! - `XCL_LREPLAY_OUT=<dir>` (default `<capture>/replay`): outputs (PNG previews, summary.txt, raw targets)
//! - `XCL_LREPLAY_DRAWS=a..b` only those draws
//! - `XCL_LREPLAY_SET` / `XCL_LREPLAY_SETFILE`: other values without recompiling (see `directives`)
//! - `XCL_LREPLAY_BIN=0|1` raw `*_replay.bin` targets, `XCL_LREPLAY_DESCRIBE=0|1` value histograms in
//!   the log (both: on for a single replay, off in the persistent mode), `XCL_LREPLAY_BMP=1` full-size
//!   BMP previews besides the PNGs, `XCL_LREPLAY_PNG_SCALE=<k>` (default 2: every k-th texel)
//!
//! Persistent mode (agent V): `XCL_LREPLAY_SERVE=stdin` or `=<control dir>`. The process loads the
//! capture and creates the device, the resources and the pipelines ONCE, replays once with its
//! environment, prints `LREPLAY ready`, then replays again for every command:
//! - `stdin`: one command per line; a control dir: every `<name>.cmd` file put there (write it under
//!   another name and rename it), answered by `<name>.done` (the SUMMARY lines, then `OK` / `ERROR`).
//! - a command is a list of directives separated by `;` in the format of `XCL_LREPLAY_SET`; a
//!   directive `XCL_LREPLAY_<X>=<value>` gives that variable this value for this replay only and
//!   `out=<dir>` is short for `XCL_LREPLAY_OUT`. Every command starts again from the process
//!   environment (nothing is remembered from the previous command). `quit` ends the process.
//! - the process ends by itself when a file named in `XCL_LREPLAY_YIELD=<path>;<path>` exists, when a
//!   process `xcl.exe` runs (`XCL_LREPLAY_YIELD_PROC`, `none` to turn that off) or after
//!   `XCL_LREPLAY_IDLE_S` seconds without a command (default 900): it must not hold the GPU while
//!   a title runs.
//! Between two replays only what changed is redone: written resources are restored from upload
//! buffers kept alive, shaders are kept by tag and source text, pipelines by their DXIL.
//!
//! The memory windows are rebuilt as in agent R's `iso_cull`: 4 GiB reserved buffers with the
//! captured ranges' tiles mapped at their own offsets (no relocation). Textures are recreated
//! with their PC descriptions and filled with the captured bytes; every captured view is made
//! at its shared-heap index with its PC description; samplers at their sampler-heap index.

use std::collections::{HashMap, HashSet};
use std::mem::{transmute_copy, ManuallyDrop};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D::*;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::*;

use xcl::gpu::recomp;
use xcl::hle::d3d12x::lreplay_api as api;

/// per-replay values of `XCL_LREPLAY_*` variables (persistent mode), before the process environment
static TRIAL_ENV: Mutex<Vec<(String, String)>> = Mutex::new(Vec::new());

fn env(name: &str) -> Option<String> {
    if let Some(v) = TRIAL_ENV.lock().unwrap().iter().find(|(k, _)| k == name).map(|x| x.1.clone()) {
        return Some(v).filter(|v| !v.trim().is_empty());
    }
    std::env::var(name).ok().filter(|v| !v.trim().is_empty())
}

fn flag(name: &str, default: bool) -> bool {
    match env(name).as_deref() {
        Some("0") => false,
        Some(_) => true,
        None => default,
    }
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
}

fn words(s: &str) -> Vec<u32> {
    s.split_whitespace().map(|w| u32::from_str_radix(w, 16).unwrap_or(0)).collect()
}

fn load(dir: &Path) -> Capture {
    let man = std::fs::read_to_string(dir.join("manifest.txt")).expect("manifest.txt");
    let mut f = Files { dir: dir.to_path_buf(), cache: HashMap::new(), missing: 0 };
    let mut c = Capture { uma_hi: 4, ranges: vec![], texs: vec![], descs: HashMap::new(), views: vec![], samplers: vec![], psos: HashMap::new(), dispatch: None, targets: (vec![], 0), draws: vec![], missing: 0, complete: vec![], rec_info: vec![] };
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
                c.psos.insert(it[1].parse().unwrap(), Pso { gfx: it[2] == "gfx", hashes: [hex(it[3]), hex(it[4])], files: tok(l, "files").unwrap().split(',').map(String::from).collect(), desc: unhex(tok(l, "desc").unwrap()).unwrap() });
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
            // LOCAL (agent Li): an ExecuteIndirect dispatch (`groups 0 0 0`): its first argument record
            "csargs" => {
                let n = hex(tok(l, "bytes").unwrap()) as usize;
                let Some(b) = f.try_get(tok(l, "file").unwrap(), n) else { continue };
                if let Some((_, g, _)) = c.dispatch.as_mut().filter(|d| d.1 == [0, 0, 0] && env("XCL_LREPLAY_GROUPS").is_none() && n >= 12) {
                    *g = std::array::from_fn(|k| u32::from_le_bytes(b[4 * k..4 * k + 4].try_into().unwrap()));
                    c.rec_info.push(format!("dispatch groups from its argument record: {g:?}"));
                }
            }
            _ => {}
        }
    }
    for d in c.draws.iter_mut() {
        d.recs = recs.get(&d.k).cloned().unwrap_or_default();
        d.rec = d.recs.first().copied();
    }
    c.missing = f.missing;
    c.complete = man.lines().filter(|l| l.starts_with("complete ")).map(String::from).collect();
    c
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

// =======================================================================================
// LOCAL (agent V): the session form of the replay. Everything that does not depend on the
// directives is made once (`Session::new`); `trial` replays with the current directives.
// A single replay is a session with one trial.
// =======================================================================================

fn device() -> Option<(ID3D12Device, String)> {
    unsafe {
        let factory: IDXGIFactory6 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).ok()?;
        let sel = xcl::gpu::adapter::Selector::from_env_or("auto");
        let (a, desc) = xcl::gpu::adapter::candidates(&factory, &sel).into_iter().next()?;
        let dev = xcl::gpu::agility::create_device(&a, D3D_FEATURE_LEVEL_12_0).ok()?;
        Some((dev, xcl::gpu::adapter::name(&desc)))
    }
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
            let e = t0.elapsed();
            if e.as_secs() > 30 {
                panic!("GPU did not finish in 30 s (device removed: {:?})", self.dev.GetDeviceRemovedReason());
            }
            // LOCAL (agent V): a 1 ms sleep is up to 15 ms on Windows; most lists finish in under 1 ms
            if e.as_millis() < 20 {
                std::thread::yield_now();
            } else {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
        self.keep.push(l.cast().unwrap());
    }
}

// ---------------------------------------------------------------------------------------
// LOCAL (agent L): replay with different values, without recompiling.
// `XCL_LREPLAY_SET="<directive>;<directive>;..."` (or one per line in the file `XCL_LREPLAY_SETFILE`):
//   tex<ID>=zero | const:<hex bytes, repeated> | file:<raw file in footprint layout, e.g. a replay's *_replay.bin>
//   view<INDEX>=null | fmt:<DXGI format number>
//   mem:<window>:<hex offset>=<hex bytes>        bytes of a memory window (tables, constants, argument copies)
//   io<K>=<hex>                                  user-data word K of every draw and of the dispatch
//   pso<ID>_vs|_ps|_cs=hlsl:<file>               that (edited) HLSL instead of the translation
// `XCL_LREPLAY_EXPOSE=<x>`: HDR previews multiplied by x (default: scaled so that the 99th percentile is white)
// Every run writes `<out>/summary.txt` (same lines as printed with the prefix SUMMARY).
// ---------------------------------------------------------------------------------------
static DIRS: Mutex<Option<Vec<(String, String)>>> = Mutex::new(None);

fn with_directives<T>(f: impl FnOnce(&[(String, String)]) -> T) -> T {
    let mut d = DIRS.lock().unwrap();
    if d.is_none() {
        let mut text = env("XCL_LREPLAY_SET").unwrap_or_default();
        if let Some(f) = env("XCL_LREPLAY_SETFILE") {
            text.push(';');
            text.push_str(&std::fs::read_to_string(f).unwrap_or_default().replace('\n', ";"));
        }
        *d = Some(text.split(';').filter_map(|d| d.trim().split_once('=').map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))).collect());
    }
    f(d.as_ref().unwrap())
}

fn directives() -> Vec<(String, String)> {
    with_directives(|d| d.to_vec())
}

fn directive(key: &str) -> Option<String> {
    with_directives(|d| d.iter().find(|(k, _)| k == key).map(|x| x.1.clone()))
}

/// the directives of one command of the persistent mode, for the next replay only
fn set_trial(line: &str) {
    let mut envs: Vec<(String, String)> = Vec::new();
    let mut set = String::new();
    for d in line.split(';') {
        let d = d.trim();
        match d.split_once('=') {
            Some((k, v)) if k.trim().starts_with("XCL_LREPLAY_") => envs.push((k.trim().to_string(), v.trim().to_string())),
            Some((k, v)) if k.trim() == "out" => envs.push(("XCL_LREPLAY_OUT".into(), v.trim().to_string())),
            Some(_) => {
                set.push_str(d);
                set.push(';');
            }
            None => {}
        }
    }
    envs.push(("XCL_LREPLAY_SET".into(), set));
    *TRIAL_ENV.lock().unwrap() = envs;
    *DIRS.lock().unwrap() = None;
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

static SUMMARY: Mutex<Vec<String>> = Mutex::new(Vec::new());

fn summary(line: String) {
    println!("SUMMARY {line}");
    SUMMARY.lock().unwrap().push(line);
}

fn fnv(parts: &[&[u8]]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for p in parts {
        for &b in *p {
            h ^= b as u64;
            h = h.wrapping_mul(0x100_0000_01b3);
        }
        h ^= 0xff;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

fn differ(a: &[u8], b: &[u8]) -> usize {
    let n = a.len().min(b.len());
    if a[..n] == b[..n] {
        0
    } else {
        a[..n].iter().zip(&b[..n]).filter(|(x, y)| x != y).count()
    }
}

// ---- RGBA16F statistics and previews without sorting: a histogram of the brightest channel's
// half-float bits (for non-negative halves the order of the bits is the order of the values)
struct Hdr {
    hist: Vec<u32>,
    texels: usize,
    nz_rgb: usize,
    nz_a: usize,
    nan: usize,
}

fn hdr_hist(bytes: &[u8], fp: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT) -> Hdr {
    let (w, h, p) = (fp.Footprint.Width as usize, fp.Footprint.Height as usize, fp.Footprint.RowPitch as usize);
    let mut r = Hdr { hist: vec![0u32; 65536], texels: w * h, nz_rgb: 0, nz_a: 0, nan: 0 };
    for y in 0..h {
        let o = fp.Offset as usize + y * p;
        let Some(row) = bytes.get(o..o + w * 8) else {
            r.hist[0] += w as u32;
            continue;
        };
        for px in row.chunks_exact(8) {
            let x = u64::from_le_bytes(px.try_into().unwrap());
            if x & 0xFFFF_FFFF_FFFF != 0 {
                r.nz_rgb += 1;
            }
            if x >> 48 != 0 {
                r.nz_a += 1;
            }
            // as `f16` + `f32::max`: exponent 31 is not a number (skipped unless all three are),
            // a negative value is below every other one and ends as 0
            let (mut key, mut any) = (0u32, false);
            for c in 0..3 {
                let b = (x >> (16 * c)) as u32 & 0xFFFF;
                if b & 0x7C00 == 0x7C00 {
                    continue;
                }
                any = true;
                if b & 0x8000 == 0 {
                    key = key.max(b);
                }
            }
            if any {
                r.hist[key as usize] += 1;
            } else {
                r.nan += 1;
            }
        }
    }
    r
}

impl Hdr {
    /// the value at position `x` (0..1) of the sorted finite values; `positive`: of those above 0 only
    fn q(&self, x: f64, positive: bool) -> f32 {
        let from = if positive { 1 } else { 0 };
        let n: u64 = self.hist[from..].iter().map(|c| *c as u64).sum();
        if n == 0 {
            return 0.0;
        }
        let idx = ((n - 1) as f64 * x) as u64;
        let mut acc = 0u64;
        for (b, c) in self.hist.iter().enumerate().skip(from) {
            acc += *c as u64;
            if acc > idx {
                return f16(b as u16);
            }
        }
        0.0
    }
    fn stats(&self) -> String {
        format!("texels {} rgb-nonzero {} alpha-nonzero {} nan/inf {} p50 {:e} p90 {:e} p99 {:e} max {:e}", self.texels, self.nz_rgb, self.nz_a, self.nan, self.q(0.5, false), self.q(0.9, false), self.q(0.99, false), self.q(1.0, false))
    }
    fn scale(&self) -> f32 {
        let p99 = self.q(0.99, true);
        match env("XCL_LREPLAY_EXPOSE").and_then(|v| v.parse::<f32>().ok()) {
            Some(x) if x > 0.0 => 1.0 / x,
            _ => {
                if p99 > 0.0 {
                    p99
                } else {
                    1.0
                }
            }
        }
    }
    fn text(&self) -> String {
        let pos: u64 = self.hist[1..].iter().map(|c| *c as u64).sum();
        format!("non-zero texels {pos} of {}; brightest channel p50 {} p99 {} max {} (preview scaled to p99)", self.texels, self.q(0.5, true), self.q(0.99, true), self.q(1.0, true))
    }
}

fn crc32(parts: &[&[u8]]) -> u32 {
    static T: std::sync::OnceLock<[u32; 256]> = std::sync::OnceLock::new();
    let t = T.get_or_init(|| {
        std::array::from_fn(|n| {
            let mut c = n as u32;
            for _ in 0..8 {
                c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
            }
            c
        })
    });
    let mut c = 0xFFFF_FFFFu32;
    for p in parts {
        for &b in *p {
            c = t[((c ^ b as u32) & 0xFF) as usize] ^ (c >> 8);
        }
    }
    !c
}

/// RGBA16F texels -> PNG (linear, every `k`-th texel, scaled as `Hdr::scale`), written here so
/// that a replay needs no conversion step afterwards
fn write_hdr_png(path: &Path, bytes: &[u8], fp: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT, hdr: &Hdr, k: usize) {
    let (w, h, p) = (fp.Footprint.Width as usize, fp.Footprint.Height as usize, fp.Footprint.RowPitch as usize);
    let scale = hdr.scale();
    let lut: Vec<u8> = (0..65536u32)
        .map(|b| {
            let v = f16(b as u16);
            if v.is_finite() {
                ((v / scale).clamp(0.0, 1.0) * 255.0) as u8
            } else {
                255
            }
        })
        .collect();
    let (pw, ph) = (w.div_ceil(k), h.div_ceil(k));
    let mut raw = Vec::with_capacity(ph * (1 + 3 * pw));
    for y in (0..h).step_by(k) {
        raw.push(0u8);
        for x in (0..w).step_by(k) {
            let o = fp.Offset as usize + y * p + x * 8;
            match bytes.get(o..o + 6) {
                Some(t) => raw.extend_from_slice(&[lut[u16::from_le_bytes([t[0], t[1]]) as usize], lut[u16::from_le_bytes([t[2], t[3]]) as usize], lut[u16::from_le_bytes([t[4], t[5]]) as usize]]),
                None => raw.extend_from_slice(&[0, 0, 0]),
            }
        }
    }
    let z = miniz_oxide::deflate::compress_to_vec_zlib(&raw, 1);
    let mut f = Vec::with_capacity(z.len() + 64);
    f.extend_from_slice(b"\x89PNG\r\n\x1a\n");
    let mut chunk = |t: &[u8; 4], d: &[u8]| {
        f.extend_from_slice(&(d.len() as u32).to_be_bytes());
        f.extend_from_slice(t);
        f.extend_from_slice(d);
        f.extend_from_slice(&crc32(&[&t[..], d]).to_be_bytes());
    };
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&(pw as u32).to_be_bytes());
    ihdr.extend_from_slice(&(ph as u32).to_be_bytes());
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]);
    chunk(b"IHDR", &ihdr);
    chunk(b"IDAT", &z);
    chunk(b"IEND", &[]);
    let _ = std::fs::write(path, &f);
}

// ---- shaders, kept by tag and by what they were made from
static SHADERS: Mutex<Option<HashMap<String, (String, recomp::pso::Compiled)>>> = Mutex::new(None);
/// profile and static samplers of a tag's translation (an edited HLSL keeps them)
static META: Mutex<Option<HashMap<String, (&'static str, Vec<recomp::pso::StaticSampler>)>>> = Mutex::new(None);

/// DXIL of the HLSL the replay compiles itself (edited sources, noted sources): by source text,
/// in `XCL_LREPLAY_CACHE` (default `<capture>/lreplay-dxil`), apart from the title's caches
/// `XCL_LREPLAY_DXC=<arguments>` (e.g. `-O1`) are passed to DXC for these sources only (a faster
/// compile of a big edited shader; the title's own translations are never compiled that way).
fn compile_replay(src: &str, profile: &str) -> Result<Vec<u8>, String> {
    let dir = env("XCL_LREPLAY_CACHE").map(PathBuf::from).unwrap_or_else(|| env("XCL_LREPLAY").map(PathBuf::from).unwrap_or_default().join("lreplay-dxil"));
    let args = env("XCL_LREPLAY_DXC").unwrap_or_default();
    let file = dir.join(format!("{:016x}.dxil", fnv(&[src.as_bytes(), profile.as_bytes(), args.as_bytes()])));
    if let Ok(b) = std::fs::read(&file) {
        if !b.is_empty() {
            return Ok(b);
        }
    }
    let extra: Vec<&str> = args.split_whitespace().collect();
    let t0 = Instant::now();
    let dxil = recomp::dxc::compile(src, "main", profile, &extra)?;
    println!("  DXC {profile} {args}: {} B of HLSL -> {} B of DXIL in {:.2} s", src.len(), dxil.len(), t0.elapsed().as_secs_f64());
    let _ = std::fs::create_dir_all(&dir);
    let tmp = dir.join(format!("{}.{}.tmp", file.file_name().unwrap().to_string_lossy(), std::process::id()));
    if std::fs::write(&tmp, &dxil).is_ok() {
        let _ = std::fs::rename(&tmp, &file);
    }
    Ok(dxil)
}

/// `tag` = "pso<ID>_vs|ps|cs". `XCL_LREPLAY_HLSL=1` writes the translated HLSL to
/// `<out>/<tag>.hlsl`; the directive `<tag>=hlsl:<file>` compiles that (edited) HLSL instead of
/// the translation; `XCL_LREPLAY_NOTE=1`: the shader from its HLSL with every descriptor-heap
/// index it reads noted in a bitmap in `res` (+0x1000); else the runtime's (cached) DXIL, which
/// XCL keys by the translated HLSL: a changed translator recompiles only the programs whose
/// HLSL changed.
#[allow(clippy::too_many_arguments)]
fn shader_tagged(tag: &str, out: &Path, blob: &[u8], stage: recomp::pso::Stage, sb: u32, split: bool, windows: u32) -> recomp::pso::Compiled {
    let over = directive(tag).and_then(|v| v.strip_prefix("hlsl:").map(String::from));
    let over_text = over.as_ref().map(|f| std::fs::read_to_string(f).unwrap_or_else(|e| panic!("{tag}=hlsl:{f}: {e}")));
    let (note, dump) = (env("XCL_LREPLAY_NOTE").is_some(), env("XCL_LREPLAY_HLSL").is_some());
    let key = format!("sb{sb} split{split} windows{windows:x} note{note} over{:016x} dxc{}", over_text.as_ref().map_or(0, |t| fnv(&[t.as_bytes()])), env("XCL_LREPLAY_DXC").unwrap_or_default());
    if !dump {
        if let Some((k, c)) = SHADERS.lock().unwrap().get_or_insert_with(HashMap::new).get(tag) {
            if *k == key {
                return c.clone();
            }
        }
    }
    let meta_key = format!("{tag} sb{sb}");
    let meta = META.lock().unwrap().get_or_insert_with(HashMap::new).get(&meta_key).cloned();
    let c = if dump || note || (over_text.is_some() && meta.is_none()) {
        let (hlsl, profile, samplers) = api::console_translation(blob, stage, sb, windows, split).unwrap();
        META.lock().unwrap().get_or_insert_with(HashMap::new).insert(meta_key, (profile, samplers.clone()));
        if dump {
            let _ = std::fs::write(out.join(format!("{tag}.hlsl")), &hlsl);
        }
        if let Some(h) = &over_text {
            let dxil = compile_replay(h, profile).unwrap_or_else(|e| panic!("{tag}: the given HLSL does not compile: {e}"));
            recomp::pso::Compiled { dxil, samplers }
        } else if note {
            let dxil = compile_replay(&noted(&hlsl), profile).unwrap_or_else(|e| panic!("noted HLSL does not compile: {e}"));
            recomp::pso::Compiled { dxil, samplers }
        } else {
            recomp::pso::console_to_dxil_in(blob, stage, sb, split, windows).unwrap()
        }
    } else if let (Some(h), Some((profile, samplers))) = (&over_text, meta) {
        let dxil = compile_replay(h, profile).unwrap_or_else(|e| panic!("{tag}: the given HLSL does not compile: {e}"));
        recomp::pso::Compiled { dxil, samplers }
    } else {
        recomp::pso::console_to_dxil_in(blob, stage, sb, split, windows).unwrap()
    };
    SHADERS.lock().unwrap().get_or_insert_with(HashMap::new).insert(tag.to_string(), (key, c.clone()));
    c
}

/// the summary line of a `<tag>=hlsl:<file>` directive
fn say_override(tag: &str) {
    if let Some(f) = directive(tag).and_then(|v| v.strip_prefix("hlsl:").map(String::from)) {
        let n = SHADERS.lock().unwrap().as_ref().and_then(|m| m.get(tag)).map_or(0, |x| x.1.dxil.len());
        summary(format!("override {tag} = hlsl:{f} ({n} B DXIL)"));
    }
}

/// the translated HLSL with every descriptor-heap index it reads noted (`XCL_LREPLAY_NOTE`)
fn noted(hlsl: &str) -> String {
    let mut h = String::new();
    let pat = "ResourceDescriptorHeap[NonUniformResourceIndex(";
    let mut rest = hlsl;
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
    h
}

const TILE: u64 = 65536;
const WIN: u64 = 1 << 32;
const ZERO: u64 = 4 << 20;

struct Win {
    r: ID3D12Resource,
    heap: Option<ID3D12Heap>,
    /// every tile a replay mode can map, sorted; tile i lies at heap offset i
    tiles: Vec<u64>,
}

struct GfxPso {
    sig: u64,
    rs: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    rtv: [DXGI_FORMAT; 8],
    dsv: DXGI_FORMAT,
}

type Res = HashMap<u32, (ID3D12Resource, D3D12_RESOURCE_STATES)>;

struct Session {
    dir: PathBuf,
    cap: Capture,
    name: String,
    g: Gpu,
    wins: Vec<Win>,
    /// the phases whose tiles are mapped now
    win_key: String,
    win_used: bool,
    /// the bytes of every range a mode places, and where range i starts in it
    range_up: ID3D12Resource,
    range_at: Vec<u64>,
    zero_up: ID3D12Resource,
    ids: Vec<u32>,
    rt_ids: Vec<u32>,
    ds_id: Option<u32>,
    res: Res,
    /// what each texture holds (a phase, "zero", "override:<directive>") and which a replay may have written
    content: HashMap<u32, String>,
    dirty: HashSet<u32>,
    failed: HashSet<u32>,
    /// upload buffers of the textures a replay can write, kept to restore them
    tex_up: HashMap<(u32, String), ID3D12Resource>,
    zero_tex: HashMap<(u32, String), bool>,
    dh: ID3D12DescriptorHeap,
    sh: ID3D12DescriptorHeap,
    rtvh: ID3D12DescriptorHeap,
    dsvh: ID3D12DescriptorHeap,
    inc: usize,
    max_idx: u32,
    split_cs: bool,
    split_gfx: bool,
    blobs: HashMap<String, std::sync::Arc<Vec<u8>>>,
    cs_psos: HashMap<u32, (u64, ID3D12RootSignature, ID3D12PipelineState)>,
    gfx_psos: HashMap<u32, GfxPso>,
    hdr_memo: HashMap<(u32, &'static str), Hdr>,
    written: HashSet<String>,
    trials: u32,
    /// a replay is between its first and its last GPU command (a failure there leaves the
    /// tracked states unknown)
    in_gpu: bool,
}

unsafe fn upload_bytes(dev: &ID3D12Device, bytes: &[u8], total: u64) -> ID3D12Resource {
    let up = buf(dev, total, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, false);
    let mut q: *mut std::ffi::c_void = std::ptr::null_mut();
    up.Map(0, None, Some(&mut q)).unwrap();
    std::ptr::copy_nonoverlapping(bytes.as_ptr(), q as *mut u8, bytes.len().min(total as usize));
    up.Unmap(0, None);
    up
}

impl Session {
    unsafe fn new(dir: PathBuf, cap: Capture, dev: &ID3D12Device, name: String) -> Session {
        let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
        let g = Gpu { dev: dev.clone(), queue, fence: dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap(), fv: 0, alloc: dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap(), keep: vec![] };
        // ---- windows: reserved buffers; one heap per window for the tiles of every placeable range
        let placeable = |r: &Range| r.phase == "csbefore" || r.phase == "before";
        let mut wins: Vec<Win> = Vec::new();
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
            let mut tiles: Vec<u64> = Vec::new();
            for rg in cap.ranges.iter().filter(|r| r.w == k && placeable(r)) {
                for t in rg.off / TILE..=(rg.off + rg.bytes.len() as u64).saturating_sub(1) / TILE {
                    tiles.push(t);
                }
            }
            tiles.sort();
            tiles.dedup();
            let mut heap: Option<ID3D12Heap> = None;
            if !tiles.is_empty() {
                let hd = D3D12_HEAP_DESC { SizeInBytes: tiles.len() as u64 * TILE, Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, Alignment: 0, Flags: D3D12_HEAP_FLAG_DENY_RT_DS_TEXTURES | D3D12_HEAP_FLAG_DENY_NON_RT_DS_TEXTURES };
                dev.CreateHeap(&hd, &mut heap).unwrap();
            }
            wins.push(Win { r: r.unwrap(), heap, tiles });
        }
        let mut range_at = vec![u64::MAX; cap.ranges.len()];
        let mut total = 0u64;
        for (i, rg) in cap.ranges.iter().enumerate().filter(|(_, r)| placeable(r)) {
            range_at[i] = total;
            total = (total + rg.bytes.len() as u64 + 255) & !255;
        }
        let range_up = buf(dev, total + 4096, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, false);
        let mut q: *mut std::ffi::c_void = std::ptr::null_mut();
        range_up.Map(0, None, Some(&mut q)).unwrap();
        for (i, rg) in cap.ranges.iter().enumerate().filter(|(_, r)| placeable(r)) {
            std::ptr::copy_nonoverlapping(rg.bytes.as_ptr(), (q as *mut u8).add(range_at[i] as usize), rg.bytes.len());
        }
        range_up.Unmap(0, None);
        let zero_up = buf(dev, ZERO, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, false);
        // ---- every resource a view or a target names
        let mut ids: Vec<u32> = cap.views.iter().filter_map(|v| v.res).collect();
        ids.extend(cap.targets.0.iter().copied());
        ids.sort();
        ids.dedup();
        let (rt_ids, ds_id) = (cap.targets.0[..cap.targets.1.min(cap.targets.0.len())].to_vec(), cap.targets.0.get(cap.targets.1).copied());
        // ---- descriptor heaps
        let max_idx = cap.views.iter().map(|v| v.idx).max().unwrap_or(0).max(8) + 2;
        let dh: ID3D12DescriptorHeap = dev.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, NumDescriptors: max_idx, Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, NodeMask: 0 }).unwrap();
        let inc = dev.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) as usize;
        for k in 0..4u32 {
            let mut h = dh.GetCPUDescriptorHandleForHeapStart();
            h.ptr += (recomp::MEM_HEAP_BASE + k) as usize * inc;
            dev.CreateUnorderedAccessView(&wins[k as usize].r, None::<&ID3D12Resource>, Some(&recomp::runner::raw_window_uav(WIN)), h);
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
        let rtvh: ID3D12DescriptorHeap = dev.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV, NumDescriptors: 8, ..Default::default() }).unwrap();
        let dsvh: ID3D12DescriptorHeap = dev.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_DSV, NumDescriptors: 1, ..Default::default() }).unwrap();
        Session {
            dir,
            name,
            split_cs: recomp::pso::wave32_only_for_device(dev),
            split_gfx: recomp::pso::graphics_wave32_for_device(dev),
            g,
            wins,
            win_key: String::new(),
            win_used: false,
            range_up,
            range_at,
            zero_up,
            ids,
            rt_ids,
            ds_id,
            res: HashMap::new(),
            content: HashMap::new(),
            dirty: HashSet::new(),
            failed: HashSet::new(),
            tex_up: HashMap::new(),
            zero_tex: HashMap::new(),
            dh,
            sh,
            rtvh,
            dsvh,
            inc,
            max_idx,
            blobs: HashMap::new(),
            cs_psos: HashMap::new(),
            gfx_psos: HashMap::new(),
            hdr_memo: HashMap::new(),
            written: HashSet::new(),
            trials: 0,
            in_gpu: false,
            cap,
        }
    }

    fn blob(&mut self, file: &str) -> std::sync::Arc<Vec<u8>> {
        let dir = &self.dir;
        self.blobs.entry(file.to_string()).or_insert_with(|| std::sync::Arc::new(std::fs::read(dir.join(file)).unwrap_or_else(|e| panic!("{file}: {e}")))).clone()
    }
}

/// the windows and the textures as the capture had them before the stage(s) of this mode:
/// the first time everything is uploaded; afterwards only what a replay may have written or
/// what a directive changes (one command list)
unsafe fn restore(s: &mut Session, do_cs: bool, do_tl: bool) {
    let dev = s.g.dev.clone();
    // ---- windows: the phase's ranges (the compute's first, the draws' over them: the draws' own
    // argument copies and tables do not overlap the compute's argument copy)
    let phases: &[&str] = if do_cs && do_tl { &["csbefore", "before"] } else if do_cs { &["csbefore"] } else { &["before"] };
    let placed: Vec<usize> = phases.iter().flat_map(|p| s.cap.ranges.iter().enumerate().filter(move |(_, r)| r.phase == *p).map(|(i, _)| i)).collect();
    let mut want: Vec<Vec<u64>> = vec![Vec::new(); 4];
    for &i in &placed {
        let rg = &s.cap.ranges[i];
        for t in rg.off / TILE..=(rg.off + rg.bytes.len() as u64).saturating_sub(1) / TILE {
            want[rg.w as usize].push(t);
        }
    }
    for w in want.iter_mut() {
        w.sort();
        w.dedup();
    }
    let key = phases.join("+");
    if s.win_key != key {
        for (k, win) in s.wins.iter().enumerate() {
            println!("  window {k}: {} tiles mapped", want[k].len());
            let Some(h) = &win.heap else { continue };
            let n = win.tiles.len();
            let coords: Vec<D3D12_TILED_RESOURCE_COORDINATE> = win.tiles.iter().map(|&t| D3D12_TILED_RESOURCE_COORDINATE { X: t as u32, Y: 0, Z: 0, Subresource: 0 }).collect();
            let sizes: Vec<D3D12_TILE_REGION_SIZE> = win.tiles.iter().map(|_| D3D12_TILE_REGION_SIZE { NumTiles: 1, UseBox: false.into(), Width: 1, Height: 1, Depth: 1 }).collect();
            // a tile no range of this mode touches is not mapped (reads 0, writes dropped), as in a single replay
            let flags: Vec<D3D12_TILE_RANGE_FLAGS> = win.tiles.iter().map(|t| if want[k].binary_search(t).is_ok() { D3D12_TILE_RANGE_FLAG_NONE } else { D3D12_TILE_RANGE_FLAG_NULL }).collect();
            let starts: Vec<u32> = (0..n as u32).collect();
            let counts: Vec<u32> = win.tiles.iter().map(|_| 1).collect();
            s.g.queue.UpdateTileMappings(&win.r, n as u32, Some(coords.as_ptr()), Some(sizes.as_ptr()), h, n as u32, Some(flags.as_ptr()), Some(starts.as_ptr()), Some(counts.as_ptr()), D3D12_TILE_MAPPING_FLAG_NONE);
        }
        s.win_key = key;
    }
    let l = s.g.list();
    let from = if s.win_used { D3D12_RESOURCE_STATE_UNORDERED_ACCESS } else { D3D12_RESOURCE_STATE_COMMON };
    for w in &s.wins {
        l.ResourceBarrier(&[transition(&w.r, from, D3D12_RESOURCE_STATE_COPY_DEST)]);
    }
    if s.win_used {
        // what an earlier replay left in the mapped tiles (a new heap is zero)
        for (k, tiles) in want.iter().enumerate() {
            let mut i = 0;
            while i < tiles.len() {
                let mut j = i + 1;
                while j < tiles.len() && tiles[j] == tiles[j - 1] + 1 {
                    j += 1;
                }
                let (mut at, end) = (tiles[i] * TILE, (tiles[j - 1] + 1) * TILE);
                while at < end {
                    let n = (end - at).min(ZERO);
                    l.CopyBufferRegion(&s.wins[k].r, at, &s.zero_up, 0, n);
                    at += n;
                }
                i = j;
            }
        }
    }
    let mut bytes = 0u64;
    for &i in &placed {
        let rg = &s.cap.ranges[i];
        l.CopyBufferRegion(&s.wins[rg.w as usize].r, rg.off, &s.range_up, s.range_at[i], rg.bytes.len() as u64);
        bytes += rg.bytes.len() as u64;
    }
    for w in &s.wins {
        l.ResourceBarrier(&[transition(&w.r, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)]);
    }
    s.win_used = true;
    println!("  {} ranges placed ({} MB)", placed.len(), bytes >> 20);
    // ---- textures: data of the first phase that has it, or a `tex<ID>=...` directive
    let tex_pref: &[&str] = if do_cs { &["csbefore", "before", "csafter", "after"] } else { &["before", "csafter", "csbefore", "after"] };
    let mut missing = Vec::new();
    let mut once: Vec<ID3D12Resource> = Vec::new();
    let (mut restored, mut restored_bytes) = (0, 0u64);
    for id in s.ids.clone() {
        let Some(d) = s.cap.descs.get(&id).copied() else {
            missing.push(id);
            continue;
        };
        if s.failed.contains(&id) {
            continue;
        }
        let ds = d.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL.0 != 0;
        let writable = d.Flags.0 & (D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL.0 | D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET.0 | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0) != 0;
        let state = if ds {
            if do_cs && !do_tl { D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE } else { D3D12_RESOURCE_STATE_DEPTH_READ | D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE }
        } else if s.rt_ids.contains(&id) && do_tl && !do_cs {
            D3D12_RESOURCE_STATE_RENDER_TARGET
        } else if d.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 != 0 {
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS
        } else {
            D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE
        };
        if !s.res.contains_key(&id) {
            let mut r: Option<ID3D12Resource> = None;
            if dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut r).is_err() {
                println!("  tex {id}: not created ({}x{} f{} flags {:#x})", d.Width, d.Height, d.Format.0, d.Flags.0);
                s.failed.insert(id);
                continue;
            }
            s.res.insert(id, (r.unwrap(), D3D12_RESOURCE_STATE_COPY_DEST));
            s.content.insert(id, "zero".into());
        }
        let over = directive(&format!("tex{id}"));
        let data = tex_pref.iter().find_map(|p| s.cap.texs.iter().find(|t| t.id == id && t.phase == *p));
        let tag = match (&over, data) {
            (Some(v), _) => format!("override:{v}"),
            (None, Some(t)) => t.phase.clone(),
            (None, None) => "zero".to_string(),
        };
        let (r, cur) = s.res[&id].clone();
        let need = s.content[&id] != tag || s.dirty.contains(&id) || over.as_deref().is_some_and(|v| v.starts_with("file:"));
        if need {
            // the footprints of what is uploaded: the captured copy's own description, as before
            let (fd, n) = match (&over, data) {
                (None, Some(t)) => (t.desc, t.nsub),
                _ => (d, nsub(&d)),
            };
            let mut fps = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); n as usize];
            let mut tot = 0u64;
            dev.GetCopyableFootprints(&fd, 0, n, 0, Some(fps.as_mut_ptr()), None, None, Some(&mut tot));
            let up = if over.is_some() {
                let up = upload_bytes(&dev, &tex_override(id, tot as usize).unwrap_or_default(), tot);
                once.push(up.clone());
                up
            } else if let Some(u) = s.tex_up.get(&(id, tag.clone())) {
                u.clone()
            } else {
                let up = match data {
                    Some(t) => upload_bytes(&dev, &t.bytes, tot),
                    None => buf(&dev, tot, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, false),
                };
                if writable {
                    s.tex_up.insert((id, tag.clone()), up.clone());
                } else {
                    once.push(up.clone());
                }
                up
            };
            if cur != D3D12_RESOURCE_STATE_COPY_DEST {
                l.ResourceBarrier(&[transition(&r, cur, D3D12_RESOURCE_STATE_COPY_DEST)]);
            }
            for (k, fp) in fps.iter().enumerate() {
                let src = D3D12_TEXTURE_COPY_LOCATION { pResource: transmute_copy(&up), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: *fp } };
                let dst = D3D12_TEXTURE_COPY_LOCATION { pResource: transmute_copy(&r), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: k as u32 } };
                l.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
            }
            l.ResourceBarrier(&[transition(&r, D3D12_RESOURCE_STATE_COPY_DEST, state)]);
            restored += 1;
            restored_bytes += tot;
        } else {
            if let Some(v) = &over {
                summary(format!("override tex{id} = {} (as in the previous replay)", &v[..v.len().min(60)]));
            }
            if data.is_none() && over.is_none() && s.trials <= 1 {
                println!("  tex {id}: no captured content ({}x{} f{}): zero", d.Width, d.Height, d.Format.0);
            }
            if cur != state {
                l.ResourceBarrier(&[transition(&r, cur, state)]);
            }
        }
        s.res.get_mut(&id).unwrap().1 = state;
        s.content.insert(id, tag);
        if writable {
            s.dirty.insert(id);
        } else {
            s.dirty.remove(&id);
        }
    }
    s.g.run(l);
    drop(once);
    println!("  {restored} of {} textures uploaded ({} MB)", s.res.len(), restored_bytes >> 20);
    if !missing.is_empty() && s.trials <= 1 {
        println!("  resources named by views but never described (not captured): {missing:?}");
    }
    // ---- `mem:` directives over the placed ranges
    for (k, v) in directives() {
        let Some(rest) = k.strip_prefix("mem:") else { continue };
        let Some((w, off)) = rest.split_once(':') else { continue };
        let (w, off): (usize, u64) = (w.parse().unwrap(), hex(off));
        let bytes = unhex(&v).unwrap_or_default();
        let mut words = vec![0u32; bytes.len().div_ceil(4)];
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), words.as_mut_ptr() as *mut u8, bytes.len());
        let b = upload_words(&mut s.g, &words);
        let l = s.g.list();
        l.ResourceBarrier(&[transition(&b, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE), transition(&s.wins[w].r, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST)]);
        l.CopyBufferRegion(&s.wins[w].r, off, &b, 0, bytes.len() as u64);
        l.ResourceBarrier(&[transition(&s.wins[w].r, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)]);
        s.g.run(l);
        s.g.keep.push(b.cast().unwrap());
        summary(format!("override mem window {w} offset {off:#x}: {} bytes", bytes.len()));
    }
}

/// every captured view at its heap index, with the `view<INDEX>=...` directives
unsafe fn views(s: &Session) {
    let dev = &s.g.dev;
    let h_at = |i: u32| {
        let mut h = s.dh.GetCPUDescriptorHandleForHeapStart();
        h.ptr += i as usize * s.inc;
        h
    };
    let null_srv = D3D12_SHADER_RESOURCE_VIEW_DESC { Format: DXGI_FORMAT_R32_FLOAT, ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D, Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, ..Default::default() };
    let null_uav = D3D12_UNORDERED_ACCESS_VIEW_DESC { Format: DXGI_FORMAT_R32_FLOAT, ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D, ..Default::default() };
    let dirs: HashMap<u32, String> = directives().into_iter().filter_map(|(k, v)| Some((k.strip_prefix("view")?.parse().ok()?, v))).collect();
    let (mut made, mut nulls) = (0, 0);
    for v in &s.cap.views {
        if v.idx < 4 {
            continue;
        }
        let vdir = dirs.get(&v.idx);
        let r = v.res.and_then(|id| s.res.get(&id)).map(|x| &x.0).filter(|_| vdir.map(String::as_str) != Some("null"));
        let mut vdesc = v.desc.clone();
        if let (Some(f), Some(b)) = (vdir.and_then(|x| x.strip_prefix("fmt:")).and_then(|x| x.parse::<u32>().ok()), vdesc.as_mut()) {
            b[..4].copy_from_slice(&f.to_le_bytes());
        }
        if let Some(x) = vdir {
            summary(format!("override view{} = {x}", v.idx));
        }
        if v.uav {
            let d: Option<D3D12_UNORDERED_ACCESS_VIEW_DESC> = vdesc.as_ref().map(|b| pod(b));
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
            let d: Option<D3D12_SHADER_RESOURCE_VIEW_DESC> = vdesc.as_ref().map(|b| pod(b));
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
    }
    println!("  view heap {} descriptors: {made} views made, {nulls} null", s.max_idx);
}

/// the compute pipeline of the dispatch (kept while its DXIL is the same)
unsafe fn cs_pipeline(s: &mut Session, out: &Path, windows: u32) -> Option<(ID3D12RootSignature, ID3D12PipelineState)> {
    let pid = s.cap.dispatch.as_ref()?.0;
    let file = s.cap.psos[&pid].files[0].clone();
    let blob = s.blob(&file);
    let c = shader_tagged(&format!("pso{pid}_cs"), out, &blob, recomp::pso::Stage::Compute, 0, s.split_cs, windows);
    say_override(&format!("pso{pid}_cs"));
    let sig = fnv(&[&c.dxil[..], format!("{:?}", c.samplers).as_bytes()]);
    if let Some(e) = s.cs_psos.get(&pid).filter(|e| e.0 == sig) {
        return Some((e.1.clone(), e.2.clone()));
    }
    let t_c = Instant::now();
    let dev = s.g.dev.clone();
    let rs = recomp::pso::abi_root_signature_with(&dev, false, &c.samplers).unwrap();
    let pso: ID3D12PipelineState = dev.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC { pRootSignature: transmute_copy(&rs), CS: D3D12_SHADER_BYTECODE { pShaderBytecode: c.dxil.as_ptr() as _, BytecodeLength: c.dxil.len() }, ..Default::default() }).unwrap();
    println!("  cs {:016x}: DXIL {} B, pipeline in {:.2} s", s.cap.psos[&pid].hashes[0], c.dxil.len(), t_c.elapsed().as_secs_f64());
    s.cs_psos.insert(pid, (sig, rs.clone(), pso.clone()));
    Some((rs, pso))
}

/// a COM pointer used from or handed over by another thread (D3D12 device objects are free-threaded)
struct Sent<T>(T);
unsafe impl<T> Send for Sent<T> {}
unsafe impl<T> Sync for Sent<T> {}
impl<T> Sent<T> {
    fn get(&self) -> &T {
        &self.0
    }
}

struct GfxJob {
    id: u32,
    vs: std::sync::Arc<Vec<u8>>,
    ps: Option<std::sync::Arc<Vec<u8>>>,
    /// the signature of the pipeline the session holds for this id
    have: Option<u64>,
}

/// the generated geometry shaders of rect-list pipelines, by capture id
static RECT_GS: Mutex<Option<HashMap<u32, Option<Vec<u8>>>>> = Mutex::new(None);

/// the pipelines of the draws that will draw (range, instances) and always the first draw's;
/// returns their capture ids. Shaders and pipelines nobody holds yet are made on several
/// threads; a pipeline is rebuilt only when its DXIL changed.
unsafe fn gfx_pipelines(s: &mut Session, out: &Path, windows: u32, range: Option<(u32, u32)>, force_inst: Option<u32>) -> Vec<u32> {
    let all = env("XCL_LREPLAY_ALL_PSOS").is_some();
    let mut want: Vec<u32> = Vec::new();
    for d in &s.cap.draws {
        if want.contains(&d.pso) || !s.cap.psos.contains_key(&d.pso) {
            continue;
        }
        let will = !range.is_some_and(|(a, b)| d.k < a || d.k > b) && d.recs.iter().any(|r| r[0] > 0 && force_inst.unwrap_or(r[1]) > 0);
        if !will && !want.is_empty() && !all {
            continue;
        }
        want.push(d.pso);
    }
    let t_p = Instant::now();
    let mut jobs: Vec<GfxJob> = Vec::new();
    for id in &want {
        let files = s.cap.psos[id].files.clone();
        jobs.push(GfxJob { id: *id, vs: s.blob(&files[0]), ps: files.get(1).map(|f| s.blob(f)), have: s.gfx_psos.get(id).map(|e| e.sig) });
    }
    let split = s.split_gfx;
    let no_gs = env("XCL_LREPLAY_NO_RECT_GS").is_some();
    let dev = Sent(s.g.dev.clone());
    let psos = &s.cap.psos;
    // Ok(None): the pipeline held is still the right one; Err: not built (said, the draws are skipped)
    let build = |j: &GfxJob| -> Result<Option<(GfxPso, String)>, String> {
        let dev = dev.get();
        let p = &psos[&j.id];
        let vc = shader_tagged(&format!("pso{}_vs", j.id), out, &j.vs, recomp::pso::Stage::Vertex, 0, split, windows);
        let pc = j.ps.as_ref().map(|ps| shader_tagged(&format!("pso{}_ps", j.id), out, ps, recomp::pso::Stage::Pixel, vc.samplers.len() as u32, split, windows));
        let mut samplers = vc.samplers.clone();
        if let Some(pc) = &pc {
            samplers.extend(pc.samplers.iter().cloned());
        }
        assert!(p.desc.len() >= std::mem::size_of::<D3D12_GRAPHICS_PIPELINE_STATE_DESC>());
        let mut desc: D3D12_GRAPHICS_PIPELINE_STATE_DESC = std::ptr::read_unaligned(p.desc.as_ptr() as *const D3D12_GRAPHICS_PIPELINE_STATE_DESC);
        // LOCAL (agent L): as `real::translated_graphics_pso`: a rect list pipeline (console topology type 19)
        // gets the generated GS that completes each rect (without it: the triangle of its three vertices)
        let console_topology = desc.PrimitiveTopologyType.0;
        let rect_gs = if console_topology == 19 && !no_gs {
            let held = RECT_GS.lock().unwrap().get_or_insert_with(HashMap::new).get(&j.id).cloned();
            match held {
                Some(g) => g,
                None => {
                    let g = recomp::pso::rect_list_gs(&j.vs).ok();
                    RECT_GS.lock().unwrap().get_or_insert_with(HashMap::new).insert(j.id, g.clone());
                    g
                }
            }
        } else {
            None
        };
        let sig = fnv(&[&vc.dxil[..], pc.as_ref().map_or(&[][..], |c| &c.dxil[..]), rect_gs.as_deref().unwrap_or(&[]), format!("{samplers:?}").as_bytes()]);
        if j.have == Some(sig) {
            return Ok(None);
        }
        let rs = recomp::pso::abi_root_signature_with(dev, true, &samplers)?;
        std::ptr::write(&mut desc.pRootSignature, transmute_copy(&rs));
        desc.VS = D3D12_SHADER_BYTECODE { pShaderBytecode: vc.dxil.as_ptr() as _, BytecodeLength: vc.dxil.len() };
        desc.PS = pc.as_ref().map_or(D3D12_SHADER_BYTECODE::default(), |c| D3D12_SHADER_BYTECODE { pShaderBytecode: c.dxil.as_ptr() as _, BytecodeLength: c.dxil.len() });
        desc.GS = rect_gs.as_ref().map_or(D3D12_SHADER_BYTECODE::default(), |g| D3D12_SHADER_BYTECODE { pShaderBytecode: g.as_ptr() as _, BytecodeLength: g.len() });
        desc.HS = D3D12_SHADER_BYTECODE::default();
        desc.DS = D3D12_SHADER_BYTECODE::default();
        std::ptr::write(&mut desc.StreamOutput, std::mem::zeroed());
        desc.InputLayout = D3D12_INPUT_LAYOUT_DESC::default();
        desc.CachedPSO = D3D12_CACHED_PIPELINE_STATE::default();
        desc.NodeMask = 0;
        api::pc_topology_type(&mut desc);
        let pso: ID3D12PipelineState = dev.CreateGraphicsPipelineState(&desc).map_err(|e| format!("not built: {e}"))?;
        let z = &desc.DepthStencilState;
        let info = format!(
            "  first pipeline: console topology type {console_topology}, rect list GS {}\n  first pipeline vs {:016x} ps {:016x}: topology type {} RTs {:?} DSV {} depth (enable {}, write {}, func {}) stencil {} blend0 {} cull {}",
            rect_gs.as_ref().map_or("none".to_string(), |g| format!("{} B", g.len())),
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
        Ok(Some((GfxPso { sig, rs, pso, rtv: desc.RTVFormats, dsv: desc.DSVFormat }, info)))
    };
    type Built = Result<Option<(GfxPso, String)>, String>;
    let results: Mutex<Vec<Option<Sent<Built>>>> = Mutex::new((0..jobs.len()).map(|_| None).collect());
    let failed: Mutex<Option<String>> = Mutex::new(None);
    let next = std::sync::atomic::AtomicUsize::new(0);
    let work = || loop {
        let k = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let Some(j) = jobs.get(k) else { break };
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| build(j))) {
            Ok(r) => results.lock().unwrap()[k] = Some(Sent(r)),
            Err(e) => {
                failed.lock().unwrap().get_or_insert(panic_text(e.as_ref()));
            }
        }
    };
    let threads = jobs.len().min(recomp::dxc::compile_threads()).min(16).max(1);
    if threads > 1 {
        std::thread::scope(|sc| {
            for _ in 0..threads {
                sc.spawn(work);
            }
        });
    } else {
        work();
    }
    if let Some(m) = failed.into_inner().unwrap() {
        panic!("{m}");
    }
    let mut results = results.into_inner().unwrap();
    let (mut active, mut built) = (Vec::new(), 0);
    for (k, j) in jobs.iter().enumerate() {
        say_override(&format!("pso{}_vs", j.id));
        say_override(&format!("pso{}_ps", j.id));
        match results[k].take().map(|r| r.0) {
            Some(Ok(None)) => active.push(j.id),
            Some(Ok(Some((p, info)))) => {
                if active.is_empty() {
                    println!("{info}");
                }
                s.gfx_psos.insert(j.id, p);
                active.push(j.id);
                built += 1;
            }
            Some(Err(e)) => {
                s.gfx_psos.remove(&j.id);
                println!("  pso {}: {e}", j.id);
            }
            None => {}
        }
    }
    println!("  {} graphics pipelines ({built} built) in {:.2} s on {threads} threads", active.len(), t_p.elapsed().as_secs_f64());
    active
}

fn out_dir(dir: &Path) -> PathBuf {
    env("XCL_LREPLAY_OUT").map(PathBuf::from).unwrap_or_else(|| dir.join("replay"))
}

/// one replay with the current directives
unsafe fn trial(s: &mut Session, tm: &mut Vec<(&'static str, f64)>) {
    let dev = s.g.dev.clone();
    let serve = env("XCL_LREPLAY_SERVE").is_some();
    let (want_bin, want_describe) = (flag("XCL_LREPLAY_BIN", !serve), flag("XCL_LREPLAY_DESCRIBE", !serve));
    let mode = env("XCL_LREPLAY_MODE").unwrap_or_else(|| "both".into());
    let out = out_dir(&s.dir);
    let _ = std::fs::create_dir_all(&out);
    s.trials += 1;
    let mut t = Instant::now();
    let mut lap = |tm: &mut Vec<(&'static str, f64)>, what: &'static str| {
        tm.push((what, t.elapsed().as_secs_f64()));
        t = Instant::now();
    };
    summary(format!("capture {} mode {mode} depth {} set [{}] draws {:?} instances {:?}", s.dir.display(), env("XCL_LREPLAY_DEPTH").unwrap_or_else(|| "capture".into()), directives().iter().map(|(k, v)| format!("{k}={}", &v[..v.len().min(40)])).collect::<Vec<_>>().join("; "), env("XCL_LREPLAY_DRAWS"), env("XCL_LREPLAY_INSTANCES")));
    // the inputs: every captured view, null / all-zero ones named
    let (mut nulls, mut empty, mut filled) = (Vec::new(), Vec::new(), 0);
    let pref: &[&str] = if mode == "cs" { &["csbefore", "before"] } else { &["before", "csafter", "csbefore"] };
    let tex_dirs: HashSet<String> = directives().into_iter().map(|x| x.0).filter(|k| k.starts_with("tex")).collect();
    for v in &s.cap.views {
        match v.res {
            None => nulls.push(v.idx),
            Some(id) => {
                let t = pref.iter().find_map(|p| s.cap.texs.iter().find(|t| t.id == id && t.phase == *p));
                let over = tex_dirs.contains(&format!("tex{id}"));
                match t {
                    Some(t) if !over && *s.zero_tex.entry((id, t.phase.clone())).or_insert_with(|| t.bytes.iter().all(|b| *b == 0)) => empty.push(format!("view{}=tex{id}({}x{} f{})", v.idx, t.desc.Width, t.desc.Height, t.desc.Format.0)),
                    Some(_) => filled += 1,
                    None => empty.push(format!("view{}=tex{id}(not captured)", v.idx)),
                }
            }
        }
    }
    summary(format!("inputs: {} views, {filled} with content, {} null {:?}, {} all-zero or missing: {}", s.cap.views.len(), nulls.len(), nulls, empty.len(), empty.join(" ")));
    let do_cs = mode == "cs" || mode == "both";
    let do_tl = mode == "tl" || mode == "both";
    let windows = 4 | recomp::HEAP_WINDOWS;
    let range = env("XCL_LREPLAY_DRAWS").and_then(|r| {
        let (a, b) = r.split_once("..")?;
        Some((a.parse::<u32>().ok()?, b.parse::<u32>().ok()?))
    });
    let force_inst: Option<u32> = env("XCL_LREPLAY_INSTANCES").and_then(|v| v.parse().ok());
    lap(tm, "inputs");
    // ---- shaders and pipelines first: a source that does not compile stops here, before any GPU work
    let cs = if do_cs { cs_pipeline(s, &out, windows) } else { None };
    let active = if do_tl { gfx_pipelines(s, &out, windows, range, force_inst) } else { Vec::new() };
    lap(tm, "pipelines");
    let keep0 = s.g.keep.len();
    s.in_gpu = true;
    restore(s, do_cs, do_tl);
    lap(tm, "restore");
    views(s);
    let res_buf = buf(&dev, 1 << 20, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, true);
    lap(tm, "views");
    // ---- the compute program
    if let (Some((rs, pso)), Some((pid, groups, io))) = (cs, s.cap.dispatch.clone()) {
        let mut iow = io.clone();
        iow.resize(recomp::KERNEL_SGPRS, 0);
        iow[recomp::IO_UMA_BASE_HI] = s.cap.uma_hi;
        for (k, v) in directives() {
            if let Some(i) = k.strip_prefix("io").and_then(|x| x.parse::<usize>().ok()) {
                iow[i] = hex(&v) as u32;
            }
        }
        alias_copies(s, &dev, "prealias"); // LOCAL (agent Li)
        let io_buf = upload_words(&mut s.g, &iow);
        let l = s.g.list();
        l.SetDescriptorHeaps(&[Some(s.dh.clone()), Some(s.sh.clone())]);
        l.SetComputeRootSignature(&rs);
        l.SetPipelineState(&pso);
        l.SetComputeRootUnorderedAccessView(0, io_buf.GetGPUVirtualAddress());
        l.SetComputeRootUnorderedAccessView(1, res_buf.GetGPUVirtualAddress());
        for k in 0..4u32 {
            l.SetComputeRootUnorderedAccessView(2 + k, s.wins[k as usize].r.GetGPUVirtualAddress());
        }
        l.Dispatch(groups[0], groups[1], groups[2]);
        l.ResourceBarrier(&[uav_barrier()]);
        let t0 = Instant::now();
        s.g.run(l);
        println!("  pso{pid} dispatched on {} in {:.3} s, groups {groups:?}", s.name, t0.elapsed().as_secs_f64());
        s.g.keep.push(io_buf.cast().unwrap());
        noted_indices(&mut s.g, &res_buf, &s.cap, "compute");
        lap(tm, "compute");
        // the UAV textures after: compared with the game's
        for t in s.cap.texs.iter().filter(|t| t.phase == "csafter") {
            let Some((r, st)) = s.res.get(&t.id).cloned() else { continue };
            let (bytes, fps) = read_tex(&mut s.g, &r, st);
            if want_bin {
                let _ = std::fs::write(out.join(format!("cs_tex{}_replay.bin", t.id)), &bytes);
            }
            if want_describe {
                describe(&format!("tex {} ({}x{} f{}) after the dispatch", t.id, t.desc.Width, t.desc.Height, t.desc.Format.0), &bytes, &t.bytes, &fps, t.desc.Format.0);
            }
            summary(format!("cs output tex{} ({}x{} f{}): {} of {} bytes differ from the game", t.id, t.desc.Width, t.desc.Height, t.desc.Format.0, differ(&bytes, &t.bytes), bytes.len().min(t.bytes.len())));
            if want_describe {
                if let Some(b) = s.cap.texs.iter().find(|x| x.id == t.id && x.phase == "csbefore") {
                    println!("    (game: csbefore == csafter: {})", b.bytes == t.bytes);
                }
            }
        }
        lap(tm, "compute outputs");
        alias_copies(s, &dev, "alias"); // LOCAL (agent Li)
    }
    // ---- the depth before the draws, then the draws
    if do_tl {
        if let Some(ds) = s.ds_id.and_then(|id| s.res.get(&id).map(|r| (id, r.0.clone(), r.1))) {
            let how = env("XCL_LREPLAY_DEPTH").unwrap_or_else(|| "capture".into());
            if how != "capture" {
                depth_from_twin(&mut s.g, &s.cap, &s.res, ds, &how, &s.dh);
            }
        }
        draws(s, &active, &res_buf, range, force_inst, &out, do_cs, want_bin, want_describe, tm);
    }
    s.g.keep.truncate(keep0);
    s.in_gpu = false;
}

/// LOCAL (agent Li): `alias<W>=<R>` (after the dispatch) / `prealias<W>=<R>` (before it): the texels of texture W
/// (a writer, R32_UINT) copied raw into texture R (its R9G9B9E5 reader) through a buffer, as XCL's format-alias
/// sync does (`format_alias::sync_pair`)
unsafe fn alias_copies(s: &mut Session, dev: &ID3D12Device, prefix: &str) {
    for (k, v) in directives() {
        let (Some(w), Ok(r)) = (k.strip_prefix(prefix).and_then(|x| x.parse::<u32>().ok()), v.trim().parse::<u32>()) else { continue };
        let (Some((wr, ws)), Some((rr, rs))) = (s.res.get(&w).cloned(), s.res.get(&r).cloned()) else {
            summary(format!("{prefix}{w}={r}: texture not in this replay"));
            continue;
        };
        let (wd, rd) = (wr.GetDesc(), rr.GetDesc());
        let (mut wf, mut rf) = (D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(), D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default());
        let (mut wt, mut rt) = (0u64, 0u64);
        dev.GetCopyableFootprints(&wd, 0, 1, 0, Some(&mut wf), None, None, Some(&mut wt));
        dev.GetCopyableFootprints(&rd, 0, 1, 0, Some(&mut rf), None, None, Some(&mut rt));
        if wf.Footprint.RowPitch != rf.Footprint.RowPitch || wf.Footprint.Height != rf.Footprint.Height {
            summary(format!("{prefix}{w}={r}: footprints differ ({:?} / {:?})", (wf.Footprint.RowPitch, wf.Footprint.Height), (rf.Footprint.RowPitch, rf.Footprint.Height)));
            continue;
        }
        let b = buf(&dev, wt.max(rt), D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST, false);
        let l = s.g.list();
        if ws != D3D12_RESOURCE_STATE_COPY_SOURCE {
            l.ResourceBarrier(&[transition(&wr, ws, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
        }
        let src = D3D12_TEXTURE_COPY_LOCATION { pResource: transmute_copy(&wr), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 } };
        let mid = D3D12_TEXTURE_COPY_LOCATION { pResource: transmute_copy(&b), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: wf } };
        l.CopyTextureRegion(&mid, 0, 0, 0, &src, None);
        l.ResourceBarrier(&[transition(&b, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
        if rs != D3D12_RESOURCE_STATE_COPY_DEST {
            l.ResourceBarrier(&[transition(&rr, rs, D3D12_RESOURCE_STATE_COPY_DEST)]);
        }
        let mid2 = D3D12_TEXTURE_COPY_LOCATION { pResource: transmute_copy(&b), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: rf } };
        let dst = D3D12_TEXTURE_COPY_LOCATION { pResource: transmute_copy(&rr), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 } };
        l.CopyTextureRegion(&dst, 0, 0, 0, &mid2, None);
        if rs != D3D12_RESOURCE_STATE_COPY_DEST {
            l.ResourceBarrier(&[transition(&rr, D3D12_RESOURCE_STATE_COPY_DEST, rs)]);
        }
        if ws != D3D12_RESOURCE_STATE_COPY_SOURCE {
            l.ResourceBarrier(&[transition(&wr, D3D12_RESOURCE_STATE_COPY_SOURCE, ws)]);
        }
        s.g.run(l);
        s.g.keep.push(b.cast().unwrap());
        s.dirty.insert(r);
        summary(format!("{prefix}{w}={r}: writer texels copied into the reader ({wt} bytes)"));
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn draws(s: &mut Session, active: &[u32], res_buf: &ID3D12Resource, range: Option<(u32, u32)>, force_inst: Option<u32>, out: &Path, after_cs: bool, want_bin: bool, want_describe: bool, tm: &mut Vec<(&'static str, f64)>) {
    let dev = s.g.dev.clone();
    let mut t = Instant::now();
    let mut lap = |tm: &mut Vec<(&'static str, f64)>, what: &'static str| {
        tm.push((what, t.elapsed().as_secs_f64()));
        t = Instant::now();
    };
    // targets: RTVs / DSV, the first draw's pipeline gives the formats
    let Some(first) = s.cap.draws.first().filter(|d| active.contains(&d.pso)).and_then(|d| s.gfx_psos.get(&d.pso)) else {
        println!("  no draw pipeline");
        return;
    };
    let rinc = dev.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV) as usize;
    let mut rtvs = Vec::new();
    for (k, id) in s.rt_ids.iter().enumerate() {
        let (r, _) = &s.res[id];
        let mut h = s.rtvh.GetCPUDescriptorHandleForHeapStart();
        h.ptr += k * rinc;
        let v = D3D12_RENDER_TARGET_VIEW_DESC { Format: first.rtv[k], ViewDimension: D3D12_RTV_DIMENSION_TEXTURE2D, ..Default::default() };
        dev.CreateRenderTargetView(r, Some(&v), h);
        rtvs.push(h);
    }
    let dsv = s.ds_id.map(|id| {
        let h = s.dsvh.GetCPUDescriptorHandleForHeapStart();
        let v = D3D12_DEPTH_STENCIL_VIEW_DESC { Format: first.dsv, ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2D, Flags: D3D12_DSV_FLAG_READ_ONLY_DEPTH | D3D12_DSV_FLAG_READ_ONLY_STENCIL, ..Default::default() };
        dev.CreateDepthStencilView(&s.res[&id].0, Some(&v), h);
        h
    });
    // io of every draw in one buffer
    let n = s.cap.draws.len();
    let io_dirs: Vec<(usize, u32)> = directives().into_iter().filter_map(|(k, v)| Some((k.strip_prefix("io")?.parse::<usize>().ok()?, hex(&v) as u32))).collect();
    let mut all = vec![0u32; n * recomp::KERNEL_SGPRS];
    for (k, d) in s.cap.draws.iter().enumerate() {
        let w = &mut all[k * recomp::KERNEL_SGPRS..(k + 1) * recomp::KERNEL_SGPRS];
        for (i, v) in d.io.iter().take(recomp::KERNEL_SGPRS).enumerate() {
            w[i] = *v;
        }
        w[recomp::IO_UMA_BASE_HI] = s.cap.uma_hi;
        for (i, v) in &io_dirs {
            w[*i] = *v;
        }
    }
    let io_buf = upload_words(&mut s.g, &all);
    let l = s.g.list();
    // the targets to RENDER_TARGET (from what the setup gave them)
    for id in s.rt_ids.clone() {
        let e = s.res.get_mut(&id).unwrap();
        if e.1 != D3D12_RESOURCE_STATE_RENDER_TARGET {
            l.ResourceBarrier(&[transition(&e.0, e.1, D3D12_RESOURCE_STATE_RENDER_TARGET)]);
            e.1 = D3D12_RESOURCE_STATE_RENDER_TARGET;
        }
    }
    l.SetDescriptorHeaps(&[Some(s.dh.clone()), Some(s.sh.clone())]);
    l.OMSetRenderTargets(rtvs.len() as u32, Some(rtvs.as_ptr()), false, dsv.as_ref().map(|h| h as *const _));
    let (mut drawn, mut inst_sum, mut skipped) = (0, 0u64, 0);
    let mut vp_text = String::new();
    for (k, d) in s.cap.draws.iter().enumerate() {
        if range.is_some_and(|(a, b)| d.k < a || d.k > b) {
            continue;
        }
        let will = d.recs.iter().any(|r| r[0] > 0 && force_inst.unwrap_or(r[1]) > 0);
        let Some(e) = s.gfx_psos.get(&d.pso).filter(|_| active.contains(&d.pso)) else {
            if will {
                skipped += 1;
            }
            continue;
        };
        if will {
            let p = &s.cap.psos[&d.pso];
            summary(format!("draw {} pso{} vs {:016x} ps {:016x} record {:x?}", d.k, d.pso, p.hashes[0], p.hashes[1], d.recs.first()));
        }
        let Some(rec) = d.rec else {
            skipped += 1;
            continue;
        };
        let vp = parse_vp(&d.state).unwrap_or((0.0, 0.0, 2560.0, 1440.0, 0.0, 1.0));
        if k == 0 {
            vp_text = format!("viewport {vp:?}, state '{}', record {rec:x?}", d.state);
        }
        l.SetGraphicsRootSignature(&e.rs);
        l.SetPipelineState(&e.pso);
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
            l.SetGraphicsRootUnorderedAccessView(2 + w, s.wins[w as usize].r.GetGPUVirtualAddress());
        }
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
    s.g.run(l);
    noted_indices(&mut s.g, res_buf, &s.cap, "draws");
    println!("  {drawn} draws with instances ({inst_sum} instances), {skipped} without a pipeline or record, of {n}; GPU {:.3} s (after the dispatch: {after_cs})", t0.elapsed().as_secs_f64());
    summary(format!("draws: {drawn} with instances ({inst_sum} instances) of {n} captured, {skipped} without a pipeline or record"));
    s.g.keep.push(io_buf.cast().unwrap());
    lap(tm, "draws");
    // outputs: each target vs the game's 'after'
    let k_png: usize = env("XCL_LREPLAY_PNG_SCALE").and_then(|v| v.parse().ok()).filter(|k| *k > 0).unwrap_or(2);
    let expose = env("XCL_LREPLAY_EXPOSE").unwrap_or_default();
    let ids: Vec<u32> = s.rt_ids.iter().chain(s.ds_id.iter()).copied().collect();
    for id in &ids {
        let (r, st) = s.res[id].clone();
        let (bytes, fps) = read_tex(&mut s.g, &r, st);
        lap(tm, "readback");
        let d = r.GetDesc();
        if want_bin {
            let _ = std::fs::write(out.join(format!("tl_tex{id}_replay.bin")), &bytes);
        }
        let game: &[u8] = s.cap.texs.iter().find(|t| t.id == *id && t.phase == "after").map_or(&[][..], |t| &t.bytes[..]);
        let before: &[u8] = s.cap.texs.iter().find(|t| t.id == *id && t.phase == "before").map_or(&[][..], |t| &t.bytes[..]);
        if want_describe {
            describe(&format!("target {id} ({}x{} f{})", d.Width, d.Height, d.Format.0), &bytes, game, &fps, d.Format.0);
        }
        if game.len() >= bytes.len() {
            summary(format!("target tex{id} ({}x{} f{}): {} of {} bytes differ from the game", d.Width, d.Height, d.Format.0, differ(&bytes, &game[..bytes.len()]), bytes.len()));
        } else {
            summary(format!("target tex{id} ({}x{} f{}): no game bytes after the draws in this capture", d.Width, d.Height, d.Format.0));
        }
        lap(tm, "compare");
        if d.Format.0 == 10 {
            let h = hdr_hist(&bytes, &fps[0]);
            summary(format!("HDR tex{id} replay: {}", h.stats()));
            write_hdr_png(&out.join(format!("tl_tex{id}_replay.png")), &bytes, &fps[0], &h, k_png);
            println!("    HDR replay: {} -> tl_tex{id}_replay.png", h.text());
            if flag("XCL_LREPLAY_BMP", false) {
                let (w, hh) = (fps[0].Footprint.Width as usize, fps[0].Footprint.Height as usize);
                write_hdr_bmp(&out.join(format!("tl_tex{id}_replay.bmp")), &texels(&bytes, &fps[0], 8), w, hh);
            }
            for (tag, what, b) in [("after", "game:  ", game), ("before", "before:", before)] {
                if b.len() < bytes.len() {
                    continue;
                }
                // the title's own targets do not change between replays: read once, written once per output folder
                let g = s.hdr_memo.entry((*id, tag)).or_insert_with(|| hdr_hist(b, &fps[0]));
                summary(format!("HDR tex{id} {what} {}", g.stats()));
                let file = out.join(format!("tl_tex{id}_game_{tag}.png"));
                if s.written.insert(format!("{} {k_png} {expose}", file.display())) || !file.exists() {
                    write_hdr_png(&file, b, &fps[0], g, k_png);
                    println!("    HDR game_{tag}: {} -> tl_tex{id}_game_{tag}.png", g.text());
                }
            }
            lap(tm, "statistics + previews");
        }
    }
}

fn panic_text(e: &(dyn std::any::Any + Send)) -> String {
    e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|m| m.to_string())).unwrap_or_else(|| "panic".into())
}

/// a replay with its summary; false when it failed (the message is in the summary)
fn run_trial(s: &mut Session, t_start: Instant) -> bool {
    SUMMARY.lock().unwrap().clear();
    *DIRS.lock().unwrap() = None;
    let mut tm: Vec<(&'static str, f64)> = Vec::new();
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe { trial(s, &mut tm) }));
    if let Err(e) = &r {
        summary(format!("ERROR: {}", panic_text(e.as_ref())));
    }
    // the same step several times (one readback per target): summed
    let mut steps: Vec<(&'static str, f64)> = Vec::new();
    for (k, v) in tm {
        match steps.iter_mut().find(|x| x.0 == k) {
            Some(x) => x.1 += v,
            None => steps.push((k, v)),
        }
    }
    summary(format!("timing: {}", steps.iter().map(|(k, v)| format!("{k} {:.0} ms", v * 1000.0)).collect::<Vec<_>>().join(", ")));
    summary(format!("total {:.2} s", t_start.elapsed().as_secs_f64()));
    let _ = std::fs::write(out_dir(&s.dir).join("summary.txt"), SUMMARY.lock().unwrap().join("\n") + "\n");
    r.is_ok()
}

fn process_runs(name: &str) -> bool {
    use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::System::Diagnostics::ToolHelp::*;
    unsafe {
        let snap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
        if snap == INVALID_HANDLE_VALUE {
            return false;
        }
        let mut e: PROCESSENTRY32W = std::mem::zeroed();
        e.dwSize = std::mem::size_of::<PROCESSENTRY32W>() as u32;
        let mut found = false;
        let mut ok = Process32FirstW(snap, &mut e);
        while ok != 0 {
            let len = e.szExeFile.iter().position(|c| *c == 0).unwrap_or(e.szExeFile.len());
            if String::from_utf16_lossy(&e.szExeFile[..len]).eq_ignore_ascii_case(name) {
                found = true;
                break;
            }
            ok = Process32NextW(snap, &mut e);
        }
        CloseHandle(snap);
        found
    }
}

fn stop_serving(ctl: Option<&Path>, why: &str, code: i32) -> ! {
    use std::io::Write;
    println!("LREPLAY stopped: {why}");
    let _ = std::io::stdout().flush();
    if let Some(dir) = ctl {
        let _ = std::fs::write(dir.join("stopped"), format!("{why}\n"));
        let _ = std::fs::remove_file(dir.join("ready"));
    }
    std::process::exit(code)
}

/// Ends the process as soon as a file named in `XCL_LREPLAY_YIELD` exists or a process
/// `XCL_LREPLAY_YIELD_PROC` (default `xcl.exe`, `none`: no check) runs: one GPU, the title first.
fn watchdog(ctl: Option<PathBuf>) {
    let yields: Vec<PathBuf> = std::env::var("XCL_LREPLAY_YIELD").map(|v| v.split(';').filter(|p| !p.trim().is_empty()).map(|p| PathBuf::from(p.trim())).collect()).unwrap_or_default();
    let yield_proc = std::env::var("XCL_LREPLAY_YIELD_PROC").unwrap_or_else(|_| "xcl.exe".into());
    println!("lreplay: yields to {yields:?} and to a process {yield_proc}");
    std::thread::spawn(move || {
        let mut k = 0u32;
        loop {
            if let Some(p) = yields.iter().find(|p| p.exists()) {
                stop_serving(ctl.as_deref(), &format!("yield: {} exists", p.display()), 5);
            }
            if k % 4 == 0 && yield_proc != "none" && process_runs(&yield_proc) {
                stop_serving(ctl.as_deref(), &format!("yield: a process {yield_proc} runs"), 5);
            }
            k += 1;
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
    });
}

/// the persistent mode: commands from stdin or from `<ctl>/<name>.cmd` files until `quit`, a
/// yield condition or the idle limit
fn serve(s: &mut Session, ctl: &str) -> ! {
    use std::io::Write;
    let idle_s: u64 = env("XCL_LREPLAY_IDLE_S").and_then(|v| v.parse().ok()).unwrap_or(900);
    let stdin_mode = ctl == "stdin";
    let dir = PathBuf::from(ctl);
    let ctl_dir = (!stdin_mode).then(|| dir.clone());
    let stop = |why: String| stop_serving(ctl_dir.as_deref(), &why, 0);
    let (tx, rx) = std::sync::mpsc::channel::<String>();
    if stdin_mode {
        std::thread::spawn(move || {
            for l in std::io::stdin().lines() {
                let Ok(l) = l else { break };
                if tx.send(l).is_err() {
                    break;
                }
            }
        });
    } else {
        let _ = std::fs::create_dir_all(&dir);
        // what an earlier server or client left (only our own files: the folder may hold others)
        for e in std::fs::read_dir(&dir).into_iter().flatten().flatten() {
            let f = e.file_name().to_string_lossy().into_owned();
            if f == "ready" || f == "stopped" || [".cmd", ".done", ".tmp", ".new"].iter().any(|x| f.ends_with(x)) {
                let _ = std::fs::remove_file(e.path());
            }
        }
        let _ = std::fs::write(dir.join("ready"), format!("{}\n", std::process::id()));
    }
    println!("LREPLAY ready ({ctl}); idle limit {idle_s} s");
    let _ = std::io::stdout().flush();
    let mut last = Instant::now();
    let mut n = 0u32;
    loop {
        if last.elapsed().as_secs() > idle_s {
            stop(format!("no command for {idle_s} s"));
        }
        let cmd: Option<(String, String)> = if stdin_mode {
            match rx.recv_timeout(std::time::Duration::from_millis(50)) {
                Ok(l) => {
                    n += 1;
                    Some((n.to_string(), l))
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => None,
                Err(_) => stop("stdin closed".into()),
            }
        } else {
            let mut names: Vec<String> = std::fs::read_dir(&dir).into_iter().flatten().flatten().filter_map(|e| e.file_name().to_str().and_then(|f| f.strip_suffix(".cmd")).map(String::from)).collect();
            names.sort();
            match names.first() {
                Some(name) => {
                    let l = std::fs::read_to_string(dir.join(format!("{name}.cmd"))).unwrap_or_default();
                    let _ = std::fs::remove_file(dir.join(format!("{name}.cmd")));
                    Some((name.clone(), l))
                }
                None => {
                    std::thread::sleep(std::time::Duration::from_millis(2));
                    None
                }
            }
        };
        let Some((name, line)) = cmd else { continue };
        let line = line.replace(['\r', '\n'], ";");
        if line.trim_matches(|c: char| c == ';' || c.is_whitespace()) == "quit" {
            stop("quit".into());
        }
        let t0 = Instant::now();
        set_trial(&line);
        let ok = run_trial(s, t0);
        if !stdin_mode {
            let text = SUMMARY.lock().unwrap().join("\n") + if ok { "\nOK\n" } else { "\nERROR\n" };
            let tmp = dir.join(format!("{name}.tmp"));
            let _ = std::fs::write(&tmp, text);
            let _ = std::fs::rename(&tmp, dir.join(format!("{name}.done")));
        }
        println!("LREPLAY done {name}: {} in {:.3} s", if ok { "OK" } else { "ERROR" }, t0.elapsed().as_secs_f64());
        let _ = std::io::stdout().flush();
        if !ok && s.in_gpu {
            stop("a replay failed during its GPU work (resource states unknown): start again".into());
        }
        last = Instant::now();
    }
}

fn main() {
    let Some(dir) = env("XCL_LREPLAY").map(PathBuf::from) else {
        eprintln!("lreplay: XCL_LREPLAY=<capture dir> is not set (see the head of crates/xcl/examples/lreplay.rs)");
        std::process::exit(2);
    };
    let t_start = Instant::now();
    watchdog(env("XCL_LREPLAY_SERVE").filter(|c| c != "stdin").map(PathBuf::from));
    let cap = load(&dir);
    println!("lreplay: {} | {} ranges, {} texture copies, {} views, {} samplers, {} pipelines, dispatch {}, {} draws; loaded in {:.2} s", dir.display(), cap.ranges.len(), cap.texs.len(), cap.views.len(), cap.samplers.len(), cap.psos.len(), cap.dispatch.is_some(), cap.draws.len(), t_start.elapsed().as_secs_f64());
    println!("lreplay: {} entries without bytes on disk (skipped); complete lines: {:?}; records: {}", cap.missing, cap.complete, cap.rec_info.iter().take(3).cloned().collect::<Vec<_>>().join(" | "));
    let t_dev = Instant::now();
    let Some((dev, name)) = device() else {
        println!("lreplay: no GPU");
        std::process::exit(1);
    };
    let mut s = unsafe { Session::new(dir, cap, &dev, name) };
    println!("lreplay: device and session in {:.2} s", t_dev.elapsed().as_secs_f64());
    let ok = run_trial(&mut s, t_start);
    println!("LREPLAY total {:.2} s", t_start.elapsed().as_secs_f64());
    match env("XCL_LREPLAY_SERVE") {
        Some(ctl) => serve(&mut s, &ctl),
        // no orderly teardown: the process ends here
        None => std::process::exit(if ok { 0 } else { 1 }),
    }
}
