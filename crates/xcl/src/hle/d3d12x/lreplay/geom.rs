//! LOCAL (agent R): offline replay of a geometry capture (`lcap.rs` with `XCL_LOCAL_LCAP_GEOM`): the
//! two-pass mesh draws that write the G-buffer and the main depth.
//!
//! Each captured draw is run as the title's list ran it: the compute pass (the console MS translated by
//! `recomp::pso::console_mesh_in`, the same text and caches as in the title) writes its records, then
//! the pass-through mesh pipeline (XCL's ABI root signature, the title's state sub-objects, the
//! translated console PS) draws them into the captured targets. Compared with the title's bytes: the
//! records after each compute pass, every target after the run.
//!
//! - `XCL_LREPLAY_DRAWS=a..b` only those draws
//! - `XCL_LREPLAY_REC=game`: the mesh pass draws the title's records (not the replayed compute pass's)
//! - `XCL_LREPLAY_NOTE=1`: instrumented shaders note every descriptor-heap index read and every 64 KiB
//!   page of the memory windows read (and the reads outside the windows), per draw and stage
//! - `XCL_LREPLAY_HLSL=1`: writes `pso<ID>_cs.hlsl` / `pso<ID>_ps.hlsl` / `pso<ID>_mesh.hlsl`
//! - directives (`XCL_LREPLAY_SET`): `pso<ID>_cs=hlsl:<file>`, `pso<ID>_ps=hlsl:<file>`,
//!   `recparam<slot>=x,y,z,w` (every vertex's output slot forced before the mesh pass; slot 0 = the first
//!   output after the position), `cio<K>=<hex>` / `mio<K>=<hex>` (a user-data word of the compute / mesh stage)

use super::*;

pub(super) struct GRec {
    pub record_bytes: usize,
    pub split: u32,
    pub groups: u32,
    pub bytes: Option<Vec<u8>>,
    /// the rewritten indirect arguments: (compute command, mesh command, offset of the DispatchMesh argument)
    pub args: Option<(Vec<u8>, Vec<u8>, usize)>,
}

pub(super) struct GDraw {
    pub k: u32,
    pub list: u64,
    pub pso: u32,
    pub targets: Vec<Option<u32>>,
    pub state: String,
    pub cio: Vec<u32>,
    pub mio: Option<Vec<u32>>,
    pub rec: Option<GRec>,
}

pub(super) fn parse_line(first: &str, l: &str, f: &mut Files, out: &mut Vec<GDraw>) {
    let it: Vec<&str> = l.split_whitespace().collect();
    let k: u32 = it.get(1).and_then(|v| v.parse().ok()).unwrap_or(u32::MAX);
    match first {
        "gdraw" => {
            let (head, io) = l.split_once(" io ").unwrap();
            out.push(GDraw {
                k,
                list: tok(l, "list").map_or(0, hex),
                pso: tok(l, "pso").unwrap().parse().unwrap(),
                targets: tok(l, "targets").unwrap().split(',').map(|v| v.parse().ok()).collect(),
                state: head.split_once(" state ").map(|x| x.1.to_string()).unwrap_or_default(),
                cio: words(io),
                mio: None,
                rec: None,
            });
        }
        "gmesh" => {
            if let Some(d) = out.iter_mut().find(|d| d.k == k) {
                d.mio = Some(words(l.split_once(" io ").unwrap().1));
            }
        }
        "grec" => {
            let Some(d) = out.iter_mut().find(|d| d.k == k) else { return };
            let num = |key: &str| tok(l, key).and_then(|v| v.parse::<usize>().ok()).unwrap_or(0);
            let bytes = tok(l, "file").and_then(|s| f.try_get(s, hex(tok(l, "bytes").unwrap()) as usize));
            let st = num("stride");
            let args = tok(l, "args").and_then(|s| f.try_get(s, 2 * st)).map(|b| (b[..st].to_vec(), b[st..].to_vec(), num("offset")));
            d.rec = Some(GRec { record_bytes: num("record_bytes"), split: num("split") as u32, groups: num("groups") as u32, bytes, args });
        }
        _ => {}
    }
}

const NB: u32 = 0x300_0000;
const RES_BYTES: u64 = 64 << 20;

pub(super) fn res_bytes() -> u64 {
    RES_BYTES
}

/// the translated HLSL with notes: every descriptor-heap index read (bitmap at `NB`), every 64 KiB page of
/// the memory windows read (bitmap at `NB + 0x20000`), reads outside the windows (count, last window, last
/// address at `NB + 0x28000`)
fn instrument(hlsl: &str) -> String {
    let mut h = String::new();
    let pat = "ResourceDescriptorHeap[NonUniformResourceIndex(";
    let mut rest = hlsl;
    while let Some(i) = rest.find(pat) {
        h += &rest[..i + pat.len()];
        rest = &rest[i + pat.len()..];
        let (mut depth, mut end) = (1, 0);
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
    let note = format!(
        "static const uint XCL_NB = {NB}u;\nuint xcl_note(uint i) {{ if (i >= 4u && i < 0x100000u) {{ res.InterlockedOr(XCL_NB + (i >> 5) * 4u, 1u << (i & 31u)); }} return i; }}\nvoid xcl_pg(uint a) {{ if (xcl_w >= 4u) {{ res.InterlockedAdd(XCL_NB + 0x28000u, 1u); res.Store(XCL_NB + 0x28004u, xcl_w); res.Store(XCL_NB + 0x28008u, a); return; }} uint p = (xcl_w << 16) | (a >> 16); res.InterlockedOr(XCL_NB + 0x20000u + (p >> 5) * 4u, 1u << (p & 31u)); }}\n"
    );
    let Some(at) = h.find("#define XCL_MW") else {
        panic!("the translated HLSL has no memory-window macro: notes not possible in this form");
    };
    h.insert_str(at, &note);
    for (a, b) in [
        ("uint xcl_mLoad(uint a) { ", "xcl_pg(a); "),
        ("uint2 xcl_mLoad2(uint a) { ", "xcl_pg(a); "),
        ("uint3 xcl_mLoad3(uint a) { ", "xcl_pg(a); "),
        ("uint4 xcl_mLoad4(uint a) { ", "xcl_pg(a); "),
        ("uint4 xcl_loadx4(uint4 a) { ", "xcl_pg(a.x); xcl_pg(a.y); xcl_pg(a.z); xcl_pg(a.w); "),
    ] {
        h = h.replacen(a, &format!("{a}{b}"), 1);
    }
    h
}

/// DXIL of a stage: the runtime's (cached) unless notes, an HLSL dump or an edited HLSL are asked
fn stage_dxil(tag: &str, out: &Path, translated: impl FnOnce() -> (String, &'static str), cached: impl FnOnce() -> Vec<u8>) -> Vec<u8> {
    let over = directive(tag).and_then(|v| v.strip_prefix("hlsl:").map(String::from));
    let note = env("XCL_LREPLAY_NOTE").is_some();
    if over.is_none() && !note && env("XCL_LREPLAY_HLSL").is_none() {
        return cached();
    }
    let (hlsl, profile) = translated();
    if env("XCL_LREPLAY_HLSL").is_some() {
        let _ = std::fs::write(out.join(format!("{tag}.hlsl")), &hlsl);
    }
    let src = match &over {
        Some(f) => std::fs::read_to_string(f).unwrap_or_else(|e| panic!("{tag}=hlsl:{f}: {e}")),
        None if note => hlsl,
        None => return cached(),
    };
    let src = if note { instrument(&src) } else { src };
    let dxil = recomp::dxc::compile(&src, "main", profile, &[]).unwrap_or_else(|e| panic!("{tag}: the HLSL does not compile: {e}"));
    if let Some(f) = over {
        summary(format!("override {tag} = hlsl:{f} ({} B DXIL)", dxil.len()));
    }
    dxil
}

struct MeshPso {
    crs: ID3D12RootSignature,
    cpso: ID3D12PipelineState,
    mrs: ID3D12RootSignature,
    mpso: ID3D12PipelineState,
    rec_dwords: usize,
    split: u32,
    maxv: usize,
    maxp: usize,
    vdw: usize,
    rt_formats: Vec<DXGI_FORMAT>,
    ds_format: DXGI_FORMAT,
}

fn sub(stream: &mut Vec<u8>, ty: u32, payload: &[u8], align8: bool) {
    let at = stream.len();
    stream.extend_from_slice(&ty.to_le_bytes());
    stream.resize(at + if align8 { 8 } else { 4 }, 0);
    stream.extend_from_slice(payload);
    let end = (stream.len() + 7) & !7;
    stream.resize(end, 0);
}

fn bytecode(code: &[u8]) -> Vec<u8> {
    let mut b = (code.as_ptr() as u64).to_le_bytes().to_vec();
    b.extend_from_slice(&(code.len() as u64).to_le_bytes());
    b
}

/// (type, payload) of the state sub-objects
fn subobjects(buf: &[u8]) -> Vec<(u32, &[u8])> {
    let mut out = Vec::new();
    let mut off = 0usize;
    while off + 8 <= buf.len() {
        let ty = u32::from_le_bytes(buf[off..off + 4].try_into().unwrap());
        let Some((psize, palign)) = super::super::null::subobject_payload(ty) else { break };
        let poff = if palign == 8 { 8 } else { 4 };
        let Some(p) = buf.get(off + poff..off + poff + psize) else { break };
        out.push((ty, p));
        off += (poff + psize + 7) & !7;
    }
    out
}

unsafe fn build(dev: &ID3D12Device, id: u32, p: &Pso, dir: &Path, out: &Path, windows: u32) -> Result<MeshPso, String> {
    let ms = std::fs::read(dir.join(&p.files[0])).map_err(|e| format!("{}: {e}", p.files[0]))?;
    let ps = p.files.get(1).map(|f| std::fs::read(dir.join(f)).unwrap());
    let wave32 = recomp::pso::wave32_only_for_device(dev);
    let gsplit = recomp::pso::graphics_wave32_for_device(dev);
    let tp = recomp::pso::console_mesh_in(&ms, ps.as_deref(), wave32, windows)?;
    if env("XCL_LREPLAY_HLSL").is_some() {
        let ps_inputs = ps.as_deref().map(|p| crate::gpu::dxbc::signatures(p).0);
        let _ = std::fs::write(out.join(format!("pso{id}_mesh.hlsl")), recomp::emit::mesh_passthrough_for(&tp.record, ps_inputs.as_deref()));
    }
    let cs = stage_dxil(
        &format!("pso{id}_cs"),
        out,
        || (recomp::pso::local_mesh_translation(&ms, wave32, windows).unwrap().0.hlsl, "cs_6_6"),
        || tp.cs.clone(),
    );
    let (psd, ps_samplers) = match &ps {
        Some(b) if recomp::pso::is_console(b) => {
            let c = recomp::pso::console_to_dxil_in(b, recomp::pso::Stage::Pixel, 0, gsplit, windows)?;
            let d = stage_dxil(
                &format!("pso{id}_ps"),
                out,
                || {
                    let (t, profile, _) = recomp::pso::console_translation(b, recomp::pso::Stage::Pixel, 0, windows, gsplit, recomp::CompactCs::Never).unwrap();
                    (t.hlsl, profile)
                },
                || c.dxil.clone(),
            );
            (Some(d), c.samplers)
        }
        Some(b) => (Some(b.clone()), Vec::new()),
        None => (None, Vec::new()),
    };
    let crs = recomp::pso::abi_root_signature_with(dev, true, &tp.samplers)?;
    let mrs = recomp::pso::abi_root_signature_with(dev, true, &ps_samplers)?;
    let cpso: ID3D12PipelineState = dev
        .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC { pRootSignature: transmute_copy(&crs), CS: D3D12_SHADER_BYTECODE { pShaderBytecode: cs.as_ptr() as _, BytecodeLength: cs.len() }, ..Default::default() })
        .map_err(|e| format!("compute pass: {e}"))?;
    let mut stream = Vec::new();
    sub(&mut stream, 0, &(mrs.as_raw() as u64).to_le_bytes(), true);
    sub(&mut stream, 25, &bytecode(&tp.ms), true);
    if let Some(d) = &psd {
        sub(&mut stream, 2, &bytecode(d), true);
    }
    stream.extend_from_slice(&p.desc);
    let dev2: ID3D12Device2 = dev.cast().map_err(|e| format!("ID3D12Device2: {e}"))?;
    let sd = D3D12_PIPELINE_STATE_STREAM_DESC { SizeInBytes: stream.len(), pPipelineStateSubobjectStream: stream.as_mut_ptr() as _ };
    let mpso: ID3D12PipelineState = dev2.CreatePipelineState(&sd).map_err(|e| format!("mesh pipeline: {e}"))?;
    let subs = subobjects(&p.desc);
    let word = |b: &[u8], k: usize| u32::from_le_bytes(b[4 * k..4 * k + 4].try_into().unwrap());
    let rt_formats = subs.iter().find(|s| s.0 == 15).map(|s| (0..word(s.1, 8).min(8) as usize).map(|k| DXGI_FORMAT(word(s.1, k) as i32)).collect()).unwrap_or_default();
    let ds_format = subs.iter().find(|s| s.0 == 16).map_or(DXGI_FORMAT_UNKNOWN, |s| DXGI_FORMAT(word(s.1, 0) as i32));
    Ok(MeshPso {
        crs,
        cpso,
        mrs,
        mpso,
        rec_dwords: tp.record.record_dwords(),
        split: tp.mesh_groups_per_record,
        maxv: tp.record.max_vertices as usize,
        maxp: tp.record.max_primitives as usize,
        vdw: tp.record.vert_dwords(),
        rt_formats,
        ds_format,
    })
}

unsafe fn read_res(g: &mut Gpu, res_buf: &ID3D12Resource, at: u64, n: u64) -> Vec<u8> {
    let rb = buf(&g.dev, n, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST, false);
    let l = g.list();
    l.ResourceBarrier(&[transition(res_buf, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
    l.CopyBufferRegion(&rb, 0, res_buf, at, n);
    l.ResourceBarrier(&[transition(res_buf, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)]);
    g.run(l);
    let mut q: *mut std::ffi::c_void = std::ptr::null_mut();
    rb.Map(0, None, Some(&mut q)).unwrap();
    let v = std::slice::from_raw_parts(q as *const u8, n as usize).to_vec();
    rb.Unmap(0, None);
    v
}

unsafe fn write_res(g: &mut Gpu, res_buf: &ID3D12Resource, at: u64, bytes: &[u8]) {
    if bytes.is_empty() {
        return;
    }
    let up = buf(&g.dev, bytes.len() as u64, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, false);
    let mut q: *mut std::ffi::c_void = std::ptr::null_mut();
    up.Map(0, None, Some(&mut q)).unwrap();
    std::ptr::copy_nonoverlapping(bytes.as_ptr(), q as *mut u8, bytes.len());
    up.Unmap(0, None);
    let l = g.list();
    l.ResourceBarrier(&[transition(res_buf, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST)]);
    l.CopyBufferRegion(res_buf, at, &up, 0, bytes.len() as u64);
    l.ResourceBarrier(&[transition(res_buf, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)]);
    g.run(l);
    g.keep.push(up.cast().unwrap());
}

/// The bytes of the records that mean something (the counts, the vertices below the vertex count, the
/// primitives below the primitive count: the rest of a record is what earlier draws left in the buffer):
/// (bytes compared, bytes different, different dwords in the counts / positions / each output slot / primitives)
fn rec_diff(a: &[u8], b: &[u8], p: &MeshPso, groups: usize) -> (usize, usize, String) {
    let rb = p.rec_dwords * 4;
    let slots = p.vdw / 4 - 1;
    let w = |x: &[u8], o: usize| u32::from_le_bytes(x[o..o + 4].try_into().unwrap());
    let (mut n, mut d) = (0usize, 0usize);
    let (mut dc, mut dp, mut dt) = (0usize, 0usize, 0usize);
    let mut ds = vec![0usize; slots];
    let mut cmp = |o: usize, len: usize, cnt: &mut usize| {
        n += len;
        for k in 0..len {
            if a[o + k] != b[o + k] {
                d += 1;
            }
        }
        for k in 0..len / 4 {
            if a[o + 4 * k..o + 4 * k + 4] != b[o + 4 * k..o + 4 * k + 4] {
                *cnt += 1;
            }
        }
    };
    for g in 0..groups {
        let r = g * rb;
        if r + rb > a.len().min(b.len()) {
            break;
        }
        cmp(r, 8, &mut dc);
        let nv = (w(a, r).max(w(b, r)) as usize).min(p.maxv);
        let np = (w(a, r + 4).max(w(b, r + 4)) as usize).min(p.maxp);
        for v in 0..nv {
            let o = r + 16 + v * p.vdw * 4;
            cmp(o, 16, &mut dp);
            for (s, c) in ds.iter_mut().enumerate() {
                cmp(o + 16 * (s + 1), 16, c);
            }
        }
        cmp(r + 16 + p.maxv * p.vdw * 4, 4 * np, &mut dt);
    }
    (n, d, format!("dwords different: counts {dc}, positions {dp}, slots {ds:?}, primitives {dt}"))
}

/// what the records hold: counts, positions, and the range of each output slot
fn rec_stats(b: &[u8], p: &MeshPso, groups: usize) -> String {
    let rb = p.rec_dwords * 4;
    let w = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
    let f = |o: usize| f32::from_bits(w(o));
    let (mut verts, mut prims, mut empty, mut nan, mut zero_pos, mut w_neg) = (0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
    let slots = p.vdw / 4 - 1;
    let mut lo = vec![[f32::INFINITY; 4]; slots];
    let mut hi = vec![[f32::NEG_INFINITY; 4]; slots];
    let mut zero_slot = vec![0usize; slots];
    let (mut plo, mut phi) = ([f32::INFINITY; 4], [f32::NEG_INFINITY; 4]);
    for gidx in 0..groups {
        let r = gidx * rb;
        if r + rb > b.len() {
            break;
        }
        let (nv, np) = ((w(r) as usize).min(p.maxv), (w(r + 4) as usize).min(p.maxp));
        if nv == 0 || np == 0 {
            empty += 1;
        }
        verts += nv;
        prims += np;
        for v in 0..nv {
            let o = r + 16 + v * p.vdw * 4;
            let pos = [f(o), f(o + 4), f(o + 8), f(o + 12)];
            if pos.iter().any(|x| !x.is_finite()) {
                nan += 1;
            } else {
                for c in 0..4 {
                    plo[c] = plo[c].min(pos[c]);
                    phi[c] = phi[c].max(pos[c]);
                }
            }
            if pos[0] == 0.0 && pos[1] == 0.0 && pos[2] == 0.0 {
                zero_pos += 1;
            }
            if pos[3] <= 0.0 {
                w_neg += 1;
            }
            for s in 0..slots {
                let so = o + 16 * (s + 1);
                if (0..4).all(|c| w(so + 4 * c) == 0) {
                    zero_slot[s] += 1;
                }
                for c in 0..4 {
                    let x = f(so + 4 * c);
                    if x.is_finite() {
                        lo[s][c] = lo[s][c].min(x);
                        hi[s][c] = hi[s][c].max(x);
                    }
                }
            }
        }
    }
    let mut t = format!("{groups} groups ({empty} without a vertex or a primitive), {verts} vertices, {prims} primitives; positions: {nan} not finite, {zero_pos} with x = y = z = 0, {w_neg} with w <= 0, range x {:.3e}..{:.3e} y {:.3e}..{:.3e} z {:.3e}..{:.3e} w {:.3e}..{:.3e}", plo[0], phi[0], plo[1], phi[1], plo[2], phi[2], plo[3], phi[3]);
    for s in 0..slots {
        t += &format!("; slot{s} all-zero {} range [{:.3e}..{:.3e}, {:.3e}..{:.3e}, {:.3e}..{:.3e}, {:.3e}..{:.3e}]", zero_slot[s], lo[s][0], hi[s][0], lo[s][1], hi[s][1], lo[s][2], hi[s][2], lo[s][3], hi[s][3]);
    }
    t
}

/// the notes of the last stage: indices and pages read; cleared afterwards
unsafe fn notes(g: &mut Gpu, res_buf: &ID3D12Resource, cap: &Capture, views: &HashMap<u32, (u64, u32, bool)>, what: &str, covered: &std::collections::HashSet<u32>) {
    let b = read_res(g, res_buf, NB as u64, 0x28010);
    let word = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
    let mut idx = Vec::new();
    for k in 0..0x8000usize {
        let x = word(4 * k);
        for i in 0..32 {
            if x >> i & 1 != 0 {
                idx.push((k * 32 + i) as u32);
            }
        }
    }
    let mut pages = Vec::new();
    for k in 0..0x2000usize {
        let x = word(0x20000 + 4 * k);
        for i in 0..32 {
            if x >> i & 1 != 0 {
                pages.push((k * 32 + i) as u32);
            }
        }
    }
    let oow = (word(0x28000), word(0x28004), word(0x28008));
    let missing: Vec<String> = pages.iter().filter(|p| !covered.contains(p)).map(|p| format!("{}:{:#x}", p >> 16, (p & 0xFFFF) << 16)).collect();
    summary(format!("note {what}: {} window pages read, {} not captured{}{}; reads outside the windows {} (last: window {} address {:#x})", pages.len(), missing.len(), if missing.is_empty() { "" } else { ": " }, missing.iter().take(24).cloned().collect::<Vec<_>>().join(" "), oow.0, oow.1, oow.2));
    let mut t = Vec::new();
    for i in &idx {
        let line = match cap.views.iter().find(|v| v.idx == *i) {
            Some(v) => match v.res {
                None => format!("{i}:null-view"),
                Some(id) => {
                    let d = cap.descs.get(&id);
                    let tx = cap.texs.iter().find(|t| t.id == id && t.phase == "before");
                    let content = match tx {
                        Some(t) if t.bytes.iter().all(|x| *x == 0) => "ALL-ZERO",
                        Some(_) => "content",
                        None => "not-captured",
                    };
                    format!("{i}:tex{id}({} f{} mips {} {}{content})", d.map_or(String::new(), |d| format!("{}x{}", d.Width, d.Height)), d.map_or(0, |d| d.Format.0), d.map_or(0, |d| d.MipLevels), if cap.evicted.get(&id).copied().unwrap_or(false) { "EVICTED " } else { "" })
                }
            },
            None => match views.get(i) {
                Some((res, fmt, ev)) => format!("{i}:uncaptured(res {res:#x} view f{fmt}{})", if *ev { " EVICTED" } else { "" }),
                None => format!("{i}:no-view-known"),
            },
        };
        t.push(line);
    }
    summary(format!("note {what}: {} heap indices read: {}", idx.len(), t.join(" ")));
    // cleared for the next stage
    write_res(g, res_buf, NB as u64, &vec![0u8; 0x28010]);
}

/// 8-bit RGBA / BGRA texels or a float depth plane as a BMP
fn write_bmp(path: &Path, w: usize, h: usize, px: impl Fn(usize, usize) -> [u8; 3]) {
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
            let c = px(x, y);
            bmp.extend_from_slice(&[c[2], c[1], c[0]]);
        }
        bmp.resize(start + row, 0);
    }
    let _ = std::fs::write(path, &bmp);
}

fn preview(path: &Path, bytes: &[u8], fp: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT, fmt: i32) {
    let (w, h, pitch, o) = (fp.Footprint.Width as usize, fp.Footprint.Height as usize, fp.Footprint.RowPitch as usize, fp.Offset as usize);
    if o + (h - 1) * pitch + 4 * w > bytes.len() {
        return;
    }
    match fmt {
        27..=31 => write_bmp(path, w, h, |x, y| {
            let p = o + y * pitch + 4 * x;
            [bytes[p], bytes[p + 1], bytes[p + 2]]
        }),
        87..=93 => write_bmp(path, w, h, |x, y| {
            let p = o + y * pitch + 4 * x;
            [bytes[p + 2], bytes[p + 1], bytes[p]]
        }),
        19 | 20 | 39..=41 => {
            // a float plane: grey by rank among the non-zero values
            let at = |x: usize, y: usize| f32::from_le_bytes(bytes[o + y * pitch + 4 * x..o + y * pitch + 4 * x + 4].try_into().unwrap());
            let mut v: Vec<f32> = (0..h).flat_map(|y| (0..w).map(move |x| (x, y))).map(|(x, y)| at(x, y)).filter(|d| d.is_finite() && *d != 0.0).collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let (lo, hi) = if v.is_empty() { (0.0, 1.0) } else { (v[v.len() / 100], v[v.len() - 1 - v.len() / 100]) };
            write_bmp(path, w, h, |x, y| {
                let d = at(x, y);
                if d == 0.0 || !d.is_finite() {
                    return [0, 0, 64];
                }
                let g = (48.0 + 207.0 * ((d - lo) / (hi - lo).max(1e-30)).clamp(0.0, 1.0)) as u8;
                [g, g, g]
            });
        }
        33..=38 => write_bmp(path, w, h, |x, y| {
            // two 16-bit channels: their high bytes
            let p = o + y * pitch + 4 * x;
            [bytes[p + 1], bytes[p + 3], 0]
        }),
        _ => {}
    }
}

/// per byte position of a 4-byte texel: texels where it is not zero
fn channel_counts(bytes: &[u8], fp: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT) -> [usize; 4] {
    let (w, h, pitch, o) = (fp.Footprint.Width as usize, fp.Footprint.Height as usize, fp.Footprint.RowPitch as usize, fp.Offset as usize);
    let mut n = [0usize; 4];
    if o + (h - 1) * pitch + 4 * w > bytes.len() {
        return n;
    }
    for y in 0..h {
        let r = &bytes[o + y * pitch..o + y * pitch + 4 * w];
        for t in r.chunks_exact(4) {
            for c in 0..4 {
                n[c] += (t[c] != 0) as usize;
            }
        }
    }
    n
}

#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn run(
    g: &mut Gpu,
    cap: &Capture,
    res: &HashMap<u32, (ID3D12Resource, D3D12_RESOURCE_STATES)>,
    wins: &[ID3D12Resource],
    dh: &ID3D12DescriptorHeap,
    sh: &ID3D12DescriptorHeap,
    res_buf: &ID3D12Resource,
    windows: u32,
    out: &Path,
) {
    let dev = g.dev.clone();
    let dir = dir_of(out);
    let note = env("XCL_LREPLAY_NOTE").is_some();
    let range = env("XCL_LREPLAY_DRAWS").and_then(|r| {
        let (a, b) = r.split_once("..")?;
        Some((a.parse::<u32>().ok()?, b.parse::<u32>().ok()?))
    });
    // LOCAL (agent G): `XCL_LREPLAY_SKIP=k,k,...` (or a file of numbers, `@<path>`): draws left out
    let skip: std::collections::HashSet<u32> = env("XCL_LREPLAY_SKIP")
        .map(|v| match v.strip_prefix('@') {
            Some(p) => std::fs::read_to_string(p).unwrap_or_default(),
            None => v,
        })
        .map(|v| v.split(|c: char| c == ',' || c.is_whitespace()).filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_default();
    if !skip.is_empty() {
        summary(format!("skip: {} draws left out", skip.len()));
    }
    let wanted = |d: &GDraw| !range.is_some_and(|(a, b)| d.k < a || d.k > b) && !skip.contains(&d.k);
    // every view's eviction state at the capture (views.txt: index, resource, view format, evicted)
    let mut views: HashMap<u32, (u64, u32, bool)> = HashMap::new();
    if let Ok(t) = std::fs::read_to_string(dir.join("views.txt")) {
        for l in t.lines() {
            let it: Vec<&str> = l.split_whitespace().collect();
            if it.len() >= 4 {
                views.insert(it[0].parse().unwrap_or(0), (hex(it[1]), it[2].trim_start_matches('f').parse().unwrap_or(0), it[3] == "1"));
            }
        }
    }
    // pages of the windows the capture holds
    let mut covered: std::collections::HashSet<u32> = std::collections::HashSet::new();
    for r in cap.ranges.iter().filter(|r| r.phase == "before") {
        for p in r.off >> 16..=(r.off + r.bytes.len() as u64).saturating_sub(1) >> 16 {
            covered.insert(r.w << 16 | p as u32);
        }
    }
    // the draws in the order their lists were submitted (then as recorded)
    let mut order: Vec<&GDraw> = cap.gdraws.iter().filter(|d| wanted(d)).collect();
    order.sort_by_key(|d| (cap.gexec.iter().position(|l| *l == d.list).unwrap_or(usize::MAX), d.k));
    if env("XCL_LREPLAY_ORDER").is_some_and(|v| v == "recorded") {
        order.sort_by_key(|d| d.k);
    }
    // only the draws with groups (`XCL_LREPLAY_ALL=1`: every draw, to build every pipeline)
    let w3 = |b: &[u8], o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
    let groups_of = |d: &GDraw| -> (u32, u32) {
        match d.rec.as_ref() {
            Some(GRec { args: Some((c, m, off)), .. }) => (w3(c, *off), w3(m, *off)),
            Some(r) => (r.groups, r.groups * r.split),
            None => (0, 0),
        }
    };
    let total = order.len();
    if env("XCL_LREPLAY_ALL").is_none() {
        order.retain(|d| groups_of(d).0 > 0);
    }
    summary(format!("geometry: {} draws captured on {} lists ({} lists seen submitted), {} wanted, {} of them with groups", cap.gdraws.len(), { let mut l: Vec<u64> = cap.gdraws.iter().map(|d| d.list).collect(); l.sort(); l.dedup(); l.len() }, cap.gexec.len(), total, order.len()));
    // pipelines
    let t_p = Instant::now();
    let mut psos: HashMap<u32, MeshPso> = HashMap::new();
    for d in order.iter().copied() {
        if psos.contains_key(&d.pso) {
            continue;
        }
        let Some(p) = cap.psos.get(&d.pso) else {
            summary(format!("draw {}: pipeline {} not in the capture", d.k, d.pso));
            continue;
        };
        match build(&dev, d.pso, p, &dir, out, windows) {
            Ok(m) => {
                summary(format!("pso{} ms {:016x} ps {:016x}: record {} dwords ({} vertices x {} dwords, {} primitives), split {}, targets {:?} depth f{}", d.pso, p.hashes[0], p.hashes[1], m.rec_dwords, m.maxv, m.vdw, m.maxp, m.split, m.rt_formats.iter().map(|f| f.0).collect::<Vec<_>>(), m.ds_format.0));
                psos.insert(d.pso, m);
            }
            Err(e) => summary(format!("pso{} ms {:016x} ps {:016x}: NOT BUILT: {e}", d.pso, p.hashes[0], p.hashes[1])),
        }
    }
    println!("  {} mesh pipelines in {:.1} s", psos.len(), t_p.elapsed().as_secs_f64());
    if psos.is_empty() {
        summary("no mesh pipeline built: nothing drawn".into());
    }
    // targets: views made per draw (its own set, its pipeline's formats)
    let rtvh: ID3D12DescriptorHeap = dev.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV, NumDescriptors: 8, ..Default::default() }).unwrap();
    let dsvh: ID3D12DescriptorHeap = dev.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_DSV, NumDescriptors: 1, ..Default::default() }).unwrap();
    let rinc = dev.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV) as usize;
    let oow_map = OOW_MAP.lock().unwrap().clone();
    let io_of = |w: &[u32], tag: &str, draw: u32| -> Vec<u32> {
        let mut v = w.to_vec();
        v.resize(recomp::KERNEL_SGPRS, 0);
        v[recomp::IO_UMA_BASE_HI] = cap.uma_hi;
        // `oow=map`: a root descriptor outside the windows points at the readable copy
        if let Some(m) = &oow_map {
            for (k, compute, at, ea) in &cap.oowio {
                if *k == draw && *compute == (tag == "cio") {
                    if let Some(n) = m.get(ea) {
                        v[*at] = *n as u32;
                        v[*at + 1] = (*n >> 32) as u32;
                    }
                }
            }
        }
        for (dk, x) in directives() {
            if let Some(i) = dk.strip_prefix(tag).and_then(|x| x.parse::<usize>().ok()) {
                v[i] = hex(&x) as u32;
            }
        }
        v
    };
    let recparams: Vec<(usize, [f32; 4])> = directives()
        .into_iter()
        .filter_map(|(k, v)| {
            let s = k.strip_prefix("recparam")?.parse::<usize>().ok()?;
            let f: Vec<f32> = v.split(',').filter_map(|x| x.trim().parse().ok()).collect();
            (f.len() == 4).then(|| (s, [f[0], f[1], f[2], f[3]]))
        })
        .collect();
    let use_game_rec = env("XCL_LREPLAY_REC").is_some_and(|v| v == "game");
    let (mut drawn, mut rec_diff_draws, mut rec_compared) = (0, 0, 0);
    let perdraw: Option<u32> = env("XCL_LREPLAY_PERDRAW").and_then(|v| v.parse().ok());
    let mut prev: Vec<u8> = match perdraw.and_then(|id| res.get(&id)) {
        Some((r, st)) => read_tex(g, r, *st).0,
        None => Vec::new(),
    };
    for d in order.iter().copied() {
        let Some(p) = psos.get(&d.pso) else { continue };
        let (Some(mio), Some(rec)) = (&d.mio, &d.rec) else {
            summary(format!("draw {}: its mesh stage or its records are not in the capture (cut): not drawn", d.k));
            continue;
        };
        let (nc, nm) = groups_of(d);
        // its targets
        let nrt = d.targets.len().saturating_sub(1).min(8);
        let mut rtvs = Vec::new();
        for (k, t) in d.targets[..nrt].iter().enumerate() {
            let mut h = rtvh.GetCPUDescriptorHandleForHeapStart();
            h.ptr += k * rinc;
            let r = t.and_then(|id| res.get(&id)).map(|x| &x.0);
            let fmt = p.rt_formats.get(k).copied().filter(|f| f.0 != 0).or(r.map(|r| r.GetDesc().Format)).unwrap_or(DXGI_FORMAT_R8G8B8A8_UNORM);
            let v = D3D12_RENDER_TARGET_VIEW_DESC { Format: fmt, ViewDimension: D3D12_RTV_DIMENSION_TEXTURE2D, ..Default::default() };
            match r {
                Some(r) => dev.CreateRenderTargetView(r, Some(&v), h),
                None => dev.CreateRenderTargetView(None::<&ID3D12Resource>, Some(&v), h),
            }
            rtvs.push(h);
        }
        let dsv = d.targets.last().copied().flatten().and_then(|id| res.get(&id)).map(|x| {
            let h = dsvh.GetCPUDescriptorHandleForHeapStart();
            let v = D3D12_DEPTH_STENCIL_VIEW_DESC { Format: p.ds_format, ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2D, ..Default::default() };
            dev.CreateDepthStencilView(&x.0, Some(&v), h);
            h
        });
        let rb = p.rec_dwords * 4;
        if rec.record_bytes != rb || rec.split != p.split {
            summary(format!("draw {}: the capture says records of {} B split {}, the replay's translation {} B split {}", d.k, rec.record_bytes, rec.split, rb, p.split));
        }
        // ---- compute pass
        // The record buffer is never cleared in the title: what a program does not write (a component of an
        // output it leaves alone, the vertices past its count) is what earlier draws left there. So the
        // title's bytes after the pass are put there first (`XCL_LREPLAY_PRELOAD=0`: not), and the comparison
        // shows what the replayed pass wrote differently.
        // directive `resfill=<hex bytes>`: the records the pass will write are first filled with that pattern
        // (what a program leaves unwritten then reads as it, not as an earlier draw's leftovers)
        if let Some(pat) = directive("resfill").and_then(|v| unhex(&v)).filter(|p| !p.is_empty()) {
            let n = (groups_of(d).0 as usize * p.rec_dwords * 4).min(NB as usize);
            write_res(g, res_buf, 0, &(0..n).map(|k| pat[k % pat.len()]).collect::<Vec<u8>>());
        }
        if let (Some(gb), true) = (&rec.bytes, env("XCL_LREPLAY_PRELOAD").is_none_or(|v| v != "0")) {
            write_res(g, res_buf, 0, gb);
        }
        let cio = upload_words(g, &io_of(&d.cio, "cio", d.k));
        let l = g.list();
        l.SetDescriptorHeaps(&[Some(dh.clone()), Some(sh.clone())]);
        l.SetComputeRootSignature(&p.crs);
        l.SetPipelineState(&p.cpso);
        l.SetComputeRootUnorderedAccessView(0, cio.GetGPUVirtualAddress());
        l.SetComputeRootUnorderedAccessView(1, res_buf.GetGPUVirtualAddress());
        for k in 0..4u32 {
            l.SetComputeRootUnorderedAccessView(2 + k, wins[k as usize].GetGPUVirtualAddress());
        }
        if nc > 0 {
            l.Dispatch(nc, 1, 1);
        }
        l.ResourceBarrier(&[uav_barrier()]);
        g.run(l);
        g.keep.push(cio.cast().unwrap());
        let valid = (nc as usize * rb).min(NB as usize);
        let mut mine = read_res(g, res_buf, 0, valid.max(4) as u64);
        mine.truncate(valid);
        let _ = std::fs::write(out.join(format!("g_rec{}_replay.bin", d.k)), &mine);
        let pp = &cap.psos[&d.pso];
        let mut line = format!("draw {} pso{} ms {:016x} ps {:016x}: {nc} compute groups, {nm} mesh groups", d.k, d.pso, pp.hashes[0], pp.hashes[1]);
        match &rec.bytes {
            Some(gb) => {
                let n = valid.min(gb.len());
                let (cn, diff, what) = rec_diff(&mine[..n], &gb[..n], p, n / rb);
                let _ = std::fs::write(out.join(format!("g_rec{}_game.bin", d.k)), &gb[..n]);
                line += &format!("; records: {diff} of {cn} meaningful bytes differ from the game ({} of {} records compared){}", n / rb, nc, if diff != 0 { format!("; {what}") } else { String::new() });
                rec_compared += 1;
                rec_diff_draws += (diff != 0) as u32;
                summary(line);
                if env("XCL_LREPLAY_RECSTATS").is_some() || diff != 0 {
                    summary(format!("draw {} records game:   {}", d.k, rec_stats(&gb[..n], p, n / rb)));
                }
                if diff != 0 {
                    summary(format!("draw {} records replay: {}", d.k, rec_stats(&mine[..n], p, n / rb)));
                }
                if use_game_rec {
                    write_res(g, res_buf, 0, &gb[..n]);
                    mine = gb[..n].to_vec();
                }
            }
            None => {
                line += "; the game's records are not in the capture";
                summary(line);
                summary(format!("draw {} records replay: {}", d.k, rec_stats(&mine, p, nc as usize)));
            }
        }
        if note {
            notes(g, res_buf, cap, &views, &format!("draw {} compute pass", d.k), &covered);
        }
        if !recparams.is_empty() {
            for gi in 0..mine.len() / rb {
                let r = gi * rb;
                let nv = (w3(&mine, r) as usize).min(p.maxv);
                for v in 0..nv {
                    for (s, f) in &recparams {
                        let o = r + 16 + v * p.vdw * 4 + 16 * (s + 1);
                        if o + 16 <= mine.len() && 4 * (s + 1) < p.vdw {
                            for c in 0..4 {
                                mine[o + 4 * c..o + 4 * c + 4].copy_from_slice(&f[c].to_le_bytes());
                            }
                        }
                    }
                }
            }
            write_res(g, res_buf, 0, &mine);
        }
        // ---- mesh pass
        let miob = upload_words(g, &io_of(mio, "mio", d.k));
        let l = g.list();
        l.SetDescriptorHeaps(&[Some(dh.clone()), Some(sh.clone())]);
        l.OMSetRenderTargets(rtvs.len() as u32, Some(rtvs.as_ptr()), false, dsv.as_ref().map(|h| h as *const _));
        let vp = parse_vp(&d.state).unwrap_or((0.0, 0.0, 2560.0, 1440.0, 0.0, 1.0));
        l.RSSetViewports(&[D3D12_VIEWPORT { TopLeftX: vp.0, TopLeftY: vp.1, Width: vp.2, Height: vp.3, MinDepth: vp.4, MaxDepth: vp.5 }]);
        let sc = parse_sc(&d.state).unwrap_or([0, 0, 16384, 16384]);
        l.RSSetScissorRects(&[windows::Win32::Foundation::RECT { left: sc[0], top: sc[1], right: sc[2], bottom: sc[3] }]);
        if let Some(r) = parse_some_u32(&d.state, " sref ") {
            l.OMSetStencilRef(r);
        }
        if let Some(b) = parse_bf(&d.state) {
            l.OMSetBlendFactor(Some(&b));
        }
        l.SetGraphicsRootSignature(&p.mrs);
        l.SetPipelineState(&p.mpso);
        l.SetGraphicsRootUnorderedAccessView(0, miob.GetGPUVirtualAddress());
        l.SetGraphicsRootUnorderedAccessView(1, res_buf.GetGPUVirtualAddress());
        for k in 0..4u32 {
            l.SetGraphicsRootUnorderedAccessView(2 + k, wins[k as usize].GetGPUVirtualAddress());
        }
        if nm > 0 {
            let l6: ID3D12GraphicsCommandList6 = l.cast().unwrap();
            l6.DispatchMesh(nm, 1, 1);
        }
        l.ResourceBarrier(&[uav_barrier()]);
        g.run(l);
        g.keep.push(miob.cast().unwrap());
        drawn += 1;
        // `XCL_LREPLAY_PERDRAW=<tex id>`: what each draw changed in that target (4-byte words) and the commonest new values
        if let Some((r, st)) = perdraw.and_then(|id| res.get(&id)) {
            let (now, _) = read_tex(g, r, *st);
            if prev.len() == now.len() {
                let mut vals: HashMap<u32, usize> = HashMap::new();
                let mut n = 0usize;
                for (a, b) in now.chunks_exact(4).zip(prev.chunks_exact(4)) {
                    if a != b {
                        n += 1;
                        *vals.entry(u32::from_le_bytes(a.try_into().unwrap())).or_default() += 1;
                    }
                }
                if n > 0 {
                    let mut top: Vec<(u32, usize)> = vals.into_iter().collect();
                    top.sort_by_key(|x| std::cmp::Reverse(x.1));
                    summary(format!("perdraw {} pso{} changed {n} words of tex{}; new values {}", d.k, d.pso, perdraw.unwrap(), top.iter().take(4).map(|(v, c)| format!("{v:08x} x{c}")).collect::<Vec<_>>().join(" ")));
                }
            }
            prev = now;
        }
        if note {
            notes(g, res_buf, cap, &views, &format!("draw {} pixel shader", d.k), &covered);
        }
    }
    summary(format!("geometry: {drawn} draws replayed of {} captured; records compared for {rec_compared}, different for {rec_diff_draws}{}", cap.gdraws.len(), if use_game_rec { " (mesh passes drew the GAME's records)" } else { "" }));
    // ---- targets after the run
    for id in cap.gtargets.iter().map(|t| &t.0) {
        let Some((r, st)) = res.get(id) else { continue };
        let (bytes, fps) = read_tex(g, r, *st);
        let d = r.GetDesc();
        let _ = std::fs::write(out.join(format!("g_tex{id}_replay.bin")), &bytes);
        let game = cap.texs.iter().find(|t| t.id == *id && t.phase == "after").map(|t| t.bytes.clone()).unwrap_or_default();
        let before = cap.texs.iter().find(|t| t.id == *id && t.phase == "before").map(|t| t.bytes.clone()).unwrap_or_default();
        let name = format!("target tex{id} ({}x{} f{})", d.Width, d.Height, d.Format.0);
        if game.len() >= bytes.len() {
            let diff = bytes.iter().zip(&game).filter(|(a, b)| a != b).count();
            let changed = if before.len() >= bytes.len() { game.iter().zip(&before).filter(|(a, b)| a != b).count() } else { 0 };
            summary(format!("{name}: {diff} of {} bytes differ from the game (the game's draws changed {changed} bytes)", bytes.len()));
            if before.len() >= bytes.len() {
                // by 4-byte word: what the replay changed and whether the game has the same there; what only the game changed
                let (mut rep_ch, mut rep_ok, mut game_only, mut game_ch) = (0usize, 0usize, 0usize, 0usize);
                for ((a, b), c) in bytes.chunks_exact(4).zip(game.chunks_exact(4)).zip(before.chunks_exact(4)) {
                    let (rc, gc) = (a != c, b != c);
                    rep_ch += rc as usize;
                    rep_ok += (rc && a == b) as usize;
                    game_ch += gc as usize;
                    game_only += (gc && !rc) as usize;
                }
                summary(format!("{name}: words the replay changed {rep_ch}, equal to the game there {rep_ok} ({:.2} %); words the game changed {game_ch}, of them not changed by the replay {game_only}", 100.0 * rep_ok as f64 / rep_ch.max(1) as f64));
            }
        } else {
            summary(format!("{name}: no game bytes after the draws in this capture"));
        }
        if fps[0].Footprint.RowPitch as usize >= 4 * d.Width as usize && !matches!(d.Format.0, 9..=14) {
            let mut t = format!("{name} non-zero texels per byte of the texel: replay {:?}", channel_counts(&bytes, &fps[0]));
            if game.len() >= bytes.len() {
                t += &format!(" game {:?}", channel_counts(&game, &fps[0]));
            }
            if before.len() >= bytes.len() {
                t += &format!(" before {:?}", channel_counts(&before, &fps[0]));
            }
            summary(t);
        }
        preview(&out.join(format!("g_tex{id}_replay.bmp")), &bytes, &fps[0], d.Format.0);
        if game.len() >= bytes.len() {
            preview(&out.join(format!("g_tex{id}_game_after.bmp")), &game, &fps[0], d.Format.0);
        }
        if before.len() >= bytes.len() {
            preview(&out.join(format!("g_tex{id}_game_before.bmp")), &before, &fps[0], d.Format.0);
        }
    }
}
