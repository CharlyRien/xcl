//! LOCAL diagnostic, not for upstream: what a title feeds chosen compute programs.
//!
//! `XCL_LOCAL_DUMP_CS_INPUTS=<cs>,…` (`cs` = `fnv64` of the CS container, as the `GPU culprit`
//! lines print it). At each Dispatch of such a program (at most `CAP` per program and run), at
//! record time, one `local inputs:` line:
//! - the user SGPRs s4 / s5 as the translated program gets them. XCL's ABI (`bind::bind_abi`,
//!   `console_list::user_data_for`): io s[0:1] = the address of the argument table (the root
//!   arguments past the inline part), from s4 on the inline part of the root arguments the title
//!   set (SetComputeRoot32BitConstant(s) / Root*View / RootDescriptorTable, packed in root
//!   parameter order); the line says which root parameter each comes from;
//! - the descriptor at s4 + 0xCA0 (the program loads it from s[4:5] = (s4, 0): window 0), its
//!   decoded buffer (address, stride, records, limit) and the words 0x8F4, 0x8F8, 0x8FC, 0x900,
//!   0xA1C, 0x418 of that buffer, as the CPU sees them then; for each address the window and
//!   byte offset the translated load resolves to (heap form: descriptor `MEM_HEAP_BASE + w`) and
//!   the window's kind: committed (coherent: the GPU reads this very memory, nothing to flush)
//!   or sparse (video-memory tiles: CPU writes are copied at the next ExecuteCommandLists from
//!   the write-watch, `sparse::before_execute`; an unmapped tile reads 0).
//! Then, in the same list right after the dispatch, a one-thread compute pass reads the same
//! descriptor and words through the windows on the GPU, decoding the descriptor as the
//! translated program does, into a coherent block; `local inputs: GPU read` lines give those
//! values beside the CPU's (at the next such dispatch for lists the GPU has run, and all at a
//! device removal).

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Mutex, OnceLock};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

use super::console_list::{self, Layout, Param};

/// lines per program and run; dispatches of more than `BIG_GROUPS` groups have their own count
const CAP: u32 = 12;
const CAP_BIG: u32 = 40;
const BIG_GROUPS: u64 = 256;
/// the program whose descriptor (s4 + `TABLE_OFF`) and constant words (`CB_OFFS`) are read
const WORDS_FOR: u64 = 0x5062_5695_d2e6_cbe7;
const TABLE_OFF: u32 = 0xCA0;
const CB_OFFS: [u32; 6] = [0x8F4, 0x8F8, 0x8FC, 0x900, 0xA1C, 0x418];
const DONE: u32 = 0x600D_F00D;
/// bytes per record in the readback pool: io (64) then res (64)
const SLOT: usize = 128;
const POOL: usize = 0x1_0000;

fn parse_list(v: &str) -> Vec<u64> {
    v.split(',').filter_map(|h| u64::from_str_radix(h.trim(), 16).ok()).collect()
}

fn wanted_list() -> &'static [u64] {
    static L: OnceLock<Vec<u64>> = OnceLock::new();
    L.get_or_init(|| {
        let l = std::env::var("XCL_LOCAL_DUMP_CS_INPUTS").map(|v| parse_list(&v)).unwrap_or_default();
        if !l.is_empty() {
            crate::xlog_info!("local inputs: on for {} programs", l.len());
        }
        l
    })
}

/// Whether a dispatch of `cs` with `groups` gets a line: under its cap (`CAP`; `CAP_BIG` for
/// one of more than `BIG_GROUPS` groups), counted in `counts`.
fn gate(counts: &mut HashMap<(u64, bool), u32>, wanted: &[u64], cs: u64, groups: [u32; 3]) -> bool {
    if cs == 0 || !wanted.contains(&cs) {
        return false;
    }
    let big = groups.iter().map(|&g| g as u64).product::<u64>() > BIG_GROUPS;
    let c = counts.entry((cs, big)).or_insert(0);
    if *c >= if big { CAP_BIG } else { CAP } {
        return false;
    }
    *c += 1;
    true
}

/// Whether the diagnostic is on (the culprit's pipeline → CS map is then kept too).
pub fn active() -> bool {
    !wanted_list().is_empty()
}

/// A buffer descriptor (V#) as the translated scalar buffer loads decode it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Vsharp {
    pub base: u64,
    pub stride: u32,
    pub records: u32,
    /// bytes a load may reach (`off < limit`, else 0)
    pub limit: u32,
}

pub fn decode_vsharp(w: [u32; 4]) -> Vsharp {
    let stride = (w[1] >> 16) & 0x3FFF;
    Vsharp { base: w[0] as u64 | ((w[1] & 0xFFFF) as u64) << 32, stride, records: w[2], limit: if stride == 0 { w[2] } else { w[2].wrapping_mul(stride) } }
}

/// The window and byte offset a translated load of address `ea` resolves to (`xcl_w`: 0 for a
/// 32-bit address, else the high half minus that of the unified memory's base).
pub fn window_of_ea(ea: u64, uma_base_hi: u32) -> (u32, u32) {
    let hi = (ea >> 32) as u32;
    (if hi == 0 { 0 } else { hi.wrapping_sub(uma_base_hi) }, ea as u32 & !3)
}

/// The word the CPU sees at window `w` offset `off` (None: no such window, out of it, not readable)
fn cpu_word(w: u32, off: u32) -> Option<u32> {
    let u = crate::gpu::uma::get()?;
    if w >= 4 {
        return None;
    }
    let (base, len) = u.window(w as usize);
    if len == 0 || off as usize + 4 > len {
        return None;
    }
    let p = base + off as usize;
    crate::gpu::shader::readable(p, 4).then(|| unsafe { (p as *const u32).read_volatile() })
}

/// what kind of memory window `w` offset `off` is
fn kind(w: u32, off: u32) -> String {
    let Some(u) = crate::gpu::uma::get() else { return "no unified memory".into() };
    if w >= 4 || u.window(w as usize).1 == 0 {
        return "no window (the shader reads 0)".into();
    }
    let p = u.window(w as usize).0 + off as usize;
    if u.is_committed_window(w as usize) {
        "committed/coherent".into()
    } else if u.is_sparse_window(w as usize) {
        let mapped = super::sparse::mapped_ranges().iter().any(|&(a, b)| p >= a && p < b);
        format!("sparse, tile {}", if mapped { "mapped (CPU writes uploaded at ExecuteCommandLists)" } else { "NOT mapped (GPU reads 0)" })
    } else {
        "unknown window kind".into()
    }
}

fn show(v: Option<u32>) -> String {
    v.map_or("not readable".into(), |v| format!("{v:#010x}"))
}

struct Pending {
    n: u32,
    cs: u64,
    s4: u32,
    base_hi: u32,
    /// the CPU's view at record time (`cpu_view`)
    cpu: Vec<Option<u32>>,
    /// the readback blocks of the passes just before and just after the dispatch
    before: Option<usize>,
    after: Option<usize>,
}

struct State {
    counts: HashMap<(u64, bool), u32>,
    pending: Vec<Pending>,
    pool: usize,
    next: usize,
    records: u32,
}

static STATE: Mutex<Option<State>> = Mutex::new(None);
/// the readback pipeline (raw `ID3D12PipelineState`; 0: could not be made)
static READ_PSO: OnceLock<usize> = OnceLock::new();

const READ_HLSL: &str = r#"
RWByteAddressBuffer io : register(u0);
RWByteAddressBuffer res : register(u1);
RWByteAddressBuffer m0 : register(u2);
RWByteAddressBuffer m1 : register(u3);
RWByteAddressBuffer m2 : register(u4);
RWByteAddressBuffer m3 : register(u5);
uint ld(uint w, uint a) {
    if (w == 0u) return m0.Load(a);
    if (w == 1u) return m1.Load(a);
    if (w == 2u) return m2.Load(a);
    if (w == 3u) return m3.Load(a);
    return 0xDEADDEADu;
}
[numthreads(1, 1, 1)]
void main() {
    uint s4 = io.Load(0u);
    uint basehi = io.Load(4u);
    uint t = (s4 + io.Load(8u)) & ~3u;
    uint4 v = uint4(ld(0u, t), ld(0u, t + 4u), ld(0u, t + 8u), ld(0u, t + 12u));
    res.Store4(0u, v);
    uint stride = (v.y >> 16) & 0x3FFFu;
    uint lim = stride == 0u ? v.z : v.z * stride;
    uint64_t base = ((uint64_t)(v.y & 0xFFFFu) << 32) | (uint64_t)v.x;
    [unroll] for (uint k = 0u; k < 6u; k++) {
        uint off = io.Load(12u + 4u * k);
        uint64_t ea = base + (uint64_t)off;
        uint hi = (uint)(ea >> 32);
        uint w = hi == 0u ? 0u : hi - basehi;
        res.Store(16u + 4u * k, off < lim ? ld(w, (uint)ea & ~3u) : 0u);
    }
    DeviceMemoryBarrier();
    res.Store(40u, 0x600DF00Du);
}
"#;

unsafe fn read_pso(list: &ID3D12GraphicsCommandList) -> Option<ID3D12PipelineState> {
    let raw = *READ_PSO.get_or_init(|| {
        let made = (|| -> Result<ID3D12PipelineState, String> {
            let mut dev: Option<ID3D12Device> = None;
            list.GetDevice(&mut dev).map_err(|e| e.to_string())?;
            let dev = dev.ok_or("no device")?;
            let rs = super::real::abi_rs_cached().ok_or("no ABI root signature")?;
            let rs = ID3D12RootSignature::from_raw_borrowed(&rs).ok_or("root signature")?;
            let dxil = crate::gpu::recomp::dxc::compile(READ_HLSL, "main", "cs_6_6", &[])?;
            dev.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(rs),
                CS: D3D12_SHADER_BYTECODE { pShaderBytecode: dxil.as_ptr() as _, BytecodeLength: dxil.len() },
                ..Default::default()
            })
            .map_err(|e| e.to_string())
        })();
        match made {
            Ok(p) => p.into_raw() as usize,
            Err(e) => {
                crate::xlog_warn!("local inputs: no GPU readback pipeline: {e}");
                0
            }
        }
    });
    let raw = raw as *mut c_void;
    ID3D12PipelineState::from_raw_borrowed(&raw).cloned()
}

/// The root parameter that gives bytes `[at, at + 4)` of the packed arguments.
fn source_of(l: &Layout, at: u32) -> String {
    for (i, p) in l.params.iter().enumerate() {
        let (start, len, what) = match *p {
            Param::Constants { at, count } => (at, 4 * count, "32-bit root constants"),
            Param::Descriptor { at } => (at, 8, "root CBV/SRV/UAV GPU address"),
            Param::Table { at } => (at, 4, "descriptor table address"),
        };
        if at >= start && at < start + len {
            return format!("root parameter {i} ({what}, word {})", (at - start) / 4);
        }
    }
    "no root parameter".into()
}

/// The CPU's view of the descriptor at s4 + `TABLE_OFF` (window 0) and of the `CB_OFFS` words
/// through the address it decodes: 10 values, and their description.
fn cpu_view(s4: u32, base_hi: u32) -> (Vec<Option<u32>>, Vec<String>) {
    let t = s4.wrapping_add(TABLE_OFF) & !3;
    let vw: [Option<u32>; 4] = std::array::from_fn(|k| cpu_word(0, t + 4 * k as u32));
    let mut vals: Vec<Option<u32>> = vw.to_vec();
    let mut parts = vec![format!("V# at s4+{TABLE_OFF:#x} (window 0 +{t:#x}, {}): [{}]", kind(0, t), vw.iter().map(|v| show(*v)).collect::<Vec<_>>().join(" "))];
    if vw.iter().all(|v| v.is_some()) {
        let d = decode_vsharp(vw.map(|v| v.unwrap()));
        parts.push(format!("buffer {:#x} stride {} records {:#x} limit {:#x}", d.base, d.stride, d.records, d.limit));
        for off in CB_OFFS {
            let (w, o) = window_of_ea(d.base.wrapping_add(off as u64), base_hi);
            let v = if off < d.limit { cpu_word(w, o) } else { Some(0) };
            vals.push(v);
            parts.push(format!("+{off:#x} {} (window {w} +{o:#x}, {}{})", show(v), kind(w, o), if off < d.limit { "" } else { ", past the limit: 0" }));
        }
    } else {
        vals.extend([None; 6]);
    }
    (vals, parts)
}

/// Readbacks: logged once the pass after the dispatch has run (at a device removal, all, with
/// what ran), then forgotten. Per value: the CPU's at record time, the GPU's just before and
/// just after the dispatch (each through the GPU's own descriptor), the CPU's now.
fn flush(all: bool) {
    let mut g = STATE.lock().unwrap();
    let Some(s) = g.as_mut() else { return };
    let mut keep = Vec::new();
    for p in s.pending.drain(..) {
        let read = |res: Option<usize>| -> Option<Vec<u32>> {
            let res = res?;
            let w = |i: usize| unsafe { ((res + 4 * i) as *const u32).read_volatile() };
            (w(10) == DONE).then(|| (0..10).map(w).collect())
        };
        let (before, after) = (read(p.before), read(p.after));
        let finished = after.is_some() || (p.after.is_none() && before.is_some());
        if !finished && !all {
            keep.push(p);
            continue;
        }
        let (now, _) = cpu_view(p.s4, p.base_hi);
        let gpu = |v: &Option<Vec<u32>>, k: usize| v.as_ref().map_or("not run".to_string(), |v| format!("{:#010x}", v[k]));
        let items: Vec<String> = (0..10)
            .map(|k| {
                let name = if k < 4 { format!("V#[{k}]") } else { format!("+{:#x}", CB_OFFS[k - 4]) };
                let differs = before.as_ref().is_some_and(|b| p.cpu[k] != Some(b[k]));
                format!("{name} cpu@record {} gpu@before {} gpu@after {} cpu@now {}{}", show(p.cpu[k]), gpu(&before, k), gpu(&after, k), show(now[k]), if differs { " DIFFERENT" } else { "" })
            })
            .collect();
        crate::xlog_info!(
            "local inputs: GPU read #{} (cs {:016x}{}): {}",
            p.n,
            p.cs,
            match (&before, &after) {
                (Some(_), Some(_)) => "",
                (Some(_), None) => ", the pass before the dispatch ran, the one after did not: the dispatch did not finish",
                (None, _) => ", the pass before the dispatch did not run",
            },
            items.join(", ")
        );
    }
    s.pending = keep;
}

/// At a device removal: every readback still pending.
pub fn on_device_removed() {
    if active() {
        flush(true);
    }
}

/// A coherent block of `SLOT` bytes for one readback (None: none left, or not coherent).
fn slot(u: &crate::gpu::uma::Uma) -> Option<usize> {
    let mut g = STATE.lock().unwrap();
    let s = g.as_mut()?;
    if s.pool == 0 {
        s.pool = u.alloc(POOL, POOL).map_or(usize::MAX, |p| p as usize);
        if s.pool != usize::MAX && !u.is_committed(s.pool) {
            crate::xlog_warn!("local inputs: the readback block is not in a coherent window: no GPU readback");
            s.pool = usize::MAX;
        }
    }
    if s.pool == usize::MAX || s.next + SLOT > POOL {
        return None;
    }
    s.next += SLOT;
    Some(s.pool + s.next - SLOT)
}

/// Record the one-thread read pass on `list` (it changes the compute root signature, root
/// arguments and pipeline): its result block (`DONE` at word 10 once it ran).
unsafe fn read_pass(list: &ID3D12GraphicsCommandList, s4: u32, base_hi: u32) -> Option<usize> {
    let u = crate::gpu::uma::get()?;
    let rd = read_pso(list)?;
    let slot = slot(u)?;
    let io: [u32; 9] = [s4, base_hi, TABLE_OFF, CB_OFFS[0], CB_OFFS[1], CB_OFFS[2], CB_OFFS[3], CB_OFFS[4], CB_OFFS[5]];
    std::ptr::copy_nonoverlapping(io.as_ptr(), slot as *mut u32, io.len());
    std::ptr::write_bytes((slot + 64) as *mut u8, 0, 64);
    let io_va = super::real::uma_va_of(slot)?;
    let res_va = super::real::uma_va_of(slot + 64)?;
    let windows = super::real::uma_window_vas();
    let first = *windows.first()?;
    let rs = super::real::abi_rs_cached()?;
    let rs = ID3D12RootSignature::from_raw_borrowed(&rs)?;
    list.SetComputeRootSignature(rs);
    list.SetComputeRootUnorderedAccessView(0, io_va);
    list.SetComputeRootUnorderedAccessView(1, res_va);
    for k in 0..4u32 {
        list.SetComputeRootUnorderedAccessView(2 + k, windows.get(k as usize).copied().unwrap_or(first));
    }
    list.SetPipelineState(&rd);
    list.Dispatch(1, 1, 1);
    Some(slot + 64)
}

/// A dispatch being recorded with a GPU readback (`before_dispatch` → `after_dispatch`).
pub struct Probe(u32);

/// A Dispatch of `pso` (the title's `title_pso`) is about to be recorded on `list` with the
/// root arguments `args` packed by `layout`; `source` says where the group counts come from.
/// Logs its line; for `WORDS_FOR` records the read pass before it. Returns the probe to give
/// `after_dispatch` and whether the list's compute state was changed (bind it again).
#[allow(clippy::too_many_arguments)]
pub unsafe fn before_dispatch(list: &ID3D12GraphicsCommandList, title_pso: usize, pso: usize, groups: [u32; 3], layout: Option<&Layout>, args: &[u8], user_sgprs: Option<u32>, source: &str) -> (Option<Probe>, bool) {
    if !active() {
        return (None, false);
    }
    let cs = super::culprit::cs_hash(title_pso, pso);
    let n = {
        let mut g = STATE.lock().unwrap();
        let s = g.get_or_insert_with(|| State { counts: HashMap::new(), pending: Vec::new(), pool: 0, next: 0, records: 0 });
        if !gate(&mut s.counts, wanted_list(), cs, groups) {
            return (None, false);
        }
        s.records += 1;
        s.records
    };
    flush(false);
    // s4.. as the ABI gives them: the inline part of the packed root arguments; s[0:1] the
    // address of the rest (the argument table)
    let inline = (layout.map_or(0, |l| console_list::inline_for(l, user_sgprs)) as usize).min(args.len());
    let word_at = |at: usize| args.get(at..at + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap()));
    let word = |at: usize| if at + 4 <= inline { word_at(at) } else { None };
    let (s4, s5) = (word(0), word(4));
    let src = |at: u32, v: Option<u32>| match (layout, v) {
        (Some(l), Some(_)) => source_of(l, at),
        (Some(_), None) => "not inline: 0 (the program has fewer user SGPRs, or the arguments are in the table)".into(),
        (None, _) => "unknown (no root signature layout): 0".into(),
    };
    let total: u64 = groups.iter().map(|&g| g as u64).product();
    let sgprs: Vec<String> = (0..inline / 4).map(|k| format!("s{}={:#x}", 4 + k, word(4 * k).unwrap_or(0))).collect();
    let mut parts = vec![
        format!("groups {}x{}x{} ({total}) from {source}", groups[0], groups[1], groups[2]),
        format!("user SGPRs {} inline bytes of {} (shader has {})", inline, layout.map_or("unknown".to_string(), |l| l.size.to_string()), user_sgprs.map_or("unknown".to_string(), |v| v.to_string())),
        format!("s4 {} from {}", show(s4), src(0, s4)),
        format!("s5 {} from {}", show(s5), src(4, s5)),
        format!("inline [{}]", sgprs.join(" ")),
    ];
    if total > BIG_GROUPS {
        // the argument table s[0:1] points at: the packed arguments past the inline part, as
        // XCL copies them for this dispatch
        let table: Vec<String> = (0..16).map(|k| word_at(inline + 4 * k).map_or("-".into(), |w| format!("{w:#x}"))).collect();
        let params: Vec<String> = layout.map_or(vec!["unknown".into()], |l| {
            l.params
                .iter()
                .enumerate()
                .map(|(i, p)| match *p {
                    Param::Constants { at, count } => format!("p{i} constants @{at:#x} x{count}"),
                    Param::Descriptor { at } => format!("p{i} root view @{at:#x}"),
                    Param::Table { at } => format!("p{i} table @{at:#x}"),
                })
                .collect()
        });
        parts.push(format!("argument table s[0:1] (args from byte {inline:#x}) first 16 words [{}]", table.join(" ")));
        parts.push(format!("root parameters [{}]", params.join(", ")));
    }
    let u = crate::gpu::uma::get();
    if cs != WORDS_FOR || u.is_none() {
        crate::xlog_info!("local inputs: #{n} cs {cs:016x}: {}", parts.join("; "));
        return (None, false);
    }
    let base_hi = (u.unwrap().base() >> 32) as u32;
    let s4v = s4.unwrap_or(0);
    let (cpu, view) = cpu_view(s4v, base_hi);
    parts.extend(view);
    crate::xlog_info!("local inputs: #{n} cs {cs:016x}: {}", parts.join("; "));
    let before = read_pass(list, s4v, base_hi);
    STATE.lock().unwrap().as_mut().unwrap().pending.push(Pending { n, cs, s4: s4v, base_hi, cpu, before, after: None });
    (Some(Probe(n)), before.is_some())
}

/// The dispatch of `probe` was recorded: the read pass after it. True: the list's compute state
/// was changed (bind it again).
pub unsafe fn after_dispatch(list: &ID3D12GraphicsCommandList, probe: Option<Probe>) -> bool {
    let Some(Probe(n)) = probe else { return false };
    let at = {
        let g = STATE.lock().unwrap();
        g.as_ref().and_then(|s| s.pending.iter().find(|p| p.n == n).map(|p| (p.s4, p.base_hi)))
    };
    let Some((s4, base_hi)) = at else { return false };
    let after = read_pass(list, s4, base_hi);
    if let Some(p) = STATE.lock().unwrap().as_mut().and_then(|s| s.pending.iter_mut().find(|p| p.n == n)) {
        p.after = after;
    }
    after.is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vsharp_decodes_as_the_translated_loads() {
        let d = decode_vsharp([0x1234_5600, 0x0010_0002, 0x40, 0x0002_7FAC]);
        assert_eq!(d, Vsharp { base: 0x2_1234_5600, stride: 16, records: 0x40, limit: 0x400 });
        let raw = decode_vsharp([0x10000, 0, 0x1000, 0]);
        assert_eq!((raw.base, raw.stride, raw.limit), (0x10000, 0, 0x1000));
    }

    /// The gate: a listed program logs (the hash list as the variable gives it), capped per
    /// program, large dispatches on their own cap; an unknown program (0) or another never.
    #[test]
    fn gate_logs_listed_programs_up_to_their_caps() {
        let l = parse_list("50625695d2e6cbe7, 80f65d1925a2df48,xyz");
        assert_eq!(l, vec![0x5062_5695_d2e6_cbe7, 0x80f6_5d19_25a2_df48]);
        let mut c = HashMap::new();
        let cs = 0x80f6_5d19_25a2_df48;
        assert!(!gate(&mut c, &l, 0, [1, 1, 1]));
        assert!(!gate(&mut c, &l, 0x1234, [1, 1, 1]));
        assert_eq!((0..20).filter(|_| gate(&mut c, &l, cs, [64, 1, 1])).count(), CAP as usize);
        assert_eq!((0..60).filter(|_| gate(&mut c, &l, cs, [7635, 1, 1])).count(), CAP_BIG as usize);
        assert!(gate(&mut c, &l, 0x5062_5695_d2e6_cbe7, [32, 32, 2]));
    }

    #[test]
    fn readback_pass_compiles() {
        if crate::gpu::recomp::dxc::location().is_err() {
            crate::gpu::recomp::runner::skip_test("no DXC");
            return;
        }
        crate::gpu::recomp::dxc::compile(READ_HLSL, "main", "cs_6_6", &[]).unwrap();
    }

    #[test]
    fn addresses_resolve_to_windows_as_xcl_w() {
        // a 32-bit address: window 0
        assert_eq!(window_of_ea(0x0000_0000_8000_0904, 0x7), (0, 0x8000_0904));
        // a full address in the second window of unified memory based at 0x7_0000_0000
        assert_eq!(window_of_ea(0x0000_0008_0000_0900, 0x7), (1, 0x900));
        // below the base: a window the shader cannot reach (it reads 0)
        assert!(window_of_ea(0x0000_0003_0000_0000, 0x7).0 >= 4);
        assert_eq!(window_of_ea(0x0000_0007_0000_0903, 0x7), (0, 0x900));
    }
}
