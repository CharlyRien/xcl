//! Root arguments of console code, bound when it dispatches with a translated pipeline.
//!
//! A translated console shader reads its root arguments the console's way (user SGPRs s4…,
//! the rest from a table at s[0:1]: `console_list::user_data`) and memory through XCL's
//! shader ABI (u0 io, u1 res, u2… unified memory). So on a command list of console code the
//! compute root signature and root arguments are kept here – packed by the root signature's
//! layout – instead of going to the PC list, and each Dispatch with a translated pipeline
//! binds the ABI root signature, writes the user SGPRs (`io`) and the argument table to
//! unified memory and dispatches. That memory stays until the PC allocator the recording is
//! on is reset (`alloc_pool`: later than the title's Reset when the GPU still has it to run).

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

use super::console_list::{self, Layout, Param};

#[derive(Default)]
struct State {
    layout: Option<Arc<Layout>>,
    args: Vec<u8>,
    /// the graphics root signature's layout and arguments
    layout_gfx: Option<Arc<Layout>>,
    args_gfx: Vec<u8>,
    /// the title's graphics root signature (a PC object), bound before a draw with an
    /// untranslated pipeline – created with it – since SetGraphicsRootSignature binds nothing
    rs_gfx: usize,
    /// the title's compute root signature (a PC object), preserved for internal-pass restoration
    rs_cs: usize,
    pso: usize,
    /// 32-bit addresses of the view heap and the sampler heap set (0: none)
    heaps: [u32; 2],
    /// the PC allocator this recording is on
    allocator: usize,
    /// unified memory of this recording (io, argument tables)
    uma: Vec<usize>,
    /// where the next draw's io and argument table go (blocks in `uma`)
    arena: crate::gpu::uma::Arena,
    res_buf: Option<(ID3D12Resource, u64, usize)>,
    res_retired: Vec<(ID3D12Resource, u64, usize)>,
    /// rewritten indirect arguments (`scratch_args`): the block, its bytes in use, and the
    /// bytes of blocks this recording took
    scratch: Option<(ID3D12Resource, u64, usize)>,
    scratch_used: u64,
    scratch_taken: u64,
}

/// A recording's blocks of rewritten indirect arguments, and most bytes of them (past it, its
/// two-pass ExecuteIndirects are skipped)
const SCRATCH_BLOCK: u64 = 1 << 20;
const SCRATCH_CAP: u64 = 16 << 20;
/// Most bytes of buffers kept for reuse (`FREE_RES`)
const FREE_RES_CAP: u64 = 256 << 20;

static ALLOC_RES: Mutex<Option<HashMap<usize, Vec<(ID3D12Resource, u64, usize)>>>> = Mutex::new(None);
static FREE_RES: Mutex<Vec<(ID3D12Resource, u64, usize)>> = Mutex::new(Vec::new());

fn hold_res_until_reset(allocator: usize, mut bufs: Vec<(ID3D12Resource, u64, usize)>) {
    if bufs.is_empty() || allocator == 0 {
        return;
    }
    ALLOC_RES.lock().unwrap().get_or_insert_with(HashMap::new).entry(allocator).or_default().append(&mut bufs);
}

pub(super) fn release_allocator_res(allocator: usize) {
    let Some(bufs) = ALLOC_RES.lock().unwrap().as_mut().and_then(|m| m.remove(&allocator)) else { return };
    let mut free = FREE_RES.lock().unwrap();
    for b in bufs {
        let held: u64 = free.iter().map(|f| f.1).sum();
        if pool_keeps(free.len(), held, b.1) {
            free.push(b);
        }
    }
}

/// Whether `FREE_RES`, holding `count` buffers of `held` bytes, keeps one of `bytes` more.
fn pool_keeps(count: usize, held: u64, bytes: u64) -> bool {
    count < 16 && held + bytes <= FREE_RES_CAP
}

static STATES: Mutex<Option<HashMap<usize, State>>> = Mutex::new(None);

/// draws / dispatches of console code: with a translated pipeline, skipped, with a PC one
static COUNTS: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

/// draws and dispatches of console code so far (`XCL_ACTIVITY`)
pub fn activity() -> (u64, u64) {
    (COUNTS[DRAWS].load(Ordering::Relaxed), COUNTS[DISPATCHES].load(Ordering::Relaxed))
}
const DRAWS: usize = 0;
const DRAWS_SKIPPED: usize = 1;
const DRAWS_PC: usize = 2;
const DISPATCHES: usize = 3;
const DISPATCHES_SKIPPED: usize = 4;

/// LOCAL: mesh draws (`XCL_ACTIVITY`'s "mesh (2 s)" line): two-pass DispatchMesh run / skipped,
/// two-pass ExecuteIndirect(DispatchMesh) run / skipped, ExecuteIndirect with a DispatchMesh
/// signature whose pipeline is not a two-pass one (not built yet, refused, stand-in), a
/// two-pass pipeline with a signature that is not a DispatchMesh one, DispatchMesh without a
/// two-pass pipeline, and the indirect commands (max) of the run two-pass ExecuteIndirects
pub static LOCAL_MESH: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];

fn local_mesh(k: usize, n: u64) {
    LOCAL_MESH[k].fetch_add(n, Ordering::Relaxed);
}

fn count(k: usize) {
    let n = COUNTS[k].fetch_add(1, Ordering::Relaxed) + 1;
    if n.is_power_of_two() && n >= 64 {
        let c: Vec<u64> = COUNTS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
        let (allocators, blocks) = console_list::pending_blocks();
        let (coherent_used, coherent_total, sparse_used) = crate::gpu::uma::get()
            .map_or((0, 0, 0), |u| (u.used() >> 20, u.size() >> 20, u.sparse_used() >> 20));
        crate::xlog_info!(
            "D3D12.X: console code so far: {} draws with translated pipelines ({} skipped, {} with PC pipelines), {} dispatches ({} skipped); coherent {coherent_used}/{coherent_total} MB, sparse {sparse_used} MB, {blocks} blocks held for {allocators} allocators",
            c[DRAWS], c[DRAWS_SKIPPED], c[DRAWS_PC], c[DISPATCHES], c[DISPATCHES_SKIPPED],
        );
    }
}

fn with<R>(list: *mut c_void, f: impl FnOnce(&mut State) -> R) -> R {
    f(STATES.lock().unwrap().get_or_insert_with(HashMap::new).entry(list as usize).or_default())
}

/// The PC allocator the recording `list` starts on the title's `allocator` is on (the title's
/// own where `alloc_pool` does not know the list; 0: no recording).
fn recording_allocator(list: *mut c_void, allocator: usize) -> usize {
    if allocator == 0 {
        return 0;
    }
    super::alloc_pool::recording_backing(list as usize).unwrap_or(allocator)
}

/// The list was created or reset onto `allocator` with `initial_pso`: the previous
/// recording's memory is the previous allocator's until that one is reset, and pipeline
/// and root signature state reset per D3D12 `CreateCommandList` / `Reset` semantics.
pub(super) fn reset(list: *mut c_void, allocator: usize, initial_pso: usize) {
    let allocator = recording_allocator(list, allocator);
    let (old, blocks, res_bufs) = with(list, |s| {
        let old = std::mem::replace(&mut s.allocator, allocator);
        s.arena = Default::default();
        s.pso = initial_pso;
        s.layout = None;
        s.args.clear();
        s.layout_gfx = None;
        s.args_gfx.clear();
        s.rs_gfx = 0;
        s.rs_cs = 0;
        s.heaps = [0; 2];
        // (the scratch blocks are in `res_retired` already)
        (s.scratch, s.scratch_used, s.scratch_taken) = (None, 0, 0);
        let mut res_bufs = std::mem::take(&mut s.res_retired);
        if let Some(b) = s.res_buf.take() {
            res_bufs.push(b);
        }
        (old, std::mem::take(&mut s.uma), res_bufs)
    });
    console_list::hold_until_reset(old, blocks);
    hold_res_until_reset(old, res_bufs);
}

/// `ClearState(pso)`: resets pipeline and root-argument state back to creation defaults
/// with `pso` bound, keeping the current recording's allocator and UMA arena intact.
pub(super) fn clear_state(list: *mut c_void, pso: usize) {
    with(list, |s| {
        s.pso = pso;
        s.layout = None;
        s.args.clear();
        s.layout_gfx = None;
        s.args_gfx.clear();
        s.rs_gfx = 0;
        s.heaps = [0; 2];
        s.rs_cs = 0;
    });
}

/// Carry bound pipeline and root-signature / root-argument state from `from` to `to` across a
/// deferred mid-list split (`to` records into `allocator` with its own fresh UMA arena while
/// `from` retains its previously allocated blocks until `allocator` resets).
pub(super) fn inherit_for_split(from: *mut c_void, to: *mut c_void, allocator: usize) {
    let (layout, args, layout_gfx, args_gfx, rs_gfx, rs_cs, pso, heaps) = with(from, |s| {
        (s.layout.clone(), s.args.clone(), s.layout_gfx.clone(), s.args_gfx.clone(), s.rs_gfx, s.rs_cs, s.pso, s.heaps)
    });
    let allocator = recording_allocator(to, allocator);
    let (old, blocks) = with(to, |s| {
        let old = std::mem::replace(&mut s.allocator, allocator);
        s.arena = Default::default();
        s.layout = layout;
        s.args = args;
        s.layout_gfx = layout_gfx;
        s.args_gfx = args_gfx;
        s.rs_gfx = rs_gfx;
        s.rs_cs = rs_cs;
        s.pso = pso;
        s.heaps = heaps;
        (old, std::mem::take(&mut s.uma))
    });
    console_list::hold_until_reset(old, blocks);
}

/// The list is gone.
pub(super) fn forget(list: *mut c_void) {
    if let Some(mut s) = STATES.lock().unwrap().as_mut().and_then(|m| m.remove(&(list as usize))) {
        console_list::hold_until_reset(s.allocator, s.uma);
        if let Some(b) = s.res_buf.take() {
            s.res_retired.push(b);
        }
        hold_res_until_reset(s.allocator, s.res_retired);
    }
}

#[cfg(test)]
pub(crate) fn seed_reset_state_for_test(list: *mut c_void, allocator: usize, pso: usize, uma: Vec<usize>) {
    with(list, |s| {
        s.allocator = allocator;
        s.pso = pso;
        s.layout = Some(Arc::new(Layout::default()));
        s.args = vec![1, 2, 3, 4];
        s.layout_gfx = Some(Arc::new(Layout::default()));
        s.args_gfx = vec![5, 6, 7, 8];
        s.uma = uma;
    });
}

#[cfg(test)]
pub(crate) fn reset_state_for_test(list: *mut c_void) -> (usize, usize, Vec<u8>, Vec<u8>, Vec<usize>) {
    with(list, |s| (s.allocator, s.pso, s.args.clone(), s.args_gfx.clone(), s.uma.clone()))
}

#[cfg(test)]
pub(crate) fn remove_state_for_test(list: *mut c_void) {
    STATES.lock().unwrap().as_mut().and_then(|m| m.remove(&(list as usize)));
}

/// SetDescriptorHeaps: the console addresses of `heaps`, for shaders that index them themselves
/// (`Layout::heaps`).
pub(super) fn set_heaps(list: *mut c_void, heaps: &[usize]) {
    let mut bases = [0u32; 2];
    for (sampler, base) in heaps.iter().filter_map(|&h| super::console_heap::base(h)) {
        bases[sampler as usize] = base;
    }
    with(list, |s| s.heaps = bases);
}

/// bytes of parameter `i` at constant `offset` (of the compute or the graphics arguments)
fn put_in(list: *mut c_void, gfx: bool, i: u32, offset: u32, bytes: &[u8]) {
    with(list, |s| {
        let (layout, args) = if gfx { (&s.layout_gfx, &mut s.args_gfx) } else { (&s.layout, &mut s.args) };
        let Some(at) = layout.as_ref().and_then(|l| l.params.get(i as usize)).map(|p| match *p {
            Param::Constants { at, .. } => at as usize + 4 * offset as usize,
            Param::Descriptor { at } | Param::Table { at } => at as usize,
        }) else {
            return;
        };
        if let Some(t) = args.get_mut(at..at + bytes.len()) {
            t.copy_from_slice(bytes);
        }
    })
}

fn put(list: *mut c_void, i: u32, offset: u32, bytes: &[u8]) {
    put_in(list, false, i, offset, bytes)
}

/// #34 SetGraphicsRootSignature (PC #30)
pub(super) unsafe extern "system" fn set_graphics_root_signature(this: *mut c_void, rs: *mut c_void) {
    let layout = console_list::layout_of(rs as usize);
    with(this, |s| {
        s.args_gfx = vec![0u8; layout.as_ref().map_or(0, |l| l.size as usize)];
        s.layout_gfx = layout;
        s.rs_gfx = rs as usize;
    });
}

/// #36 SetGraphicsRootDescriptorTable (PC #32)
pub(super) unsafe extern "system" fn set_graphics_root_table(this: *mut c_void, i: u32, handle: u64) {
    put_in(this, true, i, 0, &(handle as u32).to_le_bytes());
}

/// #38 SetGraphicsRoot32BitConstant (PC #34)
pub(super) unsafe extern "system" fn set_graphics_root_constant(this: *mut c_void, i: u32, value: u32, offset: u32) {
    put_in(this, true, i, offset, &value.to_le_bytes());
}

/// #40 SetGraphicsRoot32BitConstants (PC #36)
pub(super) unsafe extern "system" fn set_graphics_root_constants(this: *mut c_void, i: u32, n: u32, data: *const u32, offset: u32) {
    if !data.is_null() {
        let bytes: Vec<u8> = std::slice::from_raw_parts(data, n as usize).iter().flat_map(|w| w.to_le_bytes()).collect();
        put_in(this, true, i, offset, &bytes);
    }
}

/// #42 / #44 / #46 SetGraphicsRoot{ConstantBuffer,ShaderResource,UnorderedAccess}View (PC #38 /
/// #40 / #42)
pub(super) unsafe extern "system" fn set_graphics_root_descriptor(this: *mut c_void, i: u32, va: u64) {
    note_address(va);
    put_in(this, true, i, 0, &va.to_le_bytes());
}

/// `XCL_TRACE_ADDRESSES=1`: root descriptor addresses of console code – in unified memory
/// (translated shaders see them) or not – counted and logged
fn note_address(va: u64) {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    static IN: AtomicU64 = AtomicU64::new(0);
    static OUT: AtomicU64 = AtomicU64::new(0);
    if !*ON.get_or_init(|| std::env::var_os("XCL_TRACE_ADDRESSES").is_some()) {
        return;
    }
    let inside = crate::gpu::uma::get().is_some_and(|u| u.contains(va as usize));
    let (i, o) = if inside { (IN.fetch_add(1, Ordering::Relaxed) + 1, OUT.load(Ordering::Relaxed)) } else { (IN.load(Ordering::Relaxed), OUT.fetch_add(1, Ordering::Relaxed) + 1) };
    if (i + o).is_power_of_two() {
        crate::xlog_info!("D3D12.X: root descriptor addresses: {i} in unified memory, {o} outside (latest {va:#x})");
    }
}

/// command signatures that change root arguments (they need the title's root signature)
static ROOT_SIGNATURES: Mutex<Option<std::collections::HashSet<usize>>> = Mutex::new(None);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DispatchMeshSig {
    companion_dispatch: usize,
    byte_stride: u32,
    mesh_offset: u32,
}

static DISPATCH_MESH_SIGS: Mutex<Option<HashMap<usize, DispatchMeshSig>>> = Mutex::new(None);

pub(super) fn note_root_changing_signature(sig: usize) {
    ROOT_SIGNATURES.lock().unwrap().get_or_insert_with(Default::default).insert(sig);
}

/// command signatures with a Dispatch argument (noted at every creation, so a signature made
/// at a freed one's address does not keep its kind)
static DISPATCH_SIGNATURES: Mutex<Option<std::collections::HashSet<usize>>> = Mutex::new(None);

pub(super) fn note_signature_dispatches(sig: usize, dispatches: bool) {
    let mut s = DISPATCH_SIGNATURES.lock().unwrap();
    let s = s.get_or_insert_with(Default::default);
    if dispatches {
        s.insert(sig);
    } else {
        s.remove(&sig);
    }
}

fn signature_dispatches(sig: usize) -> bool {
    DISPATCH_SIGNATURES.lock().unwrap().as_ref().is_some_and(|s| s.contains(&sig))
}

pub(super) fn note_dispatch_mesh_signature(sig: usize, companion_dispatch: usize, byte_stride: u32, mesh_offset: u32) {
    DISPATCH_MESH_SIGS
        .lock()
        .unwrap()
        .get_or_insert_with(HashMap::new)
        .insert(sig, DispatchMeshSig { companion_dispatch, byte_stride, mesh_offset });
}

fn dispatch_mesh_sig_of(sig: usize) -> Option<DispatchMeshSig> {
    DISPATCH_MESH_SIGS.lock().unwrap().as_ref()?.get(&sig).copied()
}

unsafe fn emit_uav_barrier(list: &ID3D12GraphicsCommandList, res: Option<&ID3D12Resource>) {
    let mut b = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                pResource: std::mem::ManuallyDrop::new(res.cloned()),
            }),
        },
    };
    list.ResourceBarrier(&[std::ptr::read(&b)]);
    std::mem::ManuallyDrop::drop(&mut (*b.Anonymous.UAV).pResource);
}

// LOCAL (agent N): `XCL_LOCAL_MS_PROBE=1` – in a captured frame (rtsum EICAP), right after a two-pass
// compute pass and with its root arguments still bound, a one-thread pass reads what the retail
// title's MS programs read for their counts: the 32-bit descriptor table in s17, its first V# (the
// meshlet buffer), that buffer's dword 0 (count) and entries 0 / 1 (dword at 16g + 12: vertices - 1
// in [6:0], primitives - 1 in [13:7]), and writes them into the unused header dwords 2 / 3 of
// records 0..3: r0 = (s17, s18), r1 = (V#.x, V#.y), r2 = (V#.z, count), r3 = (entry 0, entry 1).
pub(crate) const LOCAL_MS_PROBE_HLSL: &str = r#"RWByteAddressBuffer io : register(u0);
RWByteAddressBuffer res : register(u1);
RWByteAddressBuffer m0 : register(u2);
RWByteAddressBuffer m1 : register(u3);
RWByteAddressBuffer m2 : register(u4);
RWByteAddressBuffer m3 : register(u5);
static const uint rd = XCL_RD;
uint ld(uint hi, uint lo) {
    uint w = hi == 0u ? 0u : hi - io.Load(448u);
    lo &= ~3u;
    if (w == 0u) return m0.Load(lo);
    if (w == 1u) return m1.Load(lo);
    if (w == 2u) return m2.Load(lo);
    if (w == 3u) return m3.Load(lo);
    return 0xEEEEEEEEu;
}
[numthreads(1, 1, 1)]
void main() {
    uint t = io.Load(68u);
    uint4 v = uint4(ld(0u, t), ld(0u, t + 4u), ld(0u, t + 8u), ld(0u, t + 12u));
    uint bhi = v.y & 0xFFFFu, blo = v.x;
    uint cnt = v.z != 0u ? ld(bhi, blo) : 0xBAD0BAD0u;
    uint e0 = v.z > 12u ? ld(bhi, blo + 12u) : 0xBAD0BAD0u;
    uint e1 = v.z > 28u ? ld(bhi, blo + 28u) : 0xBAD0BAD0u;
    uint d[8] = { t, io.Load(72u), v.x, v.y, v.z, cnt, e0, e1 };
    for (uint k = 0u; k < 4u; k++) {
        res.Store((k * rd + 2u) * 4u, d[2u * k]);
        res.Store((k * rd + 3u) * 4u, d[2u * k + 1u]);
    }
}
"#;

unsafe fn local_ms_probe(this: *mut c_void, list: &ID3D12GraphicsCommandList, tp: &super::real::TwoPassPso, res: &ID3D12Resource, record_bytes: u64) {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("XCL_LOCAL_MS_PROBE").is_some()) || !super::rtsum::capturing() {
        return;
    }
    static CACHE: Mutex<Option<HashMap<(usize, usize, u32), Option<ID3D12PipelineState>>>> = Mutex::new(None);
    let Some(dev) = super::real::list_device(this) else { return };
    let rd = (record_bytes / 4) as u32;
    let key = (dev.as_raw() as usize, tp.compute_rs, rd);
    let pso = {
        let mut c = CACHE.lock().unwrap();
        let m = c.get_or_insert_with(HashMap::new);
        m.entry(key)
            .or_insert_with(|| {
                let rs_raw = tp.compute_rs as *mut c_void;
                let rs = ID3D12RootSignature::from_raw_borrowed(&rs_raw)?;
                let dxil = match crate::gpu::recomp::dxc::compile_cached(&LOCAL_MS_PROBE_HLSL.replace("XCL_RD", &format!("{rd}u")), "main", "cs_6_6") {
                    Ok(d) => d.0,
                    Err(e) => {
                        crate::xlog_warn!("ms probe: {e}");
                        return None;
                    }
                };
                let mut cdesc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::ManuallyDrop::new(Some(rs.clone())),
                    CS: D3D12_SHADER_BYTECODE { pShaderBytecode: dxil.as_ptr() as *const c_void, BytecodeLength: dxil.len() },
                    ..Default::default()
                };
                let r: Result<ID3D12PipelineState, _> = dev.CreateComputePipelineState(&cdesc);
                std::mem::ManuallyDrop::drop(&mut cdesc.pRootSignature);
                match r {
                    Ok(p) => Some(p),
                    Err(e) => {
                        crate::xlog_warn!("ms probe pso: {e}");
                        None
                    }
                }
            })
            .clone()
    };
    let Some(pso) = pso else { return };
    list.SetPipelineState(&pso);
    list.Dispatch(1, 1, 1);
    emit_uav_barrier(list, Some(res));
}

unsafe fn emit_transition(list: &ID3D12GraphicsCommandList, res: &ID3D12Resource, before: D3D12_RESOURCE_STATES, after: D3D12_RESOURCE_STATES) {
    let mut b = D3D12_RESOURCE_BARRIER {
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
    list.ResourceBarrier(&[std::ptr::read(&b)]);
    std::mem::ManuallyDrop::drop(&mut (*b.Anonymous.Transition).pResource);
}

unsafe fn default_uav_buffer(dev: &ID3D12Device, bytes: u64, what: &str) -> Result<ID3D12Resource, String> {
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Alignment: 0,
        Width: bytes,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_UNKNOWN,
        SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
    };
    let mut res: Option<ID3D12Resource> = None;
    dev.CreateCommittedResource(
        &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
        D3D12_HEAP_FLAG_NONE,
        &desc,
        D3D12_RESOURCE_STATE_COMMON,
        None,
        &mut res,
    )
    .map_err(|e| format!("{what} ({bytes:#x} B): {e}"))?;
    res.ok_or_else(|| format!("null {what}"))
}

/// Groups of a two-pass dispatch that have a record: one record a group in `cap_bytes` of
/// records of `record_bytes`, and every group in X, the mesh pass's `split` times as many (at most
/// 65 535 a dimension). The compute pass writes record `gid.x` with no bound of its own.
fn record_groups(cap_bytes: u64, record_bytes: u64, split: u32) -> u32 {
    (cap_bytes / record_bytes.max(1)).min(65_535 / split.max(1) as u64) as u32
}

/// The ExecuteIndirects of a two-pass ExecuteIndirect of `max` commands, each of one command and
/// followed by a UAV barrier on the records: (command, compute pass). Each command's compute pass
/// writes its records from record 0, so the next one goes only after the mesh pass drew them.
fn two_pass_steps(max: u32) -> impl Iterator<Item = (u32, bool)> {
    (0..max).flat_map(|k| [(k, true), (k, false)])
}

/// The record buffer of this list for `need_bytes` (at most 128 MiB): buffer, address, bytes.
unsafe fn ensure_res_buffer(this: *mut c_void, need_bytes: u64) -> Result<(ID3D12Resource, u64, u64), String> {
    let need = need_bytes.max(4 << 20).next_multiple_of(1 << 20).min(128 << 20);
    if let Some((buf, cap, _dev)) = with(this, |s| s.res_buf.as_ref().filter(|(_, cap, _)| *cap >= need).cloned()) {
        let va = buf.GetGPUVirtualAddress();
        return Ok((buf, va, cap));
    }
    let dev = super::real::list_device(this).ok_or("no device for two-pass buffer")?;
    let dev_key = dev.as_raw() as usize;
    let picked = {
        let mut free = FREE_RES.lock().unwrap();
        free.iter().position(|(_, cap, d)| *d == dev_key && *cap >= need).map(|i| free.swap_remove(i))
    };
    let entry = match picked {
        Some(e) => e,
        None => (default_uav_buffer(&dev, need, "two-pass res buffer")?, need, dev_key),
    };
    let (buf, cap) = (entry.0.clone(), entry.1);
    let va = buf.GetGPUVirtualAddress();
    with(this, |s| {
        if let Some(old) = s.res_buf.replace(entry) {
            s.res_retired.push(old);
        }
    });
    Ok((buf, va, cap))
}

/// The `io` block of a two-pass stage: its user SGPRs (`compute_pass`: the MS layout) with the
/// argument table at `table_at`, and the high dword of unified memory's base, as the other binds
/// write it (the translated pair picks each access's memory window from it).
/// LOCAL: with the heaps' addresses after the arguments (#1803's `with_heaps`) when the layout
/// says its shaders index the heaps themselves (veyn-sudo's measurement on #724).
fn two_pass_io(layout: Option<&Layout>, args: &[u8], table_at: u64, compute_pass: bool, uma_base: usize) -> Vec<u32> {
    two_pass_io_heaps(layout, args, table_at, compute_pass, uma_base, [0; 2])
}

/// LOCAL (agent U): `XCL_LOCAL_MS_HEAPS` (see `two_pass_io_heaps`), logged once with `XCL_LOCAL_MS_FIRST`.
fn local_ms_heaps_mode() -> u32 {
    static M: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *M.get_or_init(|| {
        let m = match std::env::var("XCL_LOCAL_MS_HEAPS").ok().as_deref() {
            Some("0") => 0,
            Some("2") => 2,
            _ => 1,
        };
        crate::xlog_info!("LOCAL agentU: mesh compute pass root arguments from s{}, heap bases mode {} (XCL_LOCAL_MS_FIRST / XCL_LOCAL_MS_HEAPS)", console_list::first_sgpr_ms(), m);
        m
    })
}

fn two_pass_io_heaps(layout: Option<&Layout>, args: &[u8], table_at: u64, compute_pass: bool, uma_base: usize, heaps: [u32; 2]) -> Vec<u32> {
    let mut io = match layout {
        // LOCAL (agent U): `XCL_LOCAL_MS_HEAPS=0|1|2` for the mesh compute pass: 0 no heap bases, 1 (default)
        // #1803's rule (after the arguments, from the root signature's flags), 2 both bases whatever the flags
        Some(l) if compute_pass => match local_ms_heaps_mode() {
            0 => console_list::user_data_ms(l, args, table_at),
            2 => {
                let mut v = console_list::user_data_ms(l, args, table_at);
                v.extend(heaps);
                v
            }
            _ => console_list::with_heaps(l, None, console_list::user_data_ms(l, args, table_at), heaps),
        },
        Some(l) => console_list::with_heaps(l, None, console_list::user_data_gfx(l, args, table_at), heaps),
        None if compute_pass => {
            let mut v = vec![0u32; 16];
            v[8] = table_at as u32;
            v[9] = (table_at >> 32) as u32;
            v
        }
        None => vec![table_at as u32, (table_at >> 32) as u32],
    };
    io.resize(crate::gpu::recomp::KERNEL_SGPRS, 0);
    io[crate::gpu::recomp::IO_UMA_BASE_HI] = (uma_base as u64 >> 32) as u32;
    io
}

unsafe fn bind_two_pass_stage(
    this: *mut c_void,
    list: &ID3D12GraphicsCommandList,
    real_mesh_pso: usize,
    tp: &super::real::TwoPassPso,
    res_va: u64,
    compute_pass: bool,
) -> Result<(), String> {
    let u = crate::gpu::uma::get().ok_or("no unified memory")?;
    let windows = super::real::uma_window_vas();
    let first = *windows.first().ok_or("unified memory is not open on the GPU")?;
    let (layout, mut args, heaps) = with(this, |s| (s.layout_gfx.clone(), s.args_gfx.clone(), s.heaps)); // LOCAL: mut (agent Z)
    // LOCAL (agent Z): agent L's root-view copies / scratch and out-of-window scan for the two-pass mesh stages too
    super::rtsum::oow_copy_root_views(layout.as_deref(), &mut args);
    super::rtsum::oow_scratch_root_views(layout.as_deref(), &mut args);
    super::rtsum::oow_scan(with(this, |s| s.pso), !compute_pass, layout.as_deref(), &args);
    let copy = |bytes: &[u8]| -> Result<usize, String> {
        let p = with(this, |s| s.arena.take(u, bytes.len().max(16), 256, &mut s.uma)).ok_or("unified memory is full")?;
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), p, bytes.len());
        Ok(p as usize)
    };
    let inline = layout
        .as_ref()
        .map_or(0, |l| if compute_pass { l.inline_ms } else { l.inline_gfx } as usize)
        .min(args.len());
    let table_at = copy(&args[inline..])?;
    let heaps = if std::env::var_os("XCL_LOCAL_TWO_PASS_NO_HEAPS").is_some() { [0; 2] } else { heaps };
    let io = two_pass_io_heaps(layout.as_deref(), &args, table_at as u64, compute_pass, u.base(), heaps);
    super::lcap::demand_at_bind(with(this, |s| s.pso), compute_pass, &io); // LOCAL (agent R): XCL_LOCAL_OOW_DEMAND
    super::lcap::on_two_pass(this, with(this, |s| s.pso), compute_pass, &io, table_at, &args[inline..], heaps); // LOCAL (agent R)
    // LOCAL (agent N): `XCL_LOCAL_MS_CPU=N`: the first N two-pass compute binds log what the CPU view of
    // unified memory holds now (record time) where the retail MS reads its counts (see `local_ms_probe`)
    // LOCAL (agent U): how many mesh compute binds have the view-heap flag and a nonzero view-heap base
    if compute_pass {
        static B: [AtomicU64; 3] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
        let n = B[0].fetch_add(1, Ordering::Relaxed) + 1;
        if layout.as_ref().is_some_and(|l| l.heaps[0]) {
            B[1].fetch_add(1, Ordering::Relaxed);
        }
        if heaps[0] != 0 {
            B[2].fetch_add(1, Ordering::Relaxed);
        }
        if [1, 10, 100, 1000, 10000, 100000, 1000000].contains(&n) {
            crate::xlog_info!(
                "LOCAL agentU ms binds: {} binds, {} with the view-heap flag, {} with a nonzero view-heap base (passed: mode {})",
                n,
                B[1].load(Ordering::Relaxed),
                B[2].load(Ordering::Relaxed),
                local_ms_heaps_mode()
            );
        }
    }
    if compute_pass {
        static N: AtomicU64 = AtomicU64::new(0);
        static MAX: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
        let max = *MAX.get_or_init(|| std::env::var("XCL_LOCAL_MS_CPU").ok().and_then(|v| v.parse().ok()).unwrap_or(0));
        if max > 0 && N.fetch_add(1, Ordering::Relaxed) < max {
            let rd = |hi: u32, lo: u32| -> Option<u32> {
                let base_hi = (u.base() as u64 >> 32) as u32;
                let k = if hi == 0 { 0 } else { hi.wrapping_sub(base_hi) };
                if k > 3 {
                    return None;
                }
                let p = u.base() + ((k as usize) << 32) + (lo & !3) as usize;
                (u.contains(p) && crate::gpu::shader::readable(p, 4)).then(|| (p as *const u32).read_volatile())
            };
            let t = io[17];
            let v: Vec<Option<u32>> = (0..4).map(|k| rd(0, t + 4 * k)).collect();
            let (bhi, blo) = (v[1].unwrap_or(0) & 0xFFFF, v[0].unwrap_or(0));
            let words: Vec<Option<u32>> = (0..12).map(|k| rd(bhi, blo + 4 * k)).collect();
            let params: Vec<String> = layout
                .as_ref()
                .map(|l| l.params.iter().map(|p| match p { Param::Constants { at, count } => format!("c{at}x{count}"), Param::Descriptor { at } => format!("d{at}"), Param::Table { at } => format!("t{at}") }).collect())
                .unwrap_or_default();
            let aw: Vec<u32> = args.chunks(4).take(16).map(|c| u32::from_le_bytes([c[0], *c.get(1).unwrap_or(&0), *c.get(2).unwrap_or(&0), *c.get(3).unwrap_or(&0)])).collect();
            crate::xlog_info!(
                "ms cpu: s16..s19 {:x?} table {:x?} buffer[0..12] {:x?} | inline_ms {:?} params {:?} args {:x?}",
                &io[16..20],
                v,
                words,
                layout.as_ref().map(|l| l.inline_ms),
                params,
                aw
            );
            // LOCAL (agent U): the heap flags of the layout, the list's heap bases and what s15..s25 hold
            crate::xlog_info!(
                "LOCAL agentU ms sgprs: first s{} mode {} flags {:?} list heaps {:x?} s15..s25 {:x?}",
                console_list::first_sgpr_ms(),
                local_ms_heaps_mode(),
                layout.as_ref().map(|l| l.heaps),
                heaps,
                &io[15..26]
            );
        }
    }
    let io_at = copy(&io.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<u8>>())?;
    let va = |p: usize| super::real::uma_va_of(p).unwrap_or(first);
    let rs_raw = if compute_pass { tp.compute_rs } else { tp.mesh_rs } as *mut c_void;
    let rs = ID3D12RootSignature::from_raw_borrowed(&rs_raw).ok_or("two-pass root signature")?;
    let views = [va(io_at), res_va].into_iter().chain((0..4).map(|k| windows.get(k).copied().unwrap_or(first)));
    if compute_pass {
        list.SetComputeRootSignature(rs);
        for (k, v) in views.enumerate() {
            list.SetComputeRootUnorderedAccessView(k as u32, v);
        }
        let pso_raw = tp.compute as *mut c_void;
        let pc = ID3D12PipelineState::from_raw_borrowed(&pso_raw).ok_or("two-pass compute pipeline")?;
        list.SetPipelineState(pc);
    } else {
        list.SetGraphicsRootSignature(rs);
        for (k, v) in views.enumerate() {
            list.SetGraphicsRootUnorderedAccessView(k as u32, v);
        }
        let pso_raw = real_mesh_pso as *mut c_void;
        let pc = ID3D12PipelineState::from_raw_borrowed(&pso_raw).ok_or("two-pass mesh pipeline")?;
        list.SetPipelineState(pc);
    }
    Ok(())
}

/// The pass that rewrites a two-pass ExecuteIndirect's arguments, on XCL's ABI: u0 = `count`,
/// `stride`, `offset` (of the DispatchMesh argument), `split`, `cap` (groups with a record), and
/// where the commands are (u2) and go (u3), in bytes. Each command goes out twice, for the compute
/// pass and then (at `count · stride`) for the mesh pass, with at most `cap` groups, all in X:
/// both passes find a group's record by `gid.x` alone. With a count buffer (u0 dword 7 set: the
/// count at dword 8's byte offset in u4), the commands past its count get no groups: they are run
/// one by one (`two_pass_steps`), so without one.
pub(super) const SCALE_ARGS_HLSL: &str = r#"RWByteAddressBuffer io : register(u0);
RWByteAddressBuffer src : register(u2);
RWByteAddressBuffer dst : register(u3);
RWByteAddressBuffer cnt : register(u4);
[numthreads(64, 1, 1)]
void main(uint tid : SV_DispatchThreadID) {
    uint count = io.Load(0), stride = io.Load(4), offset = io.Load(8), split = io.Load(12), cap = io.Load(16);
    if (tid >= count) return;
    uint live = io.Load(28) != 0u ? min(count, cnt.Load(io.Load(32))) : count;
    uint s = io.Load(20) + tid * stride, d = io.Load(24) + tid * stride, m = d + count * stride;
    for (uint i = 0u; i < stride; i += 4u) {
        uint w = src.Load(s + i);
        dst.Store(d + i, w);
        dst.Store(m + i, w);
    }
    uint3 g = src.Load3(s + offset);
    uint n = 0u;
    if (tid < live && g.x != 0u && g.y != 0u && g.z != 0u) {
        n = min(g.x, cap);
        n = g.y > cap / n ? cap : n * g.y;
        n = g.z > cap / n ? cap : n * g.z;
    }
    dst.Store3(d + offset, uint3(n, 1u, 1u));
    dst.Store3(m + offset, uint3(n * split, 1u, 1u));
}
"#;

unsafe fn scale_indirect_pipeline(dev: &ID3D12Device) -> Result<(ID3D12RootSignature, ID3D12PipelineState), String> {
    static CACHE: Mutex<Option<HashMap<usize, (ID3D12RootSignature, ID3D12PipelineState)>>> = Mutex::new(None);
    let key = dev.as_raw() as usize;
    if let Some(pair) = CACHE.lock().unwrap().as_ref().and_then(|m| m.get(&key).cloned()) {
        return Ok(pair);
    }
    let dxil = crate::gpu::recomp::dxc::compile_cached(SCALE_ARGS_HLSL, "main", "cs_6_6")?.0;
    let rs = crate::gpu::recomp::pso::abi_root_signature(dev, false)?;
    let mut cdesc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
        pRootSignature: std::mem::ManuallyDrop::new(Some(rs.clone())),
        CS: D3D12_SHADER_BYTECODE { pShaderBytecode: dxil.as_ptr() as *const c_void, BytecodeLength: dxil.len() },
        ..Default::default()
    };
    let res_pso: Result<ID3D12PipelineState, _> = dev.CreateComputePipelineState(&cdesc);
    std::mem::ManuallyDrop::drop(&mut cdesc.pRootSignature);
    let pso = res_pso.map_err(|e| format!("scale pso create: {e}"))?;
    CACHE.lock().unwrap().get_or_insert_with(HashMap::new).insert(key, (rs.clone(), pso.clone()));
    Ok((rs, pso))
}

/// `bytes` of this recording's scratch blocks (COMMON between uses): the block and the offset.
/// Past `SCRATCH_CAP` a recording gets no more.
unsafe fn scratch_args(this: *mut c_void, dev: &ID3D12Device, bytes: u64) -> Result<(ID3D12Resource, u64), String> {
    let bytes = bytes.next_multiple_of(256);
    if let Some(r) = with(this, |s| {
        let (b, cap, _) = s.scratch.as_ref()?;
        (s.scratch_used + bytes <= *cap).then(|| {
            s.scratch_used += bytes;
            (b.clone(), s.scratch_used - bytes)
        })
    }) {
        return Ok(r);
    }
    let size = bytes.max(SCRATCH_BLOCK);
    if with(this, |s| s.scratch_taken) + size > SCRATCH_CAP {
        return Err(format!("its rewritten arguments are past {} MiB this recording", SCRATCH_CAP >> 20));
    }
    let dev_key = dev.as_raw() as usize;
    let picked = {
        let mut free = FREE_RES.lock().unwrap();
        free.iter().position(|(_, cap, d)| *d == dev_key && *cap >= size && *cap <= SCRATCH_CAP).map(|i| free.swap_remove(i))
    };
    let block = match picked {
        Some(e) => e,
        None => (default_uav_buffer(dev, size, "two-pass argument scratch")?, size, dev_key),
    };
    let b = block.0.clone();
    with(this, |s| {
        s.scratch_taken += block.1;
        s.scratch_used = bytes;
        s.scratch = Some(block.clone());
        s.res_retired.push(block);
    });
    Ok((b, 0))
}

/// The arguments of `max` commands at UMA address `src`, rewritten for the two passes
/// (`SCALE_ARGS_HLSL`) into a scratch block: the block (INDIRECT_ARGUMENT until the caller puts
/// it back to COMMON) and where the compute pass's commands are; the mesh pass's follow them.
///
/// They are read through the memory window over them, as translated shaders read memory, not
/// through the title's buffer: its state on the PC is the title's and not known here. So is the
/// count (`count`: its UMA address, when the title gave a count buffer).
#[allow(clippy::too_many_arguments)]
unsafe fn make_scaled_args_buffer(this: *mut c_void, list: &ID3D12GraphicsCommandList, src: usize, count: Option<usize>, max: u32, stride: u32, mesh_offset: u32, split: u32, cap: u32) -> Result<(ID3D12Resource, u64), String> {
    let dev = super::real::list_device(this).ok_or("no device")?;
    let (rs, pso) = scale_indirect_pipeline(&dev)?;
    let u = crate::gpu::uma::get().ok_or("no unified memory")?;
    let commands = (max as u64).saturating_mul(stride as u64);
    super::real::uma_window_buffer(src, commands as usize).ok_or("the indirect arguments are not in one memory window")?;
    let src_va = super::real::uma_va_of(src).ok_or("unified memory is not open on the GPU")?;
    let count_va = match count {
        Some(p) => {
            super::real::uma_window_buffer(p, 4).ok_or("the indirect count is not in a memory window")?;
            Some(super::real::uma_va_of(p).ok_or("unified memory is not open on the GPU")?)
        }
        None => None,
    };
    let (dst, dst_off) = scratch_args(this, &dev, commands.saturating_mul(2))?;
    let consts: Vec<u8> = [max, stride, mesh_offset, split, cap, 0, 0, count_va.is_some() as u32, 0].iter().flat_map(|w| w.to_le_bytes()).collect();
    let p = with(this, |s| s.arena.take(u, consts.len(), 256, &mut s.uma)).ok_or("unified memory is full")?;
    std::ptr::copy_nonoverlapping(consts.as_ptr(), p, consts.len());
    let consts_va = super::real::uma_va_of(p as usize).ok_or("unified memory is not open on the GPU")?;
    // the title's writes of the arguments before this (`fresh_indirect_arguments`)
    emit_uav_barrier(list, None);
    emit_transition(list, &dst, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    list.SetComputeRootSignature(&rs);
    list.SetPipelineState(&pso);
    list.SetComputeRootUnorderedAccessView(0, consts_va);
    list.SetComputeRootUnorderedAccessView(2, src_va);
    list.SetComputeRootUnorderedAccessView(3, dst.GetGPUVirtualAddress() + dst_off);
    if let Some(va) = count_va {
        list.SetComputeRootUnorderedAccessView(4, va);
    }
    list.Dispatch(max.div_ceil(64), 1, 1);
    emit_transition(list, &dst, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
    Ok((dst, dst_off))
}

pub(super) unsafe extern "system" fn dispatch_mesh(this: *mut c_void, x: u32, y: u32, z: u32) {
    type PcDm = unsafe extern "system" fn(*mut c_void, u32, u32, u32);
    let pc_dm: PcDm = std::mem::transmute(super::real::orig_fn(this, 79));
    let raw_pso = with(this, |s| s.pso);
    let pso = super::pso_async::resolve_wait(raw_pso);
    if let Some(tp) = super::real::two_pass_of(pso) {
        let groups = x.saturating_mul(y).saturating_mul(z);
        super::rtsum::direct_mesh(groups as u64); // LOCAL
        if groups == 0 {
            return;
        }
        let Some(list) = ID3D12GraphicsCommandList::from_raw_borrowed(&this) else { return };
        let record_bytes = tp.record.record_dwords() as u64 * 4;
        let need = (groups as u64).saturating_mul(record_bytes);
        let run = (|| -> Result<(), String> {
            let (res_buf, res_va, cap) = ensure_res_buffer(this, need)?;
            let n = groups.min(record_groups(cap, record_bytes, tp.mesh_groups_per_record));
            if n == 0 {
                return Err(format!("a record of {record_bytes} B does not fit the record buffer"));
            }
            if n < groups {
                static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                if N.fetch_add(1, Ordering::Relaxed) < 8 {
                    crate::xlog_warn!("D3D12.X: two-pass DispatchMesh of {groups} groups: {} past the record buffer dropped", groups - n);
                }
            }
            bind_two_pass_stage(this, list, pso, &tp, res_va, true)?;
            list.Dispatch(n, 1, 1);
            emit_uav_barrier(list, Some(&res_buf));
            super::lcap::after_ms_compute(this, res_buf.as_raw() as usize, record_bytes, tp.mesh_groups_per_record, None, n); // LOCAL (agent R)
            local_ms_probe(this, list, &tp, &res_buf, record_bytes); // LOCAL
            super::rtsum::two_pass_headers(this, &res_buf, record_bytes, n.min(4) as u64, format!("direct DispatchMesh {groups} groups, record {record_bytes} B")); // LOCAL
            bind_two_pass_stage(this, list, pso, &tp, res_va, false)?;
            pc_dm(this, n * tp.mesh_groups_per_record, 1, 1);
            emit_uav_barrier(list, Some(&res_buf));
            Ok(())
        })();
        match run {
            Ok(()) => {
                count(DRAWS);
                local_mesh(0, 1);
                super::rtsum::draw(this, super::rtsum::DRAW_TR); // LOCAL
                static LOGGED: AtomicU64 = AtomicU64::new(0);
                let n = LOGGED.fetch_add(1, Ordering::Relaxed) + 1;
                if n <= 8 || n.is_power_of_two() {
                    crate::xlog_info!("D3D12.X: two-pass DispatchMesh #{n} ({groups} groups, split {}, kind {})", tp.mesh_groups_per_record, tp.kind);
                }
            }
            Err(e) => {
                count(DRAWS_SKIPPED);
                local_mesh(1, 1);
                static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                if N.fetch_add(1, Ordering::Relaxed) < 8 {
                    crate::xlog_warn!("D3D12.X: two-pass DispatchMesh skipped: {e}");
                }
            }
        }
        return;
    }
    local_mesh(6, 1);
    if super::pso_async::is_pending(raw_pso) {
        count(DRAWS_SKIPPED);
        return;
    }
    let rs = with(this, |s| s.rs_gfx) as *mut c_void;
    if let (Some(list), Some(r)) = (ID3D12GraphicsCommandList::from_raw_borrowed(&this), ID3D12RootSignature::from_raw_borrowed(&rs)) {
        list.SetGraphicsRootSignature(r);
    }
    count(DRAWS_PC);
    pc_dm(this, x, y, z);
}

/// #59 ExecuteIndirect(signature, max count, arguments, offset, count buffer, offset): with a
/// translated pipeline, on XCL's ABI with the kept root arguments (signatures that change root
/// arguments themselves cannot be – skipped, logged)
pub(super) unsafe extern "system" fn execute_indirect(this: *mut c_void, sig: usize, max: u32, args: usize, args_off: u64, count_buf: usize, count_off: u64) {
    type F = unsafe extern "system" fn(*mut c_void, usize, u32, usize, u64, usize, u64);
    let pc: F = std::mem::transmute(super::real::orig_fn(this, 59));
    let raw_pso = with(this, |s| s.pso);
    let pso = super::pso_async::resolve_wait(raw_pso);
    // where the commands are in unified memory (a console buffer's offsets are the title's)
    let commands_at = super::real::console_buffer(args).map(|(p, n)| (p + args_off as usize, n.saturating_sub(args_off as usize)));
    let count_at = super::real::console_buffer(count_buf).filter(|&(_, n)| count_off as usize + 4 <= n).map(|(p, _)| p + count_off as usize);
    // LOCAL (agent Z, `XCL_LOCAL_EI_NO_DELTA=1`): a buffer placed below its console address gets
    // `buffer_delta` added to offsets; a retail title's ExecuteIndirect passes an offset into such
    // a buffer that equals that delta and lies past the buffer's console size (args 0x87d0 into a
    // 0x780-byte buffer placed 0x87d0 above its 64 KiB start): with the delta added the PC call
    // reads past the resource. With the switch, an offset that only fits WITHOUT the delta is
    // taken as relative to the resource start.
    let local_off = |res: usize, off: u64, need: u64| -> u64 {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let d = super::real::buffer_delta(res);
        if d == 0 || !*ON.get_or_init(|| std::env::var_os("XCL_LOCAL_EI_NO_DELTA").is_some()) {
            return off + d;
        }
        let width = { let raw = res as *mut c_void; ID3D12Resource::from_raw_borrowed(&raw).map_or(0, |r| r.GetDesc().Width) };
        if off + d + need > width && off + need <= width {
            static N: AtomicU64 = AtomicU64::new(0);
            if N.fetch_add(1, Ordering::Relaxed) < 8 {
                crate::xlog_info!("LOCAL ei-no-delta: buffer {res:#x} width {width:#x} offset {off:#x} delta {d:#x}: read at {off:#x}, not {:#x}", off + d);
            }
            return off;
        }
        off + d
    };
    let ei_stride = super::rtsum::stride_of(sig) as u64;
    let (args_off, count_off) = (local_off(args, args_off, (max.max(1) as u64) * ei_stride.max(4)), if count_buf == 0 { count_off } else { local_off(count_buf, count_off, 4) });
    if let Some(tp) = super::real::two_pass_of(pso) {
        if let Some(msig) = dispatch_mesh_sig_of(sig) {
            let Some(list) = ID3D12GraphicsCommandList::from_raw_borrowed(&this) else { return };
            let run = (|| -> Result<(), String> {
                if max == 0 || args == 0 {
                    return Ok(());
                }
                let commands = max as usize * msig.byte_stride as usize;
                let src = match commands_at {
                    Some((p, n)) if commands <= n => p,
                    _ => return Err("its arguments are not in a console buffer in unified memory".into()),
                };
                let record_bytes = tp.record.record_dwords() as u64 * 4;
                let (res_buf, res_va, cap) = ensure_res_buffer(this, 32 << 20)?;
                let cap = record_groups(cap, record_bytes, tp.mesh_groups_per_record);
                if cap == 0 {
                    return Err(format!("a record of {record_bytes} B does not fit the record buffer"));
                }
                let count = match (count_buf, count_at) {
                    (0, _) => None,
                    (_, Some(p)) => Some(p),
                    _ => return Err("its count is not in a console buffer in unified memory".into()),
                };
                // the groups past the record buffer, and the commands past the count, are dropped
                // on the GPU (`SCALE_ARGS_HLSL`)
                let (scaled, at) = make_scaled_args_buffer(this, list, src, count, max, msig.byte_stride, msig.mesh_offset, tp.mesh_groups_per_record, cap)?;
                let passes = (|| -> Result<(), String> {
                    for (k, compute) in two_pass_steps(max) {
                        bind_two_pass_stage(this, list, pso, &tp, res_va, compute)?;
                        let (s, first) = if compute { (msig.companion_dispatch, at) } else { (sig, at + commands as u64) };
                        pc(this, s, 1, scaled.as_raw() as usize, first + k as u64 * msig.byte_stride as u64, 0, 0);
                        emit_uav_barrier(list, Some(&res_buf));
                        if compute {
                            // LOCAL (agent R)
                            super::lcap::after_ms_compute(this, res_buf.as_raw() as usize, record_bytes, tp.mesh_groups_per_record, Some((scaled.as_raw() as usize, at + k as u64 * msig.byte_stride as u64, at + commands as u64 + k as u64 * msig.byte_stride as u64, msig.byte_stride, msig.mesh_offset)), 0);
                        }
                        if compute && k == 0 {
                            local_ms_probe(this, list, &tp, &res_buf, record_bytes); // LOCAL
                            super::rtsum::two_pass_headers(this, &res_buf, record_bytes, 4, format!("ExecuteIndirect(DispatchMesh) command 0 of {max}, record {record_bytes} B")); // LOCAL
                        }
                    }
                    Ok(())
                })();
                emit_transition(list, &scaled, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_COMMON);
                passes
            })();
            match run {
                Ok(()) => {
                    count(DRAWS);
                    local_mesh(2, 1);
                    super::rtsum::indirect(this, sig, "two-pass mesh"); // LOCAL
                    super::rtsum::after_indirect(this, sig, max, args, args_off, count_buf, count_off, true, pso); // LOCAL (the title's mesh arguments)
                    local_mesh(7, max as u64);
                    static LOGGED: AtomicU64 = AtomicU64::new(0);
                    let n = LOGGED.fetch_add(1, Ordering::Relaxed) + 1;
                    if n <= 8 || n.is_power_of_two() {
                        crate::xlog_info!("D3D12.X: two-pass ExecuteIndirect(DispatchMesh) #{n} (max {max}, split {}, kind {})", tp.mesh_groups_per_record, tp.kind);
                    }
                }
                Err(e) => {
                    count(DRAWS_SKIPPED);
                    local_mesh(3, 1);
                    super::rtsum::indirect(this, sig, "two-pass mesh skipped"); // LOCAL
                    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                    if N.fetch_add(1, Ordering::Relaxed) < 8 {
                        crate::xlog_warn!("D3D12.X: two-pass ExecuteIndirect skipped: {e}");
                    }
                }
            }
            return;
        }
        local_mesh(5, 1);
    } else if dispatch_mesh_sig_of(sig).is_some() {
        local_mesh(4, 1);
        if super::real::two_pass_of(pso).is_none() && pso != 0 && !super::pso_async::is_pending(raw_pso) && !super::real::is_translated_graphics(pso) {
            // LOCAL: a DispatchMesh signature with a pipeline that is neither two-pass nor pending
            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if N.fetch_add(1, Ordering::Relaxed) < 4 {
                crate::xlog_warn!("D3D12.X: (LOCAL) ExecuteIndirect(DispatchMesh) with pipeline {pso:#x} that is not a two-pass one – PC call");
            }
        }
    }
    let (gfx, compute) = (super::real::is_translated_graphics(pso), super::real::is_translated_compute(pso));
    if !gfx && !compute {
        // A pipeline still being built (or whose build failed): its stand-in draws nothing, and
        // the list may hold XCL's ABI root signature from a translated draw before it – the PC
        // call with that pair is invalid (debug layer 953, "the currently set Root Signature
        // doesn't match the currently set Pipeline State Object"; a retail title's world then
        // hung the GPU, 0x887a0006). Skipped, like a draw or dispatch of such a pipeline.
        if super::pso_async::is_pending(with(this, |s| s.pso)) {
            count(DRAWS_SKIPPED);
            super::rtsum::indirect(this, sig, "skipped (pending)"); // LOCAL
            super::culprit::skipped(this, || super::culprit::indirect_op(this, sig, max, count_buf != 0));
            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if N.fetch_add(1, Ordering::Relaxed) < 4 {
                crate::xlog_warn!("D3D12.X: ExecuteIndirect with a pipeline that is still being built skipped");
            }
            return;
        }
        // a console compute program without a translation (its build failed): nothing to run,
        // as for Dispatch. The PC call would run it with the list's compute root signature, which
        // can still be XCL's ABI one from a translated dispatch before it (debug layer 953).
        // A native PC compute pipeline is not one of these and still goes to the PC call.
        if signature_dispatches(sig) && super::real::is_console_compute(pso) {
            count(DISPATCHES_SKIPPED);
            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if N.fetch_add(1, Ordering::Relaxed) < 4 {
                crate::xlog_warn!("D3D12.X: ExecuteIndirect dispatch with a pipeline that has no translated console shader ({pso:#x}) skipped");
            }
            return;
        }
        // a PC pipeline: the title's root signature, as for a draw (`graphics_bound`)
        super::rtsum::indirect(this, sig, if with(this, |s| s.pso) == 0 { "PC call, no pipeline" } else { "PC/stand-in pipeline" }); // LOCAL
        let rs = with(this, |s| s.rs_gfx) as *mut c_void;
        if let (Some(list), Some(r)) = (ID3D12GraphicsCommandList::from_raw_borrowed(&this), ID3D12RootSignature::from_raw_borrowed(&rs)) {
            list.SetGraphicsRootSignature(r);
        }
        let mark = super::culprit::begin(this, || super::culprit::indirect_op(this, sig, max, count_buf != 0));
        fresh_indirect_arguments(this, || pc(this, sig, max, args, args_off, count_buf, count_off));
        return super::culprit::end(this, mark);
    }
    if ROOT_SIGNATURES.lock().unwrap().as_ref().is_some_and(|s| s.contains(&sig)) {
        super::rtsum::indirect(this, sig, "skipped (root-changing)"); // LOCAL
        super::culprit::skipped(this, || super::culprit::indirect_op(this, sig, max, count_buf != 0));
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if N.fetch_add(1, Ordering::Relaxed) < 4 {
            crate::xlog_warn!("D3D12.X: ExecuteIndirect that changes root arguments with a translated pipeline skipped");
        }
        return;
    }
    let Some(list) = ID3D12GraphicsCommandList::from_raw_borrowed(&this) else { return };
    if !gfx {
        super::depth_uav::local_before_dispatch(this); // LOCAL agent I
    }
    match bind_abi(this, list, pso, gfx) {
        Ok(()) => {
            count(if gfx { DRAWS } else { DISPATCHES });
            super::rtsum::indirect(this, sig, if gfx { "translated gfx" } else { "translated compute" }); // LOCAL
            // its draws or groups are in the argument buffer (`max` records)
            let mark = super::culprit::begin(this, || super::culprit::indirect_op(this, sig, max, count_buf != 0));
            fresh_indirect_arguments(this, || pc(this, sig, max, args, args_off, count_buf, count_off));
            super::rtsum::after_indirect(this, sig, max, args, args_off, count_buf, count_off, gfx, with(this, |s| s.pso)); // LOCAL
            if gfx { super::lcap::after_ei(this, args, args_off, max, super::rtsum::stride_of(sig), count_buf, count_off) } else { super::lcap::after_dispatch(this) } // LOCAL (agent H)
            if !gfx {
                super::depth_uav::local_after_dispatch(this, raw_pso); // LOCAL agent I (the title's handle)
            }
            super::culprit::end(this, mark)
        }
        Err(e) => {
            count(if gfx { DRAWS_SKIPPED } else { DISPATCHES_SKIPPED });
            super::rtsum::indirect(this, sig, "skipped (bind failed)"); // LOCAL
            super::culprit::skipped(this, || super::culprit::indirect_op(this, sig, max, count_buf != 0));
            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if N.fetch_add(1, Ordering::Relaxed) < 4 {
                crate::xlog_warn!("D3D12.X: ExecuteIndirect with a translated pipeline skipped: {e}");
            }
        }
    }
}

/// XCL's own buffer for `fresh_indirect_arguments` (device, buffer): COMMON between its uses
static ARGUMENT_ORDER: Mutex<Option<(usize, ID3D12Resource)>> = Mutex::new(None);

/// `op` (an ExecuteIndirect on `list`) reading the arguments and count the list wrote before it.
///
/// The console orders a dispatch that writes indirect arguments before the ExecuteIndirect that
/// reads them with a barrier that reaches the PC as a UAV barrier (`list_x::resource_barrier_x`).
/// NVIDIA's driver (RTX 5090) fetches an ExecuteIndirect's arguments without waiting for that
/// barrier: it ran the count that was there before the list in all but a few of 100 tries, also
/// when the dispatch wrote them through another resource over the same memory, as translated
/// shaders do through the memory window buffers (`execute_indirect_reads_the_arguments_written_before_it`).
/// A retail title's ExecuteIndirect after the dispatch that writes its group count then runs
/// whatever count the memory held before. A transition to INDIRECT_ARGUMENT makes the driver
/// wait for the writes – of any buffer, as measured there. The title's buffer is in the title's
/// state (its transitions reach the PC list), so a buffer of XCL's own takes it, COMMON ->
/// INDIRECT_ARGUMENT before and back after. Only on a list XCL made (`list_x::on_create`) and
/// not a bundle, which cannot hold barriers; its type comes from XCL's state, not a call to it.
unsafe fn fresh_indirect_arguments(list: *mut c_void, op: impl FnOnce()) {
    let tracked = super::list_x::tracked_list_type(list).is_some_and(|t| t != D3D12_COMMAND_LIST_TYPE_BUNDLE);
    let buffer = if tracked { ID3D12GraphicsCommandList::from_raw_borrowed(&list).and_then(|l| argument_order_buffer(l)) } else { None };
    let Some(buffer) = buffer else { return op() };
    let b = buffer.as_raw() as usize;
    super::list_x::transition(list, b, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
    op();
    super::list_x::transition(list, b, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_COMMON);
}

/// The device's buffer for `fresh_indirect_arguments` (made on first use)
unsafe fn argument_order_buffer(list: &ID3D12GraphicsCommandList) -> Option<ID3D12Resource> {
    let mut dev: Option<ID3D12Device> = None;
    list.GetDevice(&mut dev).ok()?;
    let dev = dev?;
    let mut g = ARGUMENT_ORDER.lock().unwrap();
    if let Some((_, b)) = g.as_ref().filter(|(d, _)| *d == dev.as_raw() as usize) {
        return Some(b.clone());
    }
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: 256,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    let mut b: Option<ID3D12Resource> = None;
    if let Err(e) = dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut b) {
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if N.fetch_add(1, Ordering::Relaxed) < 4 {
            crate::xlog_warn!("D3D12.X: ExecuteIndirect: no buffer to order its arguments after the writes before it ({e}) – it may read old ones");
        }
        return None;
    }
    let b = b?;
    *g = Some((dev.as_raw() as usize, b.clone()));
    Some(b)
}

/// A DrawInstanced of the console's rect list (three corners of each rectangle, the fourth
/// implied) as two triangles a rectangle; false: not a rect list, or one this cannot expand
/// (the caller draws the three vertices as a triangle – half of each rectangle).
///
/// The PC has no rect lists. The fourth corner is made here, on the CPU, from the title's
/// vertex data: v3 = v0 + v2 − v1 (seen in a retail title's full-screen passes: (−1,1) (−1,−1)
/// (1,−1) with texture coordinates (0,0) (0,1) (1,1) – the missing corner is (1,1), (1,0)).
/// The vertex layout is not looked at: each 4-byte word that is the same in all three vertices
/// is kept (a packed colour), any other is taken as a float. Vertices a shader makes from the
/// vertex index alone (no vertex buffer) are not expanded.
unsafe fn draw_rects(this: *mut c_void, list: &ID3D12GraphicsCommandList, vertices: u32, instances: u32, first_vertex: u32, first_instance: u32) -> bool {
    if !cpu_rect_expansion(with(this, |s| s.pso)) {
        return false;
    }
    let Some(vbs) = super::list_x::rect_list_state(this) else { return false };
    let rects = (vertices / 3) as usize;
    if vbs.is_empty() || rects == 0 || rects > 4096 {
        return false;
    }
    let mut views = Vec::new();
    for &(slot, (va, bytes, stride), _) in &vbs {
        let (s, at) = (stride as usize, va as usize + first_vertex as usize * stride as usize);
        let len = rects * 3 * s;
        if va == 0 || s == 0 || s % 4 != 0 || first_vertex as usize * s + len > bytes as usize || !crate::gpu::shader::readable(at, len) {
            return false;
        }
        let p = allocate_upload_memory(this, rects * 6 * s, 16);
        if p.is_null() {
            return false;
        }
        let word = |v: usize, w: usize| (at as *const u8).add(v * s + w * 4).cast::<u32>().read_unaligned();
        for r in 0..rects {
            let out = p.add(r * 6 * s);
            let src = (at as *const u8).add(r * 3 * s);
            // v0 v1 v2, then v0 v2 v3
            std::ptr::copy_nonoverlapping(src, out, 3 * s);
            std::ptr::copy_nonoverlapping(src, out.add(3 * s), s);
            std::ptr::copy_nonoverlapping(src.add(2 * s), out.add(4 * s), s);
            for w in 0..s / 4 {
                let (a, b, c) = (word(r * 3, w), word(r * 3 + 1, w), word(r * 3 + 2, w));
                let d = if a == b && b == c { a } else { (f32::from_bits(a) + f32::from_bits(c) - f32::from_bits(b)).to_bits() };
                out.add(5 * s + w * 4).cast::<u32>().write_unaligned(d);
            }
        }
        views.push((slot, D3D12_VERTEX_BUFFER_VIEW { BufferLocation: super::list_x::pc_va_of(p as u64), SizeInBytes: (rects * 6 * s) as u32, StrideInBytes: stride }));
    }
    for (slot, v) in &views {
        list.IASetVertexBuffers(*slot, Some(&[*v]));
    }
    list.DrawInstanced(rects as u32 * 6, instances, 0, first_instance);
    for &(slot, _, pc) in &vbs {
        if let Some(v) = pc {
            list.IASetVertexBuffers(slot, Some(&[v]));
        }
    }
    static N: AtomicU64 = AtomicU64::new(0);
    if N.fetch_add(1, Ordering::Relaxed) == 0 {
        crate::xlog_info!("D3D12.X: rect lists are drawn as two triangles a rectangle, the fourth corner made on the CPU");
    }
    true
}

/// Whether a rect list drawn with the bound pipeline `pso` is expanded on the CPU (`draw_rects`):
/// not with the rect list GS (`real::has_rect_list_gs`), which completes each rect itself – both
/// would draw every rect twice (each CPU half is three corners of the rect).
fn cpu_rect_expansion(pso: usize) -> bool {
    !super::real::has_rect_list_gs(super::pso_async::resolve(pso))
}

/// #12 DrawInstanced
pub(super) unsafe extern "system" fn draw_instanced(this: *mut c_void, vertices: u32, instances: u32, first_vertex: u32, first_instance: u32) {
    DRAW_ARGS.with(|d| d.set((false, vertices, instances, first_vertex, 0)));
    if let Some(list) = graphics_bound(this) {
        if draw_rects(this, &list, vertices, instances, first_vertex, first_instance) {
            return;
        }
        super::list_x::sync_title_buffers(this, None, first_vertex, vertices, if first_instance != 0 { instances.max(2) } else { instances });
        list.DrawInstanced(vertices, instances, first_vertex, first_instance);
    }
}

thread_local! {
    /// the draw being bound: (indexed, count, instances, first index / vertex, base vertex) – for the frame trace
    static DRAW_ARGS: std::cell::Cell<(bool, u32, u32, u32, i32)> = const { std::cell::Cell::new((false, 0, 0, 0, 0)) };
}

/// #13 DrawIndexedInstanced
pub(super) unsafe extern "system" fn draw_indexed_instanced(this: *mut c_void, indices: u32, instances: u32, first_index: u32, base_vertex: i32, first_instance: u32) {
    if let Some(list) = prepare_indexed_draw(this, indices, instances, first_index, base_vertex) {
        super::list_x::sync_title_buffers(this, Some((first_index, indices, base_vertex)), 0, 0, if first_instance != 0 { instances.max(2) } else { instances });
        emit_prepared_indexed_draw(&list, indices, instances, first_index, base_vertex, first_instance);
    }
}

/// Prepare an indexed draw once, before callers make temporary native state changes.
pub(super) unsafe fn prepare_indexed_draw(
    this: *mut c_void,
    indices: u32,
    instances: u32,
    first_index: u32,
    base_vertex: i32,
) -> Option<ID3D12GraphicsCommandList> {
    DRAW_ARGS.with(|d| d.set((true, indices, instances, first_index, base_vertex)));
    graphics_bound(this)
}

/// Submit an indexed draw whose translated graphics state has already been prepared.
pub(super) unsafe fn emit_prepared_indexed_draw(
    list: &ID3D12GraphicsCommandList,
    indices: u32,
    instances: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
) {
    list.DrawIndexedInstanced(indices, instances, first_index, base_vertex, first_instance);
}

/// Before a draw: with a translated pipeline its ABI root signature and the kept graphics
/// arguments are bound (None: the draw is skipped – a console pipeline without a
/// translation, or no unified memory).
unsafe fn graphics_bound(this: *mut c_void) -> Option<ID3D12GraphicsCommandList> {
    let list = ID3D12GraphicsCommandList::from_raw_borrowed(&this)?.clone();
    let pso = with(this, |s| s.pso);
    // a pipeline still being built in the background is waited for (its stand-in draws nothing)
    if !super::real::is_translated_graphics(super::pso_async::resolve_wait(pso)) {
        count(DRAWS_PC);
        super::rtsum::draw(this, super::rtsum::DRAW_PC); // LOCAL
        if super::present::trace_frame() {
            // which pipelines draw without a translation (first 12 distinct)
            static SEEN: Mutex<Vec<usize>> = Mutex::new(Vec::new());
            let mut g = SEEN.lock().unwrap();
            if !g.contains(&pso) && g.len() < 12 {
                g.push(pso);
                let real = super::pso_async::resolve_wait(pso);
                crate::xlog_info!(
                    "frame trace: draw with an untranslated pipeline {pso:#x} (resolves to {real:#x}; pending {}, two-pass {}, RT0 {:?})",
                    super::pso_async::is_pending(pso),
                    super::real::two_pass_of(real).is_some(),
                    super::present::rt0_resource(this as usize).map(|r| ID3D12Resource::from_raw_borrowed(&(r as *mut c_void)).map(|x| { let d = x.GetDesc(); (d.Width, d.Height, d.Format.0) }))
                );
            }
        }
        // a PC pipeline (also a stand-in or fallback: made with the title's root signature):
        // the title's root signature, which nothing else binds on the PC list – a draw without
        // one is invalid (the debug layer reports it on every GPU; WARP faults on it)
        let rs = with(this, |s| s.rs_gfx) as *mut c_void;
        if let Some(r) = ID3D12RootSignature::from_raw_borrowed(&rs) {
            list.SetGraphicsRootSignature(r);
        }
        return Some(list);
    }
    if super::present::trace_frame() {
        trace_ui_draw(this, pso);
    }
    match bind_abi(this, &list, pso, true) {
        Ok(()) => {
            count(DRAWS);
            super::rtsum::draw(this, super::rtsum::DRAW_TR); // LOCAL
            super::rtsum::direct_draw(this, pso, DRAW_ARGS.with(|d| d.get())); // LOCAL
            Some(list)
        }
        Err(e) => {
            count(DRAWS_SKIPPED);
            super::rtsum::draw(this, super::rtsum::DRAW_SKIP); // LOCAL
            let (_, n, instances, _, _) = DRAW_ARGS.with(|d| d.get());
            super::culprit::skipped(this, || super::culprit::draw_op("draw", this, n, instances));
            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 4 {
                crate::xlog_warn!("D3D12.X: draw with a translated pipeline skipped: {e}");
            }
            None
        }
    }
}

/// `XCL_TRACE_FRAME`: draws into a presented buffer (the UI) once the title has presented for a
/// while – their pipeline, root arguments and the resources their descriptor tables start with
/// (format, size, kept in step with title memory or not). The first 24.
/// `XCL_TRACE_FRAME`, after 300 presents: one line per distinct (VS, PS, render target) drawn –
/// the shaders' dump hashes, the target and the textures of the first descriptor table.
unsafe fn trace_draw_kinds(this: *mut c_void, pso: usize) {
    // keyed by the target's shape, not the object: a UI library's transient targets (Renoir's
    // layers) are new objects every frame and would fill the list at once
    static SEEN: Mutex<Vec<(u64, u64, Option<(u64, u32, i32, u32)>)>> = Mutex::new(Vec::new());
    if super::present::activity().0 < 300 {
        return;
    }
    let Some((vs, ps)) = super::real::pso_shaders(super::pso_async::resolve(pso)) else { return };
    let rt = super::present::rt0_resource(this as usize).unwrap_or(0);
    let desc = |r: usize| ID3D12Resource::from_raw_borrowed(&(r as *mut c_void)).map(|x| { let d = x.GetDesc(); (d.Width, d.Height, d.Format.0, d.SampleDesc.Count) });
    {
        let mut g = SEEN.lock().unwrap();
        let key = (vs, ps, desc(rt));
        if g.len() >= 160 || g.contains(&key) {
            return;
        }
        g.push(key);
    }
    let (layout, args) = with(this, |s| (s.layout_gfx.clone(), s.args_gfx.clone()));
    let word = |at: usize| args.get(at..at + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).unwrap_or(0);
    let mut tex = Vec::new();
    if let (Some(layout), Some(u)) = (layout, crate::gpu::uma::get()) {
        for p in layout.params.iter() {
            if let Param::Table { at } = *p {
                let t = word(at as usize) as usize;
                for k in 0..4usize {
                    if let Some(r) = super::console_heap::resource_at(u.base() + t + 32 * k) {
                        tex.push(format!("{:?}{}", desc(r), super::title_textures::address(r).map_or(String::new(), |a| format!(" at {a:#x}"))));
                    }
                }
            }
            if tex.len() >= 4 {
                break;
            }
        }
    }
    crate::xlog_info!(
        "frame trace: draws vs {vs:016x} ps {ps:016x} into {:?}{} – textures {}; {}",
        desc(rt),
        if super::present::is_presented(rt) { " (presented)" } else { "" },
        tex.join(", "),
        super::list_x::state_summary(this)
    );
}

/// `XCL_TRACE_SHADERS=hash,…` (dump hashes, hex): the draws traced with their root arguments
/// are those whose vertex or pixel shader is listed – into any target, at any time (the first
/// 24); without it, draws into a presented target (or any – `XCL_TRACE_UI_ANY`) after 300
/// presents.
fn traced_shaders() -> &'static [u64] {
    static S: std::sync::OnceLock<Vec<u64>> = std::sync::OnceLock::new();
    S.get_or_init(|| std::env::var("XCL_TRACE_SHADERS").unwrap_or_default().split(',').filter_map(|h| u64::from_str_radix(h.trim(), 16).ok()).collect())
}

/// `XCL_TRACE_FRAME` with a watched target (`present::watch_rt`): draws whose descriptor
/// tables name it – that object, or another at its title address (the first 16).
unsafe fn trace_reads_of_watched(this: *mut c_void, pso: usize) {
    let Some(w) = super::present::watched_rt() else { return };
    let w_addr = super::real::title_address(w).or_else(|| super::title_textures::address(w));
    let Some(u) = crate::gpu::uma::get() else { return };
    let (layout, args) = with(this, |s| (s.layout_gfx.clone(), s.args_gfx.clone()));
    let Some(layout) = layout else { return };
    let word = |at: usize| args.get(at..at + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).unwrap_or(0);
    for p in layout.params.iter() {
        let Param::Table { at } = *p else { continue };
        let t = word(at as usize) as usize;
        for k in 0..16usize {
            let Some(r) = super::console_heap::resource_at(u.base() + t + 32 * k) else { continue };
            let addr = super::real::title_address(r).or_else(|| super::title_textures::address(r));
            if super::present::is_watched(r) || (addr.is_some() && addr == w_addr) {
                let rt = super::present::rt0_resource(this as usize).unwrap_or(0);
                // each (source, target, shaders) once
                static SEEN: Mutex<Vec<(usize, usize, usize)>> = Mutex::new(Vec::new());
                {
                    let mut g = SEEN.lock().unwrap();
                    if g.len() >= 32 || g.contains(&(r, rt, pso)) {
                        return;
                    }
                    g.push((r, rt, pso));
                }
                if !super::present::is_presented(rt) {
                    super::present::follow(rt);
                }
                if let Some(dsv) = super::list_x::bound_dsv(this) {
                    super::list_x::watch_dsv(dsv);
                }
                let d = |r: usize| ID3D12Resource::from_raw_borrowed(&(r as *mut c_void)).map(|x| { let d = x.GetDesc(); (d.Width, d.Height, d.Format.0, d.Flags.0) });
                crate::xlog_info!(
                    "frame trace: a draw (shaders {:x?}, {}) reads a watched target {r:#x} {:?} at {addr:x?} (slot {k}), drawing into {rt:#x} {:?}{} – {}",
                    super::real::pso_shaders(super::pso_async::resolve(pso)),
                    super::list_x::state_summary(this),
                    d(r),
                    d(rt),
                    if super::present::is_presented(rt) { " (presented)" } else { "" },
                    root_args_summary(this, 32)
                );
                let args = DRAW_ARGS.with(|d| d.get());
                crate::xlog_info!(
                    "frame trace: … that draw: {args:?} (indexed, count, instances, first, base vertex); input layout [{}]; {}",
                    super::real::pso_layout(super::pso_async::resolve(pso)).unwrap_or_default(),
                    super::list_x::geometry_summary(this, args)
                );
                return;
            }
        }
    }
}

/// `XCL_TRACE_FRAME`: render targets a draw into the watched target samples (a UI library's
/// layers) are dumped with it (`present::follow`).
unsafe fn follow_layer_sources(this: *mut c_void) {
    let (Some(w), Some(rt)) = (super::present::watched_rt(), super::present::rt0_resource(this as usize)) else { return };
    if rt != w {
        return;
    }
    let Some(u) = crate::gpu::uma::get() else { return };
    let (layout, args) = with(this, |s| (s.layout_gfx.clone(), s.args_gfx.clone()));
    let Some(layout) = layout else { return };
    let word = |at: usize| args.get(at..at + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).unwrap_or(0);
    for p in layout.params.iter() {
        let Param::Table { at } = *p else { continue };
        let t = word(at as usize) as usize;
        for k in 0..4usize {
            let Some(r) = super::console_heap::resource_at(u.base() + t + 32 * k) else { continue };
            let layer = ID3D12Resource::from_raw_borrowed(&(r as *mut c_void)).is_some_and(|x| {
                let d = x.GetDesc();
                d.Flags.0 & 1 != 0 && d.Width >= 64 && d.Height >= 32 && d.SampleDesc.Count == 1
            });
            if layer && r != w {
                super::present::follow(r);
            }
        }
    }
}

unsafe fn trace_ui_draw(this: *mut c_void, pso: usize) {
    trace_draw_kinds(this, pso);
    trace_reads_of_watched(this, pso);
    follow_layer_sources(this);
    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    // `XCL_TRACE_MANY=n`: n draws instead of 24 (a whole frame of a UI)
    static CAP: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    let cap = *CAP.get_or_init(|| std::env::var("XCL_TRACE_MANY").ok().and_then(|v| v.parse().ok()).unwrap_or(24));
    if N.load(std::sync::atomic::Ordering::Relaxed) >= cap {
        return;
    }
    let Some(rt) = super::present::rt0_resource(this as usize) else { return };
    let wanted = traced_shaders();
    if wanted.is_empty() {
        // draws into a presented target, or – `XCL_TRACE_UI_ANY` – into any target
        if super::present::activity().0 < 300 || (!super::present::is_presented(rt) && std::env::var_os("XCL_TRACE_UI_ANY").is_none()) {
            return;
        }
    } else if !super::real::pso_shaders(super::pso_async::resolve(pso)).is_some_and(|(v, p)| wanted.contains(&v) || wanted.contains(&p)) {
        return;
    } else {
        super::present::watch_rt(rt);
    }
    N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let sh = super::real::pso_shaders(super::pso_async::resolve(pso)).map(|(v, p)| format!("vs {v:016x} ps {p:016x}"));
    crate::xlog_info!("frame trace: UI draw (pipeline {pso:#x}, {}): {}", sh.unwrap_or_default(), root_args_summary(this, 16));
    if !wanted.is_empty() {
        // the picked shaders' geometry too (vertices dumped at the next present)
        let args = DRAW_ARGS.with(|d| d.get());
        crate::xlog_info!(
            "frame trace: … that draw: {args:?} (indexed, count, instances, first, base vertex); input layout [{}]; {}",
            super::real::pso_layout(super::pso_async::resolve(pso)).unwrap_or_default(),
            super::list_x::geometry_summary(this, args)
        );
    }
}

/// A draw's root arguments for the frame trace: constants, root buffers (their first words, as
/// the CPU sees them), the first slots of descriptor tables (the PC resources they name).
unsafe fn root_args_summary(this: *mut c_void, buffer_words: usize) -> String {
    let Some(u) = crate::gpu::uma::get() else { return String::new() };
    let (layout, args) = with(this, |s| (s.layout_gfx.clone(), s.args_gfx.clone()));
    let Some(layout) = layout else { return String::new() };
    let word = |at: usize| args.get(at..at + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).unwrap_or(0);
    let mut what = Vec::new();
    for (i, p) in layout.params.iter().enumerate() {
        match *p {
            Param::Constants { at, count } => what.push(format!("p{i} constants {:x?}", (0..count.min(8) as usize).map(|k| word(at as usize + 4 * k)).collect::<Vec<_>>())),
            Param::Descriptor { at } => {
                let va = (word(at as usize) as u64 | (word(at as usize + 4) as u64) << 32) as usize;
                let w: Vec<u32> = if va != 0 && crate::gpu::shader::readable(va, 4 * buffer_words) { (0..buffer_words).map(|k| ((va + 4 * k) as *const u32).read_unaligned()).collect() } else { Vec::new() };
                what.push(format!("p{i} buffer {va:#x} {w:x?}"));
            }
            Param::Table { at } => {
                let t = word(at as usize) as usize;
                let slots: Vec<String> = (0..4usize)
                    .map(|k| {
                        let at = u.base() + t + 32 * k;
                        let w = super::console_heap::words_at((t + 32 * k) as u32);
                        match super::console_heap::resource_at(at) {
                            Some(r) => {
                                if super::title_textures::address(r).is_some() {
                                    super::title_textures::probe_gpu_content(r);
                                }
                                let d = ID3D12Resource::from_raw_borrowed(&(r as *mut c_void)).map(|x| x.GetDesc());
                                format!("[{k}] {r:#x} {:?} title {:?} T# {:x?}", d.map(|d| (d.Dimension.0, d.Width, d.Height, d.Format.0)), super::title_textures::address(r).map(|a| format!("{a:#x}")), w.map(|w| w[..4].to_vec()))
                            }
                            None => format!("[{k}] - T# {:x?}", w.map(|w| w[..4].to_vec())),
                        }
                    })
                    .collect();
                what.push(format!("p{i} table {t:#x}: {}", slots.join(", ")));
            }
        }
    }
    what.join("; ")
}

/// #33 SetComputeRootSignature (PC #29)
pub(super) unsafe extern "system" fn set_compute_root_signature(this: *mut c_void, rs: *mut c_void) {
    let layout = console_list::layout_of(rs as usize);
    with(this, |s| {
        s.args = vec![0u8; layout.as_ref().map_or(0, |l| l.size as usize)];
        s.layout = layout;
        s.rs_cs = rs as usize;
    });
}

/// #35 SetComputeRootDescriptorTable (PC #31): the table's console address, 32 bits
pub(super) unsafe extern "system" fn set_compute_root_table(this: *mut c_void, i: u32, handle: u64) {
    put(this, i, 0, &(handle as u32).to_le_bytes());
}

/// #37 SetComputeRoot32BitConstant (PC #33)
pub(super) unsafe extern "system" fn set_compute_root_constant(this: *mut c_void, i: u32, value: u32, offset: u32) {
    put(this, i, offset, &value.to_le_bytes());
}

/// #39 SetComputeRoot32BitConstants (PC #35)
pub(super) unsafe extern "system" fn set_compute_root_constants(this: *mut c_void, i: u32, n: u32, data: *const u32, offset: u32) {
    if !data.is_null() {
        let bytes: Vec<u8> = std::slice::from_raw_parts(data, n as usize).iter().flat_map(|w| w.to_le_bytes()).collect();
        put(this, i, offset, &bytes);
    }
}

/// #41 / #43 / #45 SetComputeRoot{ConstantBuffer,ShaderResource,UnorderedAccess}View (PC #37 /
/// #39 / #41): the 64-bit GPU address
pub(super) unsafe extern "system" fn set_compute_root_descriptor(this: *mut c_void, i: u32, va: u64) {
    note_address(va);
    put(this, i, 0, &va.to_le_bytes());
}

/// #25 SetPipelineState: noted, then bound as on the PC
pub(super) unsafe extern "system" fn set_pipeline_state(this: *mut c_void, pso: *mut c_void) {
    with(this, |s| s.pso = pso as usize);
    super::real::bind_pipeline(this, pso);
    super::depth_uav::local_raster_check(this, pso as usize); // LOCAL agent I
}

/// #14 Dispatch: with a translated pipeline, on XCL's ABI with the kept root arguments
pub(super) unsafe extern "system" fn dispatch(this: *mut c_void, x: u32, y: u32, z: u32) {
    let pso = super::pso_async::resolve_wait(with(this, |s| s.pso));
    let Some(list) = ID3D12GraphicsCommandList::from_raw_borrowed(&this) else { return };
    if !super::real::is_translated_compute(pso) {
        // a console pipeline without a translation (including a background build that
        // failed): nothing to run (its root arguments were not given to the PC list either)
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 4 {
            crate::xlog_warn!("D3D12.X: Dispatch with a pipeline that has no translated console shader ({pso:#x}) skipped");
        }
        count(DISPATCHES_SKIPPED);
        super::rtsum::count(super::rtsum::DISPATCH_SKIPPED); // LOCAL
        super::culprit::skipped(this, || super::culprit::compute_op("Dispatch", with(this, |s| s.pso), pso, [x, y, z]));
        return;
    }
    super::depth_uav::local_before_dispatch(this); // LOCAL agent I
    match translated_dispatch(this, list, pso, x, y, z) {
        Ok(()) => {
            super::depth_uav::local_after_dispatch(this, with(this, |s| s.pso)); // LOCAL agent I (the title's handle)
            count(DISPATCHES);
            super::rtsum::count(super::rtsum::DISPATCH_TR); // LOCAL
        }
        Err(e) => {
            count(DISPATCHES_SKIPPED);
            super::rtsum::count(super::rtsum::DISPATCH_SKIPPED); // LOCAL
            super::culprit::skipped(this, || super::culprit::compute_op("Dispatch", with(this, |s| s.pso), pso, [x, y, z]));
            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 4 {
                crate::xlog_warn!("D3D12.X: Dispatch with a translated pipeline skipped: {e}");
            }
        }
    }
}

/// `XCL_TRACE_FRAME`: a full-screen dispatch (enough groups to cover a 4K target in tiles of up
/// to 128 pixels) – what its root arguments say. A bindless title passes the indices of its
/// descriptors in root constants and constant buffers: every small value there is looked up
/// in the dispatch's descriptor tables, and the resources found (the pass's inputs and
/// outputs) are logged.
unsafe fn trace_fullscreen(this: *mut c_void, pso: usize, groups: [u32; 3]) {
    if groups[0] < 30 || groups[1] < 17 {
        return;
    }
    let Some(u) = crate::gpu::uma::get() else { return };
    let (layout, args) = with(this, |s| (s.layout.clone(), s.args.clone()));
    let Some(layout) = layout else { return };
    static SEEN: Mutex<Vec<usize>> = Mutex::new(Vec::new());
    {
        let mut g = SEEN.lock().unwrap();
        if g.contains(&pso) || g.len() >= 12 {
            return;
        }
        g.push(pso);
    }
    let word = |at: usize| args.get(at..at + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap()));
    let mut tables: Vec<u32> = Vec::new();
    let mut values: Vec<u32> = Vec::new();
    let mut what: Vec<String> = Vec::new();
    for (i, p) in layout.params.iter().enumerate() {
        match *p {
            Param::Constants { at, count } => {
                let w: Vec<u32> = (0..count as usize).filter_map(|k| word(at as usize + 4 * k)).collect();
                what.push(format!("p{i} constants {w:x?}"));
                values.extend(&w);
            }
            Param::Descriptor { at } => {
                let lo = word(at as usize).unwrap_or(0) as u64;
                let hi = word(at as usize + 4).unwrap_or(0) as u64;
                let va = (lo | hi << 32) as usize;
                let n = 64usize;
                if va != 0 && crate::gpu::shader::readable(va, 4 * n) {
                    let w: Vec<u32> = (0..n).map(|k| ((va + 4 * k) as *const u32).read_unaligned()).collect();
                    what.push(format!("p{i} buffer {va:#x}: {w:x?}"));
                    values.extend(&w);
                } else {
                    what.push(format!("p{i} buffer {va:#x} (not readable)"));
                }
            }
            Param::Table { at } => {
                let v = word(at as usize).unwrap_or(0);
                what.push(format!("p{i} table {v:#x}"));
                tables.push(v);
            }
        }
    }
    let mut found: Vec<(u32, usize)> = Vec::new();
    for &v in &values {
        if v >= 1 << 20 {
            continue;
        }
        for &t in &tables {
            let slot = u.base() + t as usize + 32 * v as usize;
            if let Some(r) = super::console_heap::resource_at(slot) {
                if !found.iter().any(|x| x.1 == r) {
                    found.push((v, r));
                }
            }
        }
    }
    let res: Vec<String> = found
        .iter()
        .map(|&(v, r)| {
            let d = ID3D12Resource::from_raw_borrowed(&(r as *mut c_void)).map(|x| x.GetDesc());
            format!("[{v}] {r:#x} {:?}{}", d.map(|d| (d.Dimension.0, d.Width, d.Height, d.Format.0)), if super::present::is_presented(r) { " PRESENTED" } else { "" })
        })
        .collect();
    crate::xlog_info!("frame trace: full-screen dispatch {groups:?}, pipeline {pso:#x}: {}", what.join("; "));
    crate::xlog_info!("frame trace: … its indices name: {}", if res.is_empty() { "nothing".to_string() } else { res.join("; ") });
}

/// LOCAL (agent Z, `XCL_LOCAL_FSTRACE`): a translated direct dispatch of at least 30x17 groups in
/// a traced frame: its shader, groups, root constants, root buffers (inside a window or not), and
/// the resources its tables name – slots 0..15 of each table and every small value of its
/// constants / root buffers used as an index into each table – with their title addresses and
/// whether they are render targets of this frame. Textures are dumped at the next present.
unsafe fn local_fs_trace(this: *mut c_void, groups: [u32; 3]) {
    if groups[0] < 30 || groups[1] < 17 {
        return;
    }
    let Some(u) = crate::gpu::uma::get() else { return };
    let (layout, args, title_pso) = with(this, |s| (s.layout.clone(), s.args.clone(), s.pso));
    let Some(layout) = layout else { return };
    let seq = super::rtsum::fs_seq();
    let desc = |r: usize| ID3D12Resource::from_raw_borrowed(&(r as *mut c_void)).map(|x| x.GetDesc());
    let word = |at: usize| args.get(at..at + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap()));
    let mut tables: Vec<u32> = Vec::new();
    let mut values: Vec<u32> = Vec::new();
    let mut what: Vec<String> = Vec::new();
    for (i, p) in layout.params.iter().enumerate() {
        match *p {
            Param::Constants { at, count } => {
                let w: Vec<u32> = (0..count as usize).filter_map(|k| word(at as usize + 4 * k)).collect();
                what.push(format!("p{i} constants {w:x?}"));
                values.extend(&w);
            }
            Param::Descriptor { at } => {
                let va = (word(at as usize).unwrap_or(0) as u64 | (word(at as usize + 4).unwrap_or(0) as u64) << 32) as usize;
                let win = if (va >> 32) == 0 { Some(0) } else { u.window_of(va) };
                let cpu = crate::hle::kernelx::mirrors::source(va).unwrap_or(va);
                let p = if (va >> 32) == 0 { u.window(0).0 + va } else { cpu };
                let n = 32usize;
                let w: Vec<u32> = if va != 0 && crate::gpu::shader::readable(p, 4 * n) { (0..n).map(|k| ((p + 4 * k) as *const u32).read_unaligned()).collect() } else { Vec::new() };
                what.push(format!("p{i} buffer {va:#x} window {win:?}{} cpu {w:x?}", if cpu != va { format!(" (GPU view of {cpu:#x})") } else { String::new() }));
                values.extend(&w);
            }
            Param::Table { at } => {
                let v = word(at as usize).unwrap_or(0);
                what.push(format!("p{i} table {v:#x}"));
                tables.push(v);
            }
        }
    }
    // the first 24 slots of each table as words (LOCAL: 8 before z9): null / image (PC index, the resource of that slot or
    // of the view heap's slot of that index) / buffer (address, window)
    let mut slots: Vec<String> = Vec::new();
    for &t in &tables {
        for k in 0..24u32 {
            let Some(w) = super::console_heap::words_at(t + 32 * k) else { continue };
            let s = if w.iter().all(|&v| v == 0) {
                "null".to_string()
            } else if w[3] >> 28 >= 8 {
                let r = super::console_heap::resource_at(u.base() + t as usize + 32 * k as usize).or_else(|| super::console_heap::resource_of_index(w[0]));
                if let Some(r) = r {
                    // LOCAL: full-size / half-size images, 1x1 images (exposure-like), big shadow-map-like ones
                    if std::env::var_os("XCL_LOCAL_FS_NODUMP").is_none() && desc(r).is_some_and(|d| d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D && ((d.Width >= 1280 && d.Height >= 720) || (d.Width <= 4 && d.Height <= 4) || (d.Width <= 64 && d.Height <= 64 && matches!(d.Format.0, 10 | 16 | 41 | 54)) || (d.Format.0 == 53 && d.Width >= 1024))) {
                        super::rtsum::fs_note(r);
                    }
                }
                format!(
                    "image{} idx {} -> {:x?} {:?}{}",
                    match super::console_heap::index_is_uav(w[0]) { Some(true) => " UAV", Some(false) => " SRV", None => "" },
                    w[0],
                    r,
                    r.and_then(desc).map(|d| (d.Width, d.Height, d.Format.0)),
                    if r.is_some_and(super::rtsum::fs_is_target) { " TARGET" } else { "" }
                )
            } else {
                let a = w[0] as u64 | ((w[1] & 0xFFFF) as u64) << 32;
                format!("buffer {a:#x} window {:?} records {:#x}", if (a >> 32) == 0 { Some(0) } else { u.window_of(a as usize) }, w[2])
            };
            slots.push(format!("t{t:#x}[{k}] {s}"));
        }
    }
    crate::xlog_info!("fstrace: #{seq} slots: {}", slots.join(" | "));
    let mut found: Vec<(String, usize)> = Vec::new();
    for &t in &tables {
        for k in 0..16u32 {
            if let Some(r) = super::console_heap::resource_at(u.base() + t as usize + 32 * k as usize) {
                if !found.iter().any(|x| x.1 == r) {
                    found.push((format!("t{t:#x}[{k}]"), r));
                }
            }
        }
    }
    for &v in &values {
        if v >= 1 << 20 {
            continue;
        }
        for &t in &tables {
            if let Some(r) = super::console_heap::resource_at(u.base() + t as usize + 32 * v as usize) {
                if !found.iter().any(|x| x.1 == r) {
                    found.push((format!("t{t:#x}+idx {v}"), r));
                }
            }
        }
    }
    let res: Vec<String> = found
        .iter()
        .take(40)
        .map(|(from, r)| {
            let d = ID3D12Resource::from_raw_borrowed(&(*r as *mut c_void)).map(|x| x.GetDesc());
            if d.is_some_and(|d| d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D && d.Width >= 64 && d.Height >= 64) {
                super::rtsum::fs_note(*r);
            }
            format!(
                "{from} {r:#x} {:?} title {:x?}/{:x?}{}{}",
                d.map(|d| (d.Dimension.0, d.Width, d.Height, d.Format.0, d.Flags.0)),
                super::real::title_address(*r),
                super::title_textures::address(*r),
                if super::rtsum::fs_is_target(*r) { " TARGET" } else { "" },
                if super::present::is_presented(*r) { " PRESENTED" } else { "" }
            )
        })
        .collect();
    crate::xlog_info!("fstrace: #{seq} dispatch {} groups {groups:?}: {}", super::rtsum::shaders(title_pso), what.join("; "));
    crate::xlog_info!("fstrace: #{seq} names: {}", if res.is_empty() { "nothing".to_string() } else { res.join(" | ") });
}

/// LOCAL (agent Z, `XCL_LOCAL_ZPROBE=<hex prefix>,…` with `XCL_LOCAL_FSTRACE`): in the traced frame,
/// the first 4 calls of each listed compute program: its user SGPRs, and for each descriptor table
/// of its root arguments the first 16 slots decoded as buffer V#s (address, records, stride), where
/// each points (window, GPU view of the 24 GB pair with its CPU view, a title reservation outside
/// the windows, null) and the first 8 words the CPU sees there now (record time).
unsafe fn local_zprobe(this: *mut c_void, io: &[u32], layout: Option<&Layout>, args: &[u8]) {
    static WANT: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    let want = WANT.get_or_init(|| std::env::var("XCL_LOCAL_ZPROBE").unwrap_or_default().split(',').map(|s| s.trim().to_lowercase()).filter(|s| !s.is_empty()).collect());
    if want.is_empty() {
        return;
    }
    let name = super::rtsum::shaders(with(this, |s| s.pso));
    let Some(h) = name.strip_prefix("cs ") else { return };
    if !want.iter().any(|w| h.starts_with(w.as_str())) {
        return;
    }
    static SEEN: Mutex<Vec<(String, u32)>> = Mutex::new(Vec::new());
    {
        let mut g = SEEN.lock().unwrap();
        let n = match g.iter_mut().find(|x| x.0 == h) {
            Some(x) => {
                x.1 += 1;
                x.1
            }
            None => {
                g.push((h.to_string(), 1));
                1
            }
        };
        if n > 4 {
            return;
        }
    }
    let Some(u) = crate::gpu::uma::get() else { return };
    let word = |at: usize| args.get(at..at + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).unwrap_or(0);
    let mut parts: Vec<String> = Vec::new();
    for (i, p) in layout.map(|l| l.params.clone()).unwrap_or_default().iter().enumerate() {
        match *p {
            Param::Table { at } => {
                let t = word(at as usize);
                let mut slots = Vec::new();
                for k in 0..16u32 {
                    let Some(w) = super::console_heap::words_at(t + 32 * k).or_else(|| {
                        let p = u.window(0).0 + t as usize + 16 * k as usize;
                        crate::gpu::shader::readable(p, 16).then(|| { let mut x = [0u32; 8]; std::ptr::copy_nonoverlapping(p as *const u32, x.as_mut_ptr(), 4); x })
                    }) else { continue };
                    if w[..4].iter().all(|&v| v == 0) {
                        slots.push(format!("[{k}] null"));
                        continue;
                    }
                    if w[3] >> 28 >= 8 {
                        let r = super::console_heap::resource_of_index(w[0]);
                        let dd = r.and_then(|r| ID3D12Resource::from_raw_borrowed(&(r as *mut c_void)).map(|x| x.GetDesc()));
                        if let (Some(r), Some(d)) = (r, dd) {
                            if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D {
                                super::rtsum::fs_note(r); // LOCAL: dumped at the next present
                            }
                        }
                        slots.push(format!("[{k}] image idx {} -> {:x?} {:?}", w[0], r, dd.map(|d| (d.Dimension.0, d.Width, d.Height, d.DepthOrArraySize, d.Format.0, d.MipLevels))));
                        continue;
                    }
                    let a = w[0] as u64 | ((w[1] & 0xFFFF) as u64) << 32;
                    let stride = (w[1] >> 16) & 0x3FFF;
                    let win = if (a >> 32) == 0 { Some(0) } else { u.window_of(a as usize) };
                    let (kind, cpu) = if (a >> 32) == 0 {
                        ("window 0", u.window(0).0 + a as usize)
                    } else if win.is_some() {
                        ("window", a as usize)
                    } else if let Some(src) = crate::hle::kernelx::mirrors::source(a as usize) {
                        ("GPU view (24 GB pair)", src)
                    } else if crate::hle::kernelx::in_graphics_reservation(a as usize) {
                        ("title reservation outside the windows", a as usize)
                    } else {
                        ("other", a as usize)
                    };
                    let words: Vec<u32> = if crate::gpu::shader::readable(cpu & !3, 32) { (0..8).map(|j| ((cpu & !3) as *const u32).add(j).read_unaligned()).collect() } else { Vec::new() };
                    slots.push(format!("[{k}] V# {a:#x} records {:#x} stride {stride} {kind} win {win:?} cpu {cpu:#x} {words:x?}", w[2]));
                }
                parts.push(format!("p{i} table {t:#x}: {}", slots.join(" ; ")));
            }
            Param::Descriptor { at } => parts.push(format!("p{i} root view {:#x}", word(at as usize) as u64 | (word(at as usize + 4) as u64) << 32)),
            Param::Constants { at, count } => parts.push(format!("p{i} constants {:x?}", (0..count.min(8) as usize).map(|k| word(at as usize + 4 * k)).collect::<Vec<_>>())),
        }
    }
    crate::xlog_info!("zprobe: {name} sgprs {:x?} | {}", &io[..16.min(io.len())], parts.join(" | "));
}

/// LOCAL (agent G, `XCL_LOCAL_GPROBE=<hex prefix>,…` with `XCL_LOCAL_FSTRACE`): in the traced frame,
/// the first 6 graphics binds whose VS or PS hash starts with a prefix: root constants / views, and
/// every non-null slot of the first 200 (table 0) / 32 (other tables) slots of 32 bytes, decoded as
/// image (PC index -> resource, shape; 2D images are dumped at the next present) or buffer V#
/// (address, records, where it points, first 4 CPU words).
unsafe fn local_gprobe(this: *mut c_void, layout: Option<&Layout>, args: &[u8]) {
    static WANT: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    let want = WANT.get_or_init(|| std::env::var("XCL_LOCAL_GPROBE").unwrap_or_default().split(',').map(|s| s.trim().to_lowercase()).filter(|s| !s.is_empty()).collect());
    if want.is_empty() {
        return;
    }
    let name = super::rtsum::shaders(with(this, |s| s.pso));
    let hit = name.split_whitespace().any(|h| want.iter().any(|w| h.starts_with(w.as_str())));
    if !hit {
        return;
    }
    static N: AtomicU64 = AtomicU64::new(0);
    if N.fetch_add(1, Ordering::Relaxed) >= 6 {
        return;
    }
    let Some(u) = crate::gpu::uma::get() else { return };
    let word = |at: usize| args.get(at..at + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).unwrap_or(0);
    let mut parts: Vec<String> = Vec::new();
    let mut ti = 0u32;
    for (i, p) in layout.map(|l| l.params.clone()).unwrap_or_default().iter().enumerate() {
        match *p {
            Param::Table { at } => {
                let t = word(at as usize);
                let max = if ti == 0 { 200u32 } else { 32 };
                ti += 1;
                let mut slots = Vec::new();
                for k in 0..max {
                    let Some(w) = super::console_heap::words_at(t + 32 * k) else { continue };
                    if w.iter().all(|&v| v == 0) {
                        continue;
                    }
                    if w[3] >> 28 >= 8 {
                        let r = super::console_heap::resource_at(u.base() + t as usize + 32 * k as usize).or_else(|| super::console_heap::resource_of_index(w[0]));
                        let dd = r.and_then(|r| ID3D12Resource::from_raw_borrowed(&(r as *mut c_void)).map(|x| x.GetDesc()));
                        if let (Some(r), Some(d)) = (r, dd) {
                            if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D && d.SampleDesc.Count == 1 {
                                super::rtsum::fs_note(r);
                            }
                        }
                        slots.push(format!("[{k}] img idx {} -> {:x?} {:?}{}", w[0], r, dd.map(|d| (d.Width, d.Height, d.DepthOrArraySize, d.Format.0, d.Flags.0)), if r.is_some_and(super::rtsum::fs_is_target) { " TARGET" } else { "" }));
                        continue;
                    }
                    let a = w[0] as u64 | ((w[1] & 0xFFFF) as u64) << 32;
                    let win = if (a >> 32) == 0 { Some(0) } else { u.window_of(a as usize) };
                    let (kind, cpu) = if (a >> 32) == 0 {
                        ("w0", u.window(0).0 + a as usize)
                    } else if win.is_some() {
                        ("win", a as usize)
                    } else if let Some(src) = crate::hle::kernelx::mirrors::source(a as usize) {
                        ("GPUVIEW24", src)
                    } else if crate::hle::kernelx::in_graphics_reservation(a as usize) {
                        ("RESV", a as usize)
                    } else {
                        ("OTHER", a as usize)
                    };
                    let words: Vec<u32> = if crate::gpu::shader::readable(cpu & !3, 16) { (0..4).map(|j| ((cpu & !3) as *const u32).add(j).read_unaligned()).collect() } else { Vec::new() };
                    slots.push(format!("[{k}] V# {a:#x} rec {:#x} {kind} {win:?} cpu {words:x?} w4.. {:x?}", w[2], &w[4..]));
                }
                parts.push(format!("p{i} table {t:#x}: {}", slots.join(" ; ")));
            }
            Param::Descriptor { at } => {
                let a = word(at as usize) as u64 | (word(at as usize + 4) as u64) << 32;
                parts.push(format!("p{i} root view {a:#x} window {:?}", if (a >> 32) == 0 { Some(0) } else { u.window_of(a as usize) }));
            }
            Param::Constants { at, count } => parts.push(format!("p{i} constants {:x?}", (0..count.min(8) as usize).map(|k| word(at as usize + 4 * k)).collect::<Vec<_>>())),
        }
    }
    crate::xlog_info!("gprobe: {name} | {}", parts.join(" | "));
}

/// LOCAL (agent Z, `XCL_LOCAL_ZWATCH=<hex prefix>:<table>:<slot>`, e.g. `55b6fb3a:1:1`): the buffer
/// V# that program finds in slot <slot> of its <table>-th descriptor table is remembered at each of
/// its binds; in the traced frame (`XCL_LOCAL_FSTRACE`) every translated compute bind whose tables
/// (first 24 slots each) hold a buffer V# overlapping that range is logged in record order.
static ZWATCH: Mutex<Option<(u64, u64)>> = Mutex::new(None);
static ZWATCH_FIXED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// LOCAL (agent Z): a watched range set from the indirect draw arguments (`rtsum::after_indirect`);
/// the first one wins until the traced frame. True when newly set (to log it once).
pub(super) fn local_zwatch_set(at: u64, len: u64) -> bool {
    if ZWATCH_FIXED.swap(true, Ordering::Relaxed) {
        return false;
    }
    *ZWATCH.lock().unwrap() = Some((at, len));
    true
}

/// LOCAL (agent Z, `XCL_LOCAL_CB_READBACK=1`): the counter buffer 55b6fb3a reads (its table 1, slot 1).
static CB_RANGE: Mutex<Option<(usize, usize)>> = Mutex::new(None);

pub(super) fn local_cb_range() -> Option<(usize, usize)> {
    *CB_RANGE.lock().unwrap()
}

unsafe fn local_zwatch(this: *mut c_void, layout: Option<&Layout>, args: &[u8]) {
    static CB: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *CB.get_or_init(|| std::env::var_os("XCL_LOCAL_CB_READBACK").is_some()) {
        if let Some(l) = layout {
            let word = |at: usize| args.get(at..at + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).unwrap_or(0);
            let tables: Vec<u32> = l.params.iter().filter_map(|p| if let Param::Table { at } = *p { Some(word(at as usize)) } else { None }).collect();
            if tables.len() >= 2 && super::rtsum::shaders(with(this, |s| s.pso)).starts_with("cs 55b6fb3a") {
                if let Some(w) = super::console_heap::words_at(tables[1] + 32) {
                    let a = (w[0] as u64 | ((w[1] & 0xFFFF) as u64) << 32) as usize;
                    let stride = ((w[1] >> 16) & 0x3FFF).max(1) as usize;
                    let n = (w[2] as usize * stride).min(0x1000);
                    let mut g = CB_RANGE.lock().unwrap();
                    if g.is_none() {
                        crate::xlog_info!("LOCAL cb-readback: counter buffer {a:#x}+{n:#x} (55b6fb3a table 1 slot 1)");
                    }
                    *g = Some((a, n));
                }
            }
        }
    }
    static CFG: std::sync::OnceLock<Option<(String, usize, u32)>> = std::sync::OnceLock::new();
    let Some((want, ti, si)) = CFG.get_or_init(|| {
        let v = std::env::var("XCL_LOCAL_ZWATCH").ok()?;
        let p: Vec<&str> = v.trim().split(':').collect();
        Some((p.first()?.to_lowercase(), p.get(1)?.parse().ok()?, p.get(2)?.parse().ok()?))
    }) else { return };
    let Some(l) = layout else { return };
    let word = |at: usize| args.get(at..at + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).unwrap_or(0);
    let tables: Vec<u32> = l.params.iter().filter_map(|p| if let Param::Table { at } = *p { Some(word(at as usize)) } else { None }).collect();
    let vsharp = |t: u32, k: u32| -> Option<(u64, u64, [u32; 8])> {
        let w = super::console_heap::words_at(t + 32 * k)?;
        if w[..4].iter().all(|&v| v == 0) || w[3] >> 28 >= 8 {
            return None;
        }
        let a = w[0] as u64 | ((w[1] & 0xFFFF) as u64) << 32;
        let stride = ((w[1] >> 16) & 0x3FFF) as u64;
        Some((a, w[2] as u64 * stride.max(1), w))
    };
    let name = super::rtsum::shaders(with(this, |s| s.pso));
    if !ZWATCH_FIXED.load(Ordering::Relaxed) && name.strip_prefix("cs ").is_some_and(|h| h.starts_with(want.as_str())) {
        if let Some((a, n, _)) = tables.get(*ti).and_then(|&t| vsharp(t, *si)) {
            let mut g = ZWATCH.lock().unwrap();
            if g.is_none() || super::rtsum::fs_active() {
                crate::xlog_info!("zwatch: {name} table {ti} slot {si}: watching {a:#x}+{n:#x}");
            }
            *g = Some((a, n.max(4)));
        }
    }
    if !super::rtsum::fs_active() {
        return;
    }
    let Some((wa, wn)) = *ZWATCH.lock().unwrap() else { return };
    let mut hits = Vec::new();
    // LOCAL: root views too (address only: a view starting at most 1 MB below the range)
    for (i, p) in l.params.iter().enumerate() {
        if let Param::Descriptor { at } = *p {
            let a = word(at as usize) as u64 | (word(at as usize + 4) as u64) << 32;
            if a != 0 && a <= wa + wn && wa < a + 0x10_0000 {
                hits.push(format!("root view p{i} {a:#x}"));
            }
        }
    }
    for (i, &t) in tables.iter().enumerate() {
        for k in 0..64u32 {
            if let Some((a, n, w)) = vsharp(t, k) {
                // LOCAL: only plausible buffer V#s (title memory, at most 256 MB)
                if a < wa + wn && wa < a + n.max(4) && n <= 0x1000_0000 && (0x4_0000_0000..0x8_0000_0000).contains(&a) {
                    hits.push(format!("table {i} slot {k} V# {a:#x}+{n:#x} words {:x?}", &w[..4]));
                }
            }
        }
    }
    // LOCAL (agent Z, `XCL_LOCAL_ZSNAP_ALL=1`): a snapshot before EVERY bind of the traced frame, with the
    // words the CPU sees in the range at record time
    static ALL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *ALL.get_or_init(|| std::env::var_os("XCL_LOCAL_ZSNAP_ALL").is_some()) {
        static SEQA: AtomicU64 = AtomicU64::new(0);
        let cpu: Vec<u32> = if crate::gpu::shader::readable(wa as usize, 16) { (0..4).map(|j| (wa as *const u32).add(j).read_unaligned()).collect() } else { Vec::new() };
        let seq = SEQA.fetch_add(1, Ordering::Relaxed);
        super::rtsum::zsnap(this, wa as usize, wn, format!("@{seq} {name}{} cpu {cpu:x?}", if hits.is_empty() { "" } else { " NAMES IT" }));
        return;
    }
    if !hits.is_empty() {
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let seq = SEQ.fetch_add(1, Ordering::Relaxed);
        crate::xlog_info!("zwatch: #{seq} {name} names the watched range: {}", hits.join("; "));
        super::rtsum::zsnap(this, wa as usize, wn, format!("#{seq} {name}"));
    }
}

/// #148 AllocateUploadMemory(size, alignment) → CPU/GPU pointer in unified memory (used by
/// `XboxBvhBuilder.dll` and other console middleware on default title command lists).
/// Freed automatically when the list's command allocator is reset.
pub(super) unsafe extern "system" fn allocate_upload_memory(this: *mut c_void, size: usize, align: usize) -> *mut u8 {
    let Some(u) = crate::gpu::uma::get() else {
        return std::ptr::null_mut();
    };
    let size = size.max(16);
    let align = align.max(16);
    if size <= crate::gpu::uma::ARENA_BLOCK / 2 {
        if let Some(p) = with(this, |s| s.arena.take(u, size, align, &mut s.uma)) {
            return p;
        }
    }
    if u.has_sparse() {
        if let Some((p, _)) = super::sparse::alloc(size, align) {
            with(this, |s| s.uma.push(p));
            return p as *mut u8;
        }
    }
    match u.alloc(size, align) {
        Some(p) => {
            with(this, |s| s.uma.push(p as usize));
            p
        }
        None => std::ptr::null_mut(),
    }
}

unsafe fn translated_dispatch(this: *mut c_void, list: &ID3D12GraphicsCommandList, pso: usize, x: u32, y: u32, z: u32) -> Result<(), String> {
    if super::present::trace_frame() {
        trace_fullscreen(this, pso, [x, y, z]);
    }
    if super::rtsum::fs_active() {
        local_fs_trace(this, [x, y, z]); // LOCAL (agent Z)
    }
    super::lcap::note_groups([x, y, z]); // LOCAL (agent H)
    bind_abi(this, list, pso, false)?;
    // LOCAL diagnostic (`XCL_LOCAL_DUMP_CS_INPUTS`): what the title feeds this program; its GPU
    // read passes (before and after the dispatch) change the list's compute state, bound again
    let probe = if super::local_inputs::active() {
        let (layout, args) = with(this, |s| (s.layout.clone(), s.args.clone()));
        let user_sgprs = super::real::pso_user_sgprs(super::pso_async::resolve_wait(pso));
        let (probe, changed) = super::local_inputs::before_dispatch(list, with(this, |s| s.pso), pso, [x, y, z], layout.as_deref(), &args, user_sgprs, "the title's D3D12 Dispatch call (x, y, z as it passed them)");
        if changed {
            bind_abi(this, list, pso, false)?;
        }
        probe
    } else {
        None
    };
    let mark = super::culprit::before_dispatch(list, with(this, |s| s.pso), pso, [x, y, z]);
    list.Dispatch(x, y, z);
    super::lcap::after_dispatch(this); // LOCAL (agent H)
    super::culprit::after_dispatch(list, mark);
    if super::local_inputs::after_dispatch(list, probe) {
        bind_abi(this, list, pso, false)?;
    }
    Ok(())
}

/// XCL's ABI for a translated pipeline: its root signature, the user SGPRs (`io`, the kept
/// arguments the console's way – compute or graphics) and the argument table in unified
/// memory, u1 res, u2… the memory windows; then the pipeline.
unsafe fn bind_abi(this: *mut c_void, list: &ID3D12GraphicsCommandList, pso: usize, gfx: bool) -> Result<(), String> {
    super::list_x::ensure_view_heap(this);
    let u = crate::gpu::uma::get().ok_or("no unified memory")?;
    let windows = super::real::uma_window_vas();
    let first = *windows.first().ok_or("unified memory is not open on the GPU")?;
    // shadowed unified memory whose sparse windows could not be made (no reserved resources
    // either): descriptor tables are read from window 0, and a null root descriptor is not bound
    if first == 0 && u.shadowed() {
        return Err("window 0 of unified memory has no buffer on the GPU".into());
    }
    let (layout, mut args, heaps) = with(this, |s| { // LOCAL: mut (agent Z)
        let (layout, args) = if gfx { (&s.layout_gfx, &s.args_gfx) } else { (&s.layout, &s.args) };
        (layout.clone(), args.clone(), s.heaps)
    });
    super::rtsum::oow_copy_root_views(layout.as_deref(), &mut args); // LOCAL experiment (agent L's XCL_LOCAL_OOW_COPY, ported by agent Z)
    super::rtsum::oow_scratch_root_views(layout.as_deref(), &mut args); // LOCAL experiment (agent L's XCL_LOCAL_OOW_SCRATCH, ported by agent Z)
    // 32-bit console addresses: window 0; pieces of the list's arena (not a 64 KiB grain each)
    let copy = |bytes: &[u8]| -> Result<usize, String> {
        let p = with(this, |s| s.arena.take(u, bytes.len().max(16), 256, &mut s.uma)).ok_or("unified memory is full")?;
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), p, bytes.len());
        Ok(p as usize)
    };
    // a pipeline still being built in the background is waited for
    let real = super::pso_async::resolve_wait(pso);
    // a compute shader takes no more arguments from registers than it has (`inline_for`)
    let user_sgprs = if gfx { None } else { super::real::pso_user_sgprs(real) };
    let inline = layout.as_ref().map_or(0, |l| if gfx { l.inline_gfx } else { console_list::inline_for(l, user_sgprs) } as usize).min(args.len());
    super::rtsum::oow_scan(with(this, |s| s.pso), gfx, layout.as_deref(), &args); // LOCAL (agent L, ported by agent Z)
    let table_at = copy(&args[inline..])?;
    let mut io = match layout.as_ref() {
        Some(l) if gfx => console_list::with_heaps(l, None, console_list::user_data_gfx(l, &args, table_at as u64), heaps),
        Some(l) => console_list::with_heaps(l, user_sgprs, console_list::user_data_for(l, user_sgprs, &args, table_at as u64), heaps),
        None => vec![table_at as u32, (table_at >> 32) as u32],
    };
    io.resize(crate::gpu::recomp::KERNEL_SGPRS, 0);
    io[crate::gpu::recomp::IO_UMA_BASE_HI] = (u.base() >> 32) as u32;
    super::lcap::on_bind(this, with(this, |s| s.pso), gfx, &io, table_at, &args[inline..]); // LOCAL (agent H)
    local_zwatch(this, layout.as_deref(), &args); // LOCAL (agent Z): compute and graphics binds
    if !gfx && super::rtsum::fs_active() {
        local_zprobe(this, &io, layout.as_deref(), &args); // LOCAL (agent Z)
    }
    if gfx && super::rtsum::fs_active() {
        local_gprobe(this, layout.as_deref(), &args); // LOCAL (agent G)
    }
    let io_at = copy(&io.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<u8>>())?;
    let res_at = copy(&[0u8; 256])?;
    let va = |p: usize| super::real::uma_va_of(p).unwrap_or(first);
    let rs = super::real::pso_root_signature(real).or_else(|| super::real::abi_rs_cached().map(|p| p as usize)).ok_or("no XCL ABI root signature")?;
    let rs = rs as *mut c_void;
    let rs = ID3D12RootSignature::from_raw_borrowed(&rs).ok_or("root signature")?;
    let views = [va(io_at), va(res_at)].into_iter().chain((0..4).map(|k| windows.get(k).copied().unwrap_or(first)));
    if gfx {
        list.SetGraphicsRootSignature(rs);
        for (k, v) in views.enumerate() {
            list.SetGraphicsRootUnorderedAccessView(k as u32, v);
        }
    } else {
        list.SetComputeRootSignature(rs);
        for (k, v) in views.enumerate() {
            list.SetComputeRootUnorderedAccessView(k as u32, v);
        }
    }
    let real = real as *mut c_void;
    let pc = ID3D12PipelineState::from_raw_borrowed(&real).ok_or("pipeline")?;
    list.SetPipelineState(pc);
    Ok(())
}

#[cfg(test)]
pub(super) fn last_draw_args() -> (bool, u32, u32, u32, i32) {
    DRAW_ARGS.with(|d| d.get())
}

#[cfg(test)]
pub(super) fn counts_snapshot() -> [u64; 5] {
    std::array::from_fn(|k| COUNTS[k].load(Ordering::Relaxed))
}

/// the pipeline the title bound on `list` (its object; 0: none known)
pub(super) fn tracked_pso(list: *mut c_void) -> usize {
    STATES
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|m| m.get(&(list as usize)))
        .map_or(0, |s| s.pso)
}

/// Tracked console `(graphics_root_signature, compute_root_signature)` on `list` for restoring
/// PC root signatures after an internal graphics/compute sync pass.
pub(super) fn tracked_root_sigs(list: *mut c_void) -> (usize, usize) {
    STATES
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|m| m.get(&(list as usize)))
        .map_or((0, 0), |s| (s.rs_gfx, s.rs_cs))
}

#[cfg(test)]
mod tests {
    use super::super::{device_x, list_x, pso_async, real, title_list};
    use super::*;
    use std::sync::atomic::AtomicUsize;

    /// A pipeline with the rect list GS draws its rect lists whole: draw_rects leaves them alone.
    #[test]
    fn rect_lists_with_the_rect_list_gs_are_not_expanded_on_the_cpu() {
        let (with_gs, without) = (0x7E57_4000usize, 0x7E57_5000usize);
        real::register_rect_list_gs_for_test(with_gs);
        assert!(!cpu_rect_expansion(with_gs));
        assert!(cpu_rect_expansion(without) && cpu_rect_expansion(0));
        real::forget_pso_for_test(with_gs);
        assert!(cpu_rect_expansion(with_gs));
    }

    /// An ExecuteIndirect whose bound pipeline is still being built is skipped (its stand-in
    /// with XCL's ABI root signature on the list is the debug layer's 953 and a hung GPU);
    /// a built or failed one goes to the PC call. The gate `execute_indirect` uses.
    #[test]
    fn execute_indirect_with_a_pipeline_still_building_is_skipped() {
        let (building, built, failed) = (0x7E57_6001usize, 0x7E57_6002usize, 0x7E57_6003usize);
        // lazy submits: no worker starts them, so the states stay under the test's control
        pso_async::submit_lazy(building, false, Box::new(|| 0x1234), 0);
        pso_async::submit_lazy(built, false, Box::new(|| 0x1235), 0);
        pso_async::submit_lazy(failed, false, Box::new(|| 0), 0);
        assert!(pso_async::is_pending(building));
        assert_eq!(pso_async::resolve_wait(built), 0x1235);
        assert_eq!(pso_async::resolve_wait(failed), failed);
        assert!(!pso_async::is_pending(built) && !pso_async::is_pending(failed));
        // never resolved: still pending (the skip case), then forgotten
        assert!(pso_async::is_pending(building));
        pso_async::released(building);
        pso_async::released(built);
        pso_async::released(failed);
        assert!(!pso_async::is_pending(building));
    }

    /// A dispatch writes an ExecuteIndirect's group count through another buffer over the same
    /// memory (as translated shaders write through the memory windows), a UAV barrier (what a
    /// console barrier becomes), then the ExecuteIndirect: it runs the count just written, not
    /// the one the memory held before the list (`fresh_indirect_arguments`; without it an RTX
    /// 5090 ran the old count in nearly every try).
    #[test]
    fn execute_indirect_reads_the_arguments_written_before_it() {
        use crate::gpu::{fence, recomp::runner::skip_test};
        use std::mem::{transmute_copy, ManuallyDrop};
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;

        const TRIES: u32 = 40;
        // mode 0: write (x, 1, 1) at a[0..12]; mode 1: count the groups
        let src = r#"
#define RS "RootConstants(num32BitConstants=2, b0), UAV(u0), UAV(u1)"
cbuffer C : register(b0) { uint mode; uint x; };
RWByteAddressBuffer a : register(u0);
RWByteAddressBuffer c : register(u1);
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint t : SV_GroupIndex) {
    if (t != 0u) return;
    if (mode == 0u) a.Store3(0, uint3(x, 1u, 1u)); else c.InterlockedAdd(0, 1u);
}
"#;
        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                skip_test("no GPU");
                return;
            }
            let dev = dev.unwrap();
            let dxil = crate::gpu::recomp::dxc::compile(src, "main", "cs_6_6", &[]).unwrap();
            let rs: ID3D12RootSignature = dev.CreateRootSignature(0, &dxil).unwrap();
            let cs = D3D12_SHADER_BYTECODE { pShaderBytecode: dxil.as_ptr() as _, BytecodeLength: dxil.len() };
            let pso: ID3D12PipelineState = dev.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC { pRootSignature: transmute_copy(&rs), CS: cs, ..Default::default() }).unwrap();
            let arg = D3D12_INDIRECT_ARGUMENT_DESC { Type: D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH, ..Default::default() };
            let mut sig: Option<ID3D12CommandSignature> = None;
            dev.CreateCommandSignature(&D3D12_COMMAND_SIGNATURE_DESC { ByteStride: 12, NumArgumentDescs: 1, pArgumentDescs: &arg, NodeMask: 0 }, None, &mut sig).unwrap();
            let sig = sig.unwrap();
            let ty = D3D12_COMMAND_LIST_TYPE_COMPUTE;
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: ty, ..Default::default() }).unwrap();
            let done: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let desc = |width| D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: width,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                ..Default::default()
            };
            // the arguments and the buffer the dispatch writes them through: one memory
            let mut heap: Option<ID3D12Heap> = None;
            let heap_desc = D3D12_HEAP_DESC { SizeInBytes: 1 << 16, Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, Alignment: 0, Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS };
            dev.CreateHeap(&heap_desc, &mut heap).unwrap();
            let heap = heap.unwrap();
            let placed = || {
                let mut r: Option<ID3D12Resource> = None;
                dev.CreatePlacedResource(&heap, 0, &desc(1 << 16), D3D12_RESOURCE_STATE_COMMON, None, &mut r).unwrap();
                r.unwrap()
            };
            let (args, written) = (placed(), placed());
            // the group count, read on the CPU
            let cpu_heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_CUSTOM, CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK, MemoryPoolPreference: D3D12_MEMORY_POOL_L0, ..Default::default() };
            let mut groups: Option<ID3D12Resource> = None;
            dev.CreateCommittedResource(&cpu_heap, D3D12_HEAP_FLAG_NONE, &desc(256), D3D12_RESOURCE_STATE_COMMON, None, &mut groups).unwrap();
            let groups = groups.unwrap();
            let mut count = std::ptr::null_mut();
            groups.Map(0, None, Some(&mut count)).unwrap();
            let count = count as *mut u32;
            let uav_barrier = D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                Anonymous: D3D12_RESOURCE_BARRIER_0 { UAV: ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER { pResource: ManuallyDrop::new(None) }) },
            };
            let run = |list: &ID3D12GraphicsCommandList, close: &dyn Fn()| {
                close();
                queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
                let v = done.GetCompletedValue() + 1;
                queue.Signal(&done, v).unwrap();
                fence::wait(&done, v, &dev, None, 10_000).unwrap();
            };
            let (mut old, mut wrong) = (0u32, Vec::new());
            for k in 0..TRIES {
                // each try writes a count the memory does not hold yet
                let new = if k % 2 == 0 { 4096 } else { 0 };
                let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(ty).unwrap();
                let record = |x: u32| -> ID3D12GraphicsCommandList {
                    let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, ty, &alloc, &pso).unwrap();
                    list.SetComputeRootSignature(&rs);
                    list.SetComputeRootUnorderedAccessView(1, written.GetGPUVirtualAddress());
                    list.SetComputeRootUnorderedAccessView(2, groups.GetGPUVirtualAddress());
                    list.SetComputeRoot32BitConstants(0, 2, [0u32, x].as_ptr() as _, 0);
                    list.Dispatch(1, 1, 1);
                    list
                };
                let before = record(old);
                run(&before, &|| before.Close().unwrap());
                alloc.Reset().unwrap();
                *count = 0;
                let list = record(new);
                list.ResourceBarrier(&[uav_barrier.clone()]);
                list.SetComputeRoot32BitConstants(0, 2, [1u32, 0].as_ptr() as _, 0);
                // from here as XCL records a title's ExecuteIndirect: on the adopted list
                let raw = list.as_raw();
                real::adopt(raw, real::Kind::CommandList);
                list_x::on_create(raw, alloc.as_raw() as usize, 0, ty);
                type Ei = unsafe extern "system" fn(*mut c_void, usize, u32, usize, u64, usize, u64);
                type Close = unsafe extern "system" fn(*mut c_void) -> windows::core::HRESULT;
                let ei: Ei = std::mem::transmute(real::orig_fn(raw, 59));
                fresh_indirect_arguments(raw, || ei(raw, sig.as_raw() as usize, 1, args.as_raw() as usize, 0, 0, 0));
                let close: Close = std::mem::transmute(real::orig_fn(raw, 9));
                run(&list, &|| close(raw).unwrap());
                let ran = std::ptr::read_volatile(count);
                if ran != new {
                    wrong.push((k, ran));
                }
                old = new;
            }
            groups.Unmap(0, None);
            assert!(wrong.is_empty(), "{} of {TRIES} ExecuteIndirects ran another group count than the one written before them (try, groups): {wrong:?}", wrong.len());
        }
    }

    /// Minimal RTS0 v2 root signature blob with one 4-constant parameter.
    fn test_rts0_blob() -> Vec<u8> {
        let words: [u32; 12] = [
            2, 1, 24, 0, 0, 0, // version=2, num_params=1, params_offset=24
            1, 0, 36,          // param 0: type=Constants(1), visibility=0, payload_offset=36
            0, 0, 4,           // shader_reg=0, reg_space=0, num_32bit_values=4
        ];
        words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    #[test]
    fn reset_and_clear_state_track_initial_pso_and_clear_root_args() {
        let list = 0x7100_0001usize as *mut c_void;
        let rs_compute = 0x7100_0010usize;
        let rs_gfx = 0x7100_0020usize;
        let alloc1 = 0x7100_0100usize;
        let alloc2 = 0x7100_0200usize;
        let pso_init = 0x7100_1000usize;
        let pso_mid = 0x7100_2000usize;
        let pso_cleared = 0x7100_3000usize;
        let pso_reset = 0x7100_4000usize;

        let blob = test_rts0_blob();
        console_list::note_root_signature(rs_compute, &blob);
        console_list::note_root_signature(rs_gfx, &blob);

        // Initial CreateCommandList / reset seeds PSO and allocator with clean root state.
        reset(list, alloc1, pso_init);
        with(list, |s| {
            assert_eq!(s.pso, pso_init);
            assert_eq!(s.allocator, alloc1);
            assert!(s.layout.is_none() && s.args.is_empty());
            assert!(s.layout_gfx.is_none() && s.args_gfx.is_empty());
        });

        // Populate compute + graphics root signatures, constants, and a different PSO.
        unsafe {
            set_compute_root_signature(list, rs_compute as *mut c_void);
            set_compute_root_constant(list, 0, 0xDEAD_BEEF, 0);
            set_graphics_root_signature(list, rs_gfx as *mut c_void);
            set_graphics_root_constant(list, 0, 0xCAFE_BABE, 0);
        }
        with(list, |s| {
            s.pso = pso_mid;
            assert!(s.layout.is_some() && !s.args.is_empty());
            assert!(s.layout_gfx.is_some() && !s.args_gfx.is_empty());
        });

        // ClearState(pso_cleared) updates PSO and clears root signatures/args while keeping allocator.
        clear_state(list, pso_cleared);
        with(list, |s| {
            assert_eq!(s.pso, pso_cleared);
            assert_eq!(s.allocator, alloc1);
            assert!(s.layout.is_none() && s.args.is_empty());
            assert!(s.layout_gfx.is_none() && s.args_gfx.is_empty());
        });

        // ClearState(null) clears PSO to 0.
        clear_state(list, 0);
        assert_eq!(tracked_pso(list), 0);

        // Re-populate root args, then Reset(alloc2, pso_reset) switches allocator/PSO and clears root args.
        unsafe {
            set_compute_root_signature(list, rs_compute as *mut c_void);
            set_compute_root_constant(list, 0, 0x1234_5678, 0);
            set_graphics_root_signature(list, rs_gfx as *mut c_void);
            set_graphics_root_constant(list, 0, 0x8765_4321, 0);
        }
        reset(list, alloc2, pso_reset);
        with(list, |s| {
            assert_eq!(s.pso, pso_reset);
            assert_eq!(s.allocator, alloc2);
            assert!(s.layout.is_none() && s.args.is_empty());
            assert!(s.layout_gfx.is_none() && s.args_gfx.is_empty());
        });

        // Reset with null initial PSO clears stale PSO back to 0.
        reset(list, alloc2, 0);
        assert_eq!(tracked_pso(list), 0);

        forget(list);
    }

    #[repr(C)]
    struct FakeAdoptedList {
        vt: *const usize,
        refs: AtomicUsize,
        dispatch_calls: AtomicUsize,
        set_compute_rs_calls: AtomicUsize,
        set_compute_uav_calls: AtomicUsize,
        set_pso_calls: AtomicUsize,
        execute_indirect_calls: AtomicUsize,
    }

    unsafe extern "system" fn fake_add_ref(this: *mut c_void) -> u32 {
        (*(this as *const FakeAdoptedList)).refs.fetch_add(1, Ordering::Relaxed) as u32 + 1
    }

    unsafe extern "system" fn fake_release(this: *mut c_void) -> u32 {
        (*(this as *const FakeAdoptedList)).refs.fetch_sub(1, Ordering::Relaxed) as u32 - 1
    }

    unsafe extern "system" fn fake_dispatch(this: *mut c_void, _x: u32, _y: u32, _z: u32) {
        (*(this as *const FakeAdoptedList)).dispatch_calls.fetch_add(1, Ordering::Relaxed);
    }

    unsafe extern "system" fn fake_set_pso(this: *mut c_void, _pso: *mut c_void) {
        (*(this as *const FakeAdoptedList)).set_pso_calls.fetch_add(1, Ordering::Relaxed);
    }

    unsafe extern "system" fn fake_set_compute_rs(this: *mut c_void, _rs: *mut c_void) {
        (*(this as *const FakeAdoptedList)).set_compute_rs_calls.fetch_add(1, Ordering::Relaxed);
    }

    unsafe extern "system" fn fake_set_compute_uav(this: *mut c_void, _slot: u32, _va: u64) {
        (*(this as *const FakeAdoptedList)).set_compute_uav_calls.fetch_add(1, Ordering::Relaxed);
    }

    unsafe extern "system" fn fake_execute_indirect(
        this: *mut c_void,
        _sig: usize,
        _max: u32,
        _args: usize,
        _args_off: u64,
        _count_buf: usize,
        _count_off: u64,
    ) {
        (*(this as *const FakeAdoptedList)).execute_indirect_calls.fetch_add(1, Ordering::Relaxed);
    }

    #[test]
    fn async_compute_pso_success_and_failure_in_dispatch_and_execute_indirect() {
        let mut orig_vt = [0usize; 64];
        orig_vt[1] = fake_add_ref as usize;
        orig_vt[2] = fake_release as usize;
        orig_vt[14] = fake_dispatch as usize;
        orig_vt[25] = fake_set_pso as usize;
        orig_vt[29] = fake_set_compute_rs as usize;
        orig_vt[41] = fake_set_compute_uav as usize;
        orig_vt[59] = fake_execute_indirect as usize;

        // Extended vtable layout expected by `real::orig_fn`: entry -1 holds `orig_vt` pointer,
        // and entries 0..64 hold the COM method pointers used via `ID3D12GraphicsCommandList`.
        let mut ext_vt = vec![0usize; 65];
        ext_vt[0] = orig_vt.as_ptr() as usize;
        ext_vt[1..].copy_from_slice(&orig_vt);

        let fake = Box::new(FakeAdoptedList {
            // SAFETY: `ext_vt` has 65 elements so `.add(1)` points to slot 0 with `orig_vt` at `-1`.
            vt: unsafe { ext_vt.as_ptr().add(1) },
            refs: AtomicUsize::new(1),
            dispatch_calls: AtomicUsize::new(0),
            set_compute_rs_calls: AtomicUsize::new(0),
            set_compute_uav_calls: AtomicUsize::new(0),
            set_pso_calls: AtomicUsize::new(0),
            execute_indirect_calls: AtomicUsize::new(0),
        });
        let list_ptr = &*fake as *const FakeAdoptedList as *mut c_void;

        // 1. Background compute build SUCCEEDS: real PSO R is registered in TRANSLATED_COMPUTE,
        // stand-in S is NOT. ExecuteIndirect resolves S -> R and treats it as translated.
        let standin_ok = 0x7200_0001usize;
        let real_ok = 0x7200_0002usize;
        let sig_root_changing = 0x7200_00F0usize;
        note_root_changing_signature(sig_root_changing);

        pso_async::submit(
            standin_ok,
            true,
            Box::new(move || {
                real::register_translated_compute_for_test(real_ok);
                real_ok
            }),
        );
        assert_eq!(pso_async::resolve_wait(standin_ok), real_ok);
        assert!(!real::is_translated_compute(standin_ok));
        assert!(real::is_translated_compute(real_ok));

        reset(list_ptr, 0x7200_0100, standin_ok);
        unsafe {
            // With `sig_root_changing`, a translated compute PSO is caught by the root-changing
            // guard instead of falling through to `pc` (which happened when `R` wasn't registered).
            execute_indirect(list_ptr, sig_root_changing, 1, 0, 0, 0, 0);
        }
        assert_eq!(
            fake.execute_indirect_calls.load(Ordering::Relaxed),
            0,
            "ExecuteIndirect must recognize resolved async compute PSO R as translated"
        );

        // 2. Background compute build FAILS (returns 0): stand-in S_fail is NOT in
        // TRANSLATED_COMPUTE. Dispatch must skip cleanly without binding XCL's ABI RS/UAVs
        // onto S_fail, and ExecuteIndirect must not enter bind_abi either.
        let standin_fail = 0x7200_0003usize;
        pso_async::submit(standin_fail, true, Box::new(|| 0));
        assert_eq!(pso_async::resolve_wait(standin_fail), standin_fail);
        assert!(!real::is_translated_compute(standin_fail));

        reset(list_ptr, 0x7200_0100, standin_fail);
        let before = counts_snapshot();
        unsafe {
            dispatch(list_ptr, 1, 1, 1);
        }
        let after = counts_snapshot();
        assert_eq!(after[DISPATCHES], before[DISPATCHES]);
        assert!(after[DISPATCHES_SKIPPED] > before[DISPATCHES_SKIPPED]);
        assert_eq!(fake.set_compute_rs_calls.load(Ordering::Relaxed), 0);
        assert_eq!(fake.set_compute_uav_calls.load(Ordering::Relaxed), 0);
        assert_eq!(fake.set_pso_calls.load(Ordering::Relaxed), 0);
        assert_eq!(fake.dispatch_calls.load(Ordering::Relaxed), 0);

        unsafe {
            execute_indirect(list_ptr, sig_root_changing, 1, 0, 0, 0, 0);
        }
        assert_eq!(fake.set_compute_rs_calls.load(Ordering::Relaxed), 0);
        assert_eq!(fake.set_compute_uav_calls.load(Ordering::Relaxed), 0);
        assert_eq!(fake.set_pso_calls.load(Ordering::Relaxed), 0);
        assert_eq!(
            fake.execute_indirect_calls.load(Ordering::Relaxed),
            1,
            "failed async compute stand-in must not enter translated ExecuteIndirect path"
        );

        // 3. The same failed stand-in (made from a console shader) with a Dispatch signature:
        // nothing to run, as for Dispatch – the PC call would pair the stand-in with the list's
        // compute root signature (XCL's ABI one after a translated dispatch: debug layer 953)
        let sig_dispatch = 0x7200_00F8usize;
        real::register_console_compute_for_test(standin_fail);
        note_signature_dispatches(sig_dispatch, true);
        let before = counts_snapshot();
        unsafe {
            execute_indirect(list_ptr, sig_dispatch, 1, 0, 0, 0, 0);
        }
        let after = counts_snapshot();
        assert_eq!(fake.execute_indirect_calls.load(Ordering::Relaxed), 1, "a dispatch of an untranslated console compute pipeline must not reach the PC ExecuteIndirect");
        assert!(after[DISPATCHES_SKIPPED] > before[DISPATCHES_SKIPPED]);

        // 4. A native PC compute pipeline (not made from a console shader) with the same Dispatch
        // signature still goes to the PC call, as on main
        let native = 0x7200_0004usize;
        reset(list_ptr, 0x7200_0100, native);
        unsafe {
            execute_indirect(list_ptr, sig_dispatch, 1, 0, 0, 0, 0);
        }
        assert_eq!(fake.execute_indirect_calls.load(Ordering::Relaxed), 2, "a native PC compute pipeline must still reach the PC ExecuteIndirect");

        // a signature made later at the same address without a Dispatch argument goes to the PC
        reset(list_ptr, 0x7200_0100, standin_fail);
        note_signature_dispatches(sig_dispatch, false);
        unsafe {
            execute_indirect(list_ptr, sig_dispatch, 1, 0, 0, 0, 0);
        }
        assert_eq!(fake.execute_indirect_calls.load(Ordering::Relaxed), 3);

        forget(list_ptr);
        let _ = pso_async::released(standin_ok);
        let _ = pso_async::released(standin_fail);
        real::forget_pso_for_test(real_ok);
        real::forget_pso_for_test(standin_fail);
        assert!(!real::is_console_compute(standin_fail), "a released pipeline's address does not stay a console one");
    }

    #[test]
    fn create_list_reset_and_clear_state_track_initial_pso_on_d3d12() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;

        if console_list::enabled() {
            return;
        }
        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let warp: IDXGIAdapter = factory.EnumWarpAdapter().expect("WARP");
                D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).expect("WARP D3D12 device");
            }
            let dev = dev.expect("D3D12 device");
            real::adopt(dev.as_raw(), real::Kind::Device);

            // Build two real PC compute PSOs from the inbox dummy CS bytecode.
            let mut ser = None;
            D3D12SerializeRootSignature(&D3D12_ROOT_SIGNATURE_DESC::default(), D3D_ROOT_SIGNATURE_VERSION_1, &mut ser, None).expect("serialize RS");
            let ser = ser.unwrap();
            let rs: ID3D12RootSignature = dev
                .CreateRootSignature(0, std::slice::from_raw_parts(ser.GetBufferPointer() as *const u8, ser.GetBufferSize()))
                .expect("CreateRootSignature");
            let cs = real::dummy_cs_bytecode();
            let make_pso = || -> ID3D12PipelineState {
                let mut desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::ManuallyDrop::new(Some(rs.clone())),
                    CS: D3D12_SHADER_BYTECODE { pShaderBytecode: cs.as_ptr() as *const c_void, BytecodeLength: cs.len() },
                    ..Default::default()
                };
                let pso = dev.CreateComputePipelineState(&desc).expect("CreateComputePipelineState");
                std::mem::ManuallyDrop::drop(&mut desc.pRootSignature);
                pso
            };
            let pso1 = make_pso();
            let pso2 = make_pso();
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).expect("alloc");

            // Stand-in synthetic key backed by real `pso1` in pso_async: CreateCommandList must
            // resolve it via resolve_for_bind for PC D3D12 while tracking `standin` in `bind`.
            let standin = 0x7300_0001usize;
            let pso1_raw = pso1.as_raw() as usize;
            pso_async::submit(standin, true, Box::new(move || pso1_raw));
            assert_eq!(pso_async::resolve_wait(standin), pso1_raw);

            let mut raw_list: *mut c_void = std::ptr::null_mut();
            let iid = windows_sys::core::GUID::from_u128(0x5b160d0f_ac1b_4185_8ba8_b3ae42a5a455);
            let hr = device_x::create_command_list_pc(
                dev.as_raw(),
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT.0 as u32,
                alloc.as_raw() as usize,
                standin,
                &iid,
                &mut raw_list,
            );
            assert!(hr >= 0 && !raw_list.is_null(), "create_command_list_pc failed: {hr:#x}");
            let backing = title_list::backing(raw_list).expect("wrapped in title_list");
            assert_eq!(tracked_pso(backing), standin, "CreateCommandList must seed bind::State.pso");

            // ClearState(pso2) updates bind::State.pso to pso2 and clears root args.
            let rs_key = 0x7300_0010usize;
            console_list::note_root_signature(rs_key, &test_rts0_blob());
            set_compute_root_signature(backing, rs_key as *mut c_void);
            set_compute_root_constant(backing, 0, 0xABCD, 0);
            let cl = ID3D12GraphicsCommandList::from_raw_borrowed(&backing).unwrap();
            cl.ClearState(&pso2);
            assert_eq!(tracked_pso(backing), pso2.as_raw() as usize, "ClearState must update bind::State.pso");
            with(backing, |s| assert!(s.layout.is_none() && s.args.is_empty()));

            // ClearState(None) clears bind::State.pso to 0.
            cl.ClearState(None);
            assert_eq!(tracked_pso(backing), 0, "ClearState(None) must clear bind::State.pso");

            // Close + reset_x(alloc, pso2) updates bind::State.pso to pso2 and clears root args.
            set_graphics_root_signature(backing, rs_key as *mut c_void);
            set_graphics_root_constant(backing, 0, 0x1111, 0);
            cl.Close().expect("Close");
            let hr = list_x::reset_x(backing, alloc.as_raw() as usize, pso2.as_raw() as usize);
            assert!(hr >= 0, "reset_x failed: {hr:#x}");
            assert_eq!(tracked_pso(backing), pso2.as_raw() as usize, "Reset must update bind::State.pso");
            with(backing, |s| assert!(s.layout_gfx.is_none() && s.args_gfx.is_empty()));

            // Release the wrapped title list cleanly (calls bind::forget + list_x::on_release).
            let release_fn: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(*(*(raw_list as *const *const usize)).add(2));
            assert_eq!(release_fn(raw_list), 0);
            let _ = pso_async::released(standin);
        }
    }

    #[test]
    fn dispatch_mesh_signature_tracking_returns_companion_and_layout() {
        let sig = 0x7400_0001usize;
        let companion = 0x7400_0002usize;
        assert_eq!(dispatch_mesh_sig_of(sig), None);
        note_dispatch_mesh_signature(sig, companion, 28, 16);
        assert_eq!(
            dispatch_mesh_sig_of(sig),
            Some(DispatchMeshSig { companion_dispatch: companion, byte_stride: 28, mesh_offset: 16 })
        );
        DISPATCH_MESH_SIGS.lock().unwrap().as_mut().unwrap().remove(&sig);
        assert_eq!(dispatch_mesh_sig_of(sig), None);
    }

    #[test]
    fn two_pass_io_carries_the_unified_memory_base() {
        use crate::gpu::recomp::{IO_UMA_BASE_HI, KERNEL_SGPRS};
        let base = 0x4_0000_0000usize;
        for compute_pass in [true, false] {
            let io = two_pass_io(None, &[], 0x4_1234_5600, compute_pass, base);
            assert_eq!(io.len(), KERNEL_SGPRS);
            assert_eq!(io[IO_UMA_BASE_HI], 4, "compute pass {compute_pass}");
        }
        let io = two_pass_io(None, &[], 0x4_1234_5600, true, base);
        assert_eq!((io[8], io[9]), (0x1234_5600, 4));
    }

    #[test]
    fn two_pass_groups_stay_inside_the_record_buffer() {
        // #304's transform record: 11 908 dwords
        let record = 11_908 * 4;
        assert_eq!(record_groups(32 << 20, record, 1), 704);
        assert_eq!(record_groups(128 << 20, record, 1), 2817);
        // the mesh pass's groups, `split` a record, all in X
        assert_eq!(record_groups(1 << 30, 64, 7), 65_535 / 7);
        assert_eq!(record_groups(1 << 20, 2 << 20, 1), 0);
    }

    /// `SCALE_ARGS_HLSL` on `commands` (dwords each, the DispatchMesh argument at dword 1):
    /// the compute pass's and the mesh pass's commands.
    fn scale_args(commands: &[[u32; 4]], split: u32, cap: u32) -> Option<(Vec<[u32; 4]>, Vec<[u32; 4]>)> {
        scale_args_counted(commands, split, cap, None)
    }

    /// `scale_args` with a count buffer holding `count`
    fn scale_args_counted(commands: &[[u32; 4]], split: u32, cap: u32, count: Option<u32>) -> Option<(Vec<[u32; 4]>, Vec<[u32; 4]>)> {
        use crate::gpu::recomp::runner;
        if runner::adapter().is_err() {
            runner::skip_test("no GPU");
            return None;
        }
        let dxil = crate::gpu::recomp::dxc::compile(SCALE_ARGS_HLSL, "main", "cs_6_6", &[]).unwrap();
        let n = commands.len();
        let (src, dst) = (0u32, 256u32);
        let mut mem: Vec<u32> = commands.iter().flatten().copied().collect();
        mem.resize(64 + 2 * 4 * n, 0xDEAD);
        // the count at dword 60 (u4 is the same memory in the runner)
        assert!(4 * n <= 60);
        mem[60] = count.unwrap_or(0);
        let io = [n as u32, 16, 4, split, cap, src, dst, count.is_some() as u32, 240];
        let out = runner::dispatch(&dxil, [(n as u32).div_ceil(64), 1, 1], &io, &mem).unwrap();
        let cmd = |k: usize| [out[64 + 4 * k], out[65 + 4 * k], out[66 + 4 * k], out[67 + 4 * k]];
        Some(((0..n).map(cmd).collect(), (n..2 * n).map(cmd).collect()))
    }

    #[test]
    fn gpu_indirect_two_pass_groups_stay_inside_the_record_buffer() {
        let Some((compute, mesh)) = scale_args(&[[7, 100, 1, 1], [8, 3, 1, 1], [9, 0, 1, 1]], 2, 10) else { return };
        assert_eq!(compute, [[7, 10, 1, 1], [8, 3, 1, 1], [9, 0, 1, 1]]);
        assert_eq!(mesh, [[7, 20, 1, 1], [8, 6, 1, 1], [9, 0, 1, 1]]);
    }

    #[test]
    fn gpu_indirect_two_pass_groups_go_in_x() {
        let commands = [[1, 1, 2, 1], [2, 2, 3, 4], [3, 300, 300, 1], [4, 5, 0, 2], [5, 70_000, 70_000, 70_000], [6, 1000, 1, 1], [7, 5000, 1, 1], [8, 5000, 1, 0]];
        let Some((compute, mesh)) = scale_args(&commands, 3, 1000) else { return };
        assert_eq!(compute, [[1, 2, 1, 1], [2, 24, 1, 1], [3, 1000, 1, 1], [4, 0, 1, 1], [5, 1000, 1, 1], [6, 1000, 1, 1], [7, 1000, 1, 1], [8, 0, 1, 1]]);
        assert_eq!(mesh, [[1, 6, 1, 1], [2, 72, 1, 1], [3, 3000, 1, 1], [4, 0, 1, 1], [5, 3000, 1, 1], [6, 3000, 1, 1], [7, 3000, 1, 1], [8, 0, 1, 1]]);
    }

    #[test]
    fn freed_two_pass_buffers_kept_for_reuse_are_capped() {
        assert!(pool_keeps(0, 0, 128 << 20));
        assert!(pool_keeps(1, 128 << 20, 128 << 20));
        assert!(!pool_keeps(2, 256 << 20, 4 << 20));
        assert!(!pool_keeps(16, 0, 1 << 20));
    }

    #[test]
    fn gpu_two_pass_argument_scratch_is_shared_and_capped() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        let mut dev: Option<ID3D12Device> = None;
        if unsafe { D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev) }.is_err() {
            crate::gpu::recomp::runner::skip_test("no GPU");
            return;
        }
        let dev = dev.unwrap();
        let list = 0x7500_0001usize as *mut c_void;
        unsafe {
            // small rewrites share one block
            let (a, at_a) = scratch_args(list, &dev, 100).unwrap();
            let (b, at_b) = scratch_args(list, &dev, 300).unwrap();
            assert_eq!((a.as_raw(), at_a), (b.as_raw(), 0));
            assert_eq!(at_b, 256);
            // a recording takes at most SCRATCH_CAP of blocks
            let mut taken = 1;
            while scratch_args(list, &dev, SCRATCH_BLOCK).is_ok() {
                taken += 1;
                assert!(taken <= SCRATCH_CAP / SCRATCH_BLOCK, "past the cap");
            }
            assert!(with(list, |s| s.scratch_taken) <= SCRATCH_CAP);
            // the next recording starts again
            reset(list, 0, 0);
            assert!(scratch_args(list, &dev, SCRATCH_BLOCK).is_ok());
        }
        forget(list);
    }

    #[test]
    fn gpu_indirect_two_pass_commands_past_the_count_get_no_groups() {
        let commands = [[1, 3, 1, 1], [2, 2, 1, 1], [3, 4, 1, 1]];
        let Some((compute, mesh)) = scale_args_counted(&commands, 2, 1000, Some(2)) else { return };
        assert_eq!(compute, [[1, 3, 1, 1], [2, 2, 1, 1], [3, 0, 1, 1]]);
        assert_eq!(mesh, [[1, 6, 1, 1], [2, 4, 1, 1], [3, 0, 1, 1]]);
        let Some((compute, _)) = scale_args_counted(&commands, 2, 1000, Some(7)) else { return };
        assert_eq!(compute, [[1, 3, 1, 1], [2, 2, 1, 1], [3, 4, 1, 1]]);
    }

    /// Commands of one two-pass ExecuteIndirect run in `two_pass_steps` order on the GPU, with
    /// stand-ins for the two passes: the "compute pass" writes each group's record (from record 0)
    /// with its command's constant, the "mesh pass" copies the records it reads out by command.
    #[test]
    fn gpu_two_pass_indirect_commands_keep_their_own_records() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
        let src = r#"
#define RS "RootConstants(num32BitConstants=1, b0), UAV(u0), UAV(u1)"
cbuffer C : register(b0) { uint v; };
RWByteAddressBuffer rec : register(u0);
RWByteAddressBuffer o : register(u1);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void cs(uint3 g : SV_GroupID) { rec.Store(g.x * 4u, v); }
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void ms(uint3 g : SV_GroupID) { o.Store((v * 16u + g.x) * 4u, rec.Load(g.x * 4u)); }
"#;
        // (constant, groups) of each command
        let commands = [(1u32, 3u32), (2, 2), (3, 4)];
        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                crate::gpu::recomp::runner::skip_test("no GPU");
                return;
            }
            let dev = dev.unwrap();
            let cs = crate::gpu::recomp::dxc::compile(src, "cs", "cs_6_6", &[]).unwrap();
            let ms = crate::gpu::recomp::dxc::compile(src, "ms", "cs_6_6", &[]).unwrap();
            let rs: ID3D12RootSignature = dev.CreateRootSignature(0, &cs).unwrap();
            let pso = |dxil: &[u8]| -> ID3D12PipelineState {
                let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&rs),
                    CS: D3D12_SHADER_BYTECODE { pShaderBytecode: dxil.as_ptr() as _, BytecodeLength: dxil.len() },
                    ..Default::default()
                };
                dev.CreateComputePipelineState(&desc).unwrap()
            };
            let (pass1, pass2) = (pso(&cs), pso(&ms));
            let args = [
                D3D12_INDIRECT_ARGUMENT_DESC {
                    Type: D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT,
                    Anonymous: D3D12_INDIRECT_ARGUMENT_DESC_0 { Constant: D3D12_INDIRECT_ARGUMENT_DESC_0_1 { RootParameterIndex: 0, DestOffsetIn32BitValues: 0, Num32BitValuesToSet: 1 } },
                },
                D3D12_INDIRECT_ARGUMENT_DESC { Type: D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH, ..Default::default() },
            ];
            let mut sig: Option<ID3D12CommandSignature> = None;
            dev.CreateCommandSignature(&D3D12_COMMAND_SIGNATURE_DESC { ByteStride: 16, NumArgumentDescs: 2, pArgumentDescs: args.as_ptr(), NodeMask: 0 }, &rs, &mut sig).unwrap();
            let sig = sig.unwrap();
            let buffer = |bytes: u64, cpu: bool| -> ID3D12Resource {
                let desc = D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: bytes,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                    ..Default::default()
                };
                let heap = if cpu {
                    D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_CUSTOM, CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK, MemoryPoolPreference: D3D12_MEMORY_POOL_L0, ..Default::default() }
                } else {
                    D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }
                };
                let mut r: Option<ID3D12Resource> = None;
                dev.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut r).unwrap();
                r.unwrap()
            };
            let (indirect, rec, out) = (buffer(256, true), buffer(256, false), buffer(1024, true));
            let map = |b: &ID3D12Resource| -> *mut u32 {
                let mut p = std::ptr::null_mut();
                b.Map(0, None, Some(&mut p)).unwrap();
                p as *mut u32
            };
            let (a, o) = (map(&indirect), map(&out));
            for (k, &(v, x)) in commands.iter().enumerate() {
                std::ptr::copy_nonoverlapping([v, x, 1, 1].as_ptr(), a.add(4 * k), 4);
            }
            std::ptr::write_bytes(o, 0, 256);
            let ty = D3D12_COMMAND_LIST_TYPE_COMPUTE;
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: ty, ..Default::default() }).unwrap();
            let done: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(ty).unwrap();
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, ty, &alloc, &pass1).unwrap();
            emit_transition(&list, &indirect, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
            list.SetComputeRootSignature(&rs);
            list.SetComputeRootUnorderedAccessView(1, rec.GetGPUVirtualAddress());
            list.SetComputeRootUnorderedAccessView(2, out.GetGPUVirtualAddress());
            for (k, compute) in two_pass_steps(commands.len() as u32) {
                list.SetPipelineState(if compute { &pass1 } else { &pass2 });
                list.ExecuteIndirect(&sig, 1, &indirect, k as u64 * 16, None::<&ID3D12Resource>, 0);
                emit_uav_barrier(&list, Some(&rec));
            }
            emit_transition(&list, &indirect, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_COMMON);
            list.Close().unwrap();
            queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
            queue.Signal(&done, 1).unwrap();
            crate::gpu::fence::wait(&done, 1, &dev, None, 10_000).unwrap();
            for &(v, x) in &commands {
                let got: Vec<u32> = (0..x).map(|g| *o.add((v * 16 + g) as usize)).collect();
                assert_eq!(got, vec![v; x as usize], "the records the mesh pass of command {v} read");
            }
            indirect.Unmap(0, None);
            out.Unmap(0, None);
        }
    }
}
