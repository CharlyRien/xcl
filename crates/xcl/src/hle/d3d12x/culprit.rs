//! `XCL_GPU_CULPRIT=1` (diagnostics, off by default): which GPU operations a stalled queue or a
//! removed device had not finished. Around every operation XCL records on a PC list – compute
//! dispatches (title lists, console-recorded lists, ExecuteIndirect), draws, ExecuteIndirect of
//! either flavour, copies, resolves and clears – the list writes a running number into a readback
//! buffer with `WriteBufferImmediate`: `MARKER_IN` before (the work before it has started),
//! `MARKER_OUT` after (the work before it is done), one pair of words per PC list. Operations the
//! title recorded that never reach the GPU (a dispatch or draw with no translated pipeline) get a
//! number without markers, so the log shows where they were in the list.
//!
//! When the GPU-wait watchdog releases a wait, a list whose last number in is above its last
//! number out has work that started and is not done: those operations are logged with their
//! console shaders (`fnv64` of the title's bytecode, as in the `[SHADER]` lines and the name of
//! the file in the translated-shader cache), their pipeline and their size. When the device is
//! removed, the same once more, then the last submissions of each queue: for each of their lists
//! not finished, in record order, the last operation whose marker out is there and the ones after
//! it (the first of them sent to the GPU: in flight). Markers in are secondary: on NVIDIA they
//! were seen to lag behind the markers out.
//!
//! DRED breadcrumbs would say the same on a device removal, but they change the driver's
//! pipeline cache keys: every pipeline is built again, and a retail title's start then takes
//! minutes.

use std::collections::{HashMap, VecDeque};
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;

pub fn enabled() -> bool {
    static E: OnceLock<bool> = OnceLock::new();
    // (LOCAL: `XCL_LOCAL_GPU_TIMES` times the marked operations: `gpu_times`)
    *E.get_or_init(|| std::env::var("XCL_GPU_CULPRIT").is_ok_and(|v| v.trim() == "1") || super::gpu_times::enabled())
}

/// Lists with a slot in the readback buffer (two words each).
const SLOTS: usize = 1 << 14;
/// Operations remembered by number (the oldest are forgotten): about 64 bytes each.
const REMEMBERED: usize = 1 << 19;
/// Submissions remembered for the report at a device removal.
const SUBMISSIONS: usize = 256;
/// Of those, the last ones of each queue that are reported.
const REPORTED_PER_QUEUE: usize = 6;

/// The console CS of the compute pipelines the title made, by the pointer it holds.
static CS: Mutex<Option<HashMap<usize, u64>>> = Mutex::new(None);
/// The console VS and PS of the graphics pipelines the title made.
static GFX: Mutex<Option<HashMap<usize, [u64; 2]>>> = Mutex::new(None);
/// The type of the last argument of each command signature (what one ExecuteIndirect record does).
static SIGNATURES: Mutex<Option<HashMap<usize, u32>>> = Mutex::new(None);

/// `fnv64` of a shader's bytes (0: none, or not readable).
unsafe fn shader_hash(b: &D3D12_SHADER_BYTECODE) -> u64 {
    let (p, n) = (b.pShaderBytecode as usize, b.BytecodeLength);
    if p == 0 || n == 0 || !crate::gpu::shader::readable(p, n) {
        return 0;
    }
    crate::gpu::shader::fnv64(std::slice::from_raw_parts(p as *const u8, n))
}

/// A compute pipeline the title made: its console CS, for the log.
pub unsafe fn note_compute(pso: usize, desc: *const c_void) {
    // LOCAL: also for `local_inputs` (`XCL_LOCAL_DUMP_CS_INPUTS`)
    if !(enabled() || super::local_inputs::active()) || pso == 0 || !crate::gpu::shader::readable(desc as usize, std::mem::size_of::<D3D12_COMPUTE_PIPELINE_STATE_DESC>()) {
        return;
    }
    let d = std::ptr::read(desc as *const D3D12_COMPUTE_PIPELINE_STATE_DESC);
    let h = shader_hash(&d.CS);
    if h != 0 {
        CS.lock().unwrap().get_or_insert_with(HashMap::new).insert(pso, h);
    }
}

/// A graphics pipeline the title made (its desc, or a stream's as one): its VS and PS.
pub unsafe fn note_graphics(pso: usize, desc: *const c_void) {
    if !enabled() || pso == 0 || !crate::gpu::shader::readable(desc as usize, std::mem::size_of::<D3D12_GRAPHICS_PIPELINE_STATE_DESC>()) {
        return;
    }
    let d = std::ptr::read(desc as *const D3D12_GRAPHICS_PIPELINE_STATE_DESC);
    let h = [shader_hash(&d.VS), shader_hash(&d.PS)];
    std::mem::forget(d); // borrowed root signature and strings
    GFX.lock().unwrap().get_or_insert_with(HashMap::new).insert(pso, h);
}

/// A command signature the title made: `last` is the type of its last argument.
pub fn note_signature(sig: usize, last: Option<u32>) {
    if enabled() && sig != 0 {
        if let Some(t) = last {
            SIGNATURES.lock().unwrap().get_or_insert_with(HashMap::new).insert(sig, t);
        }
    }
}

/// The title's pipeline, or the one it stands for (a background build).
fn lookup<T: Copy>(map: &Mutex<Option<HashMap<usize, T>>>, title_pso: usize, pso: usize) -> Option<T> {
    let g = map.lock().unwrap();
    let m = g.as_ref()?;
    [title_pso, pso, super::pso_async::resolve(title_pso)].iter().find_map(|p| m.get(p)).copied()
}

fn cs_of(title_pso: usize, pso: usize) -> u64 {
    lookup(&CS, title_pso, pso).unwrap_or(0)
}

/// LOCAL: the console CS of a compute pipeline (0: not known), for `local_inputs`.
pub fn cs_hash(title_pso: usize, pso: usize) -> u64 {
    cs_of(title_pso, pso)
}

/// How an operation is described.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub enum Shape {
    /// copies, resolves, clears: the resources
    #[default]
    Bulk,
    /// a dispatch: CS, groups
    Groups,
    /// a draw: VS, PS, vertices or indices, instances
    Draw,
    /// ExecuteIndirect: up to `args[0]` records, `args[1]` 1 with a count buffer
    Indirect,
}

/// What an operation was, for the log.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct Op {
    /// the method (and its flavour)
    pub kind: &'static str,
    /// the pipeline bound: the title's object (0: not known)
    pub pso: usize,
    /// console shaders: the CS, or the VS and the PS
    pub shaders: [u64; 2],
    pub args: [u32; 3],
    /// the destination and the source of a copy
    pub res: [usize; 2],
    pub shape: Shape,
}

impl std::fmt::Display for Op {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let [a, b, c] = self.args;
        let [h0, h1] = self.shaders;
        match self.shape {
            Shape::Groups => write!(f, "{}: cs {h0:016x}, pipeline {:#x}, groups {a}x{b}x{c}", self.kind, self.pso),
            Shape::Draw => write!(f, "{}: vs {h0:016x} ps {h1:016x}, pipeline {:#x}, {a} vertices or indices x {b} instances", self.kind, self.pso),
            Shape::Indirect => write!(f, "{}: shaders {h0:016x} {h1:016x}, pipeline {:#x}, up to {a} records{}", self.kind, self.pso, if b != 0 { ", count from a buffer" } else { "" }),
            Shape::Bulk => write!(f, "{}: {:#x} <- {:#x}{}", self.kind, self.res[0], self.res[1], if a != 0 { format!(", {a} bytes") } else { String::new() }),
        }
    }
}

/// A dispatch with `pso` (the title's `title_pso`); `groups` `[0, 0, n]`: an indirect one of at
/// most `n` dispatches.
pub fn compute_op(kind: &'static str, title_pso: usize, pso: usize, groups: [u32; 3]) -> Op {
    let shape = if groups[..2] == [0, 0] { Shape::Indirect } else { Shape::Groups };
    let args = if shape == Shape::Indirect { [groups[2], 0, 0] } else { groups };
    Op { kind, pso: if title_pso != 0 { title_pso } else { pso }, shaders: [cs_of(title_pso, pso), 0], args, res: [0; 2], shape }
}

/// A draw on `list` with the pipeline XCL keeps as bound there.
pub fn draw_op(kind: &'static str, list: *mut c_void, count: u32, instances: u32) -> Op {
    let pso = super::bind::tracked_pso(list);
    let shaders = lookup(&GFX, pso, super::pso_async::resolve(pso)).unwrap_or([0; 2]);
    Op { kind, pso, shaders, args: [count, instances, 0], res: [0; 2], shape: Shape::Draw }
}

/// ExecuteIndirect on `list` with the command signature `sig`: its flavour from the signature.
pub fn indirect_op(list: *mut c_void, sig: usize, max: u32, count_buffer: bool) -> Op {
    let kind = match SIGNATURES.lock().unwrap().as_ref().and_then(|m| m.get(&sig)).copied() {
        Some(0) => "ExecuteIndirect (draws)",
        Some(1) => "ExecuteIndirect (indexed draws)",
        Some(2) => "ExecuteIndirect (dispatches)",
        Some(10) => "ExecuteIndirect (DispatchRays)",
        Some(11) => "ExecuteIndirect (mesh dispatches)",
        Some(_) => "ExecuteIndirect (other records)",
        None => "ExecuteIndirect (signature not seen)",
    };
    let pso = super::bind::tracked_pso(list);
    let real = super::pso_async::resolve(pso);
    let shaders = lookup(&GFX, pso, real).unwrap_or([cs_of(pso, real), 0]);
    Op { kind, pso, shaders, args: [max, count_buffer as u32, 0], res: [0; 2], shape: Shape::Indirect }
}

/// A copy, resolve or clear (`bytes`: 0 when not known).
pub fn bulk_op(kind: &'static str, dst: usize, src: usize, bytes: u64) -> Op {
    Op { kind, pso: 0, shaders: [0; 2], args: [bytes.min(u32::MAX as u64) as u32, 0, 0], res: [dst, src], shape: Shape::Bulk }
}

/// An operation recorded on a list.
#[derive(Clone, Copy, Debug)]
struct Rec {
    list: usize,
    op: Op,
    /// with markers around it (false: recorded by the title, not sent to the GPU)
    marked: bool,
}

/// The lists one ExecuteCommandLists sent, with the numbers of their operations.
struct Submission {
    queue: usize,
    /// this queue's submission number (from 1)
    seq: u64,
    lists: Vec<(usize, Arc<[u32]>)>,
}

struct Markers {
    _buf: ID3D12Resource,
    gpu: u64,
    cpu: *const u32,
    /// PC list → its slot
    slots: HashMap<usize, u32>,
    /// operation `first + k` is `ops[k]`
    ops: VecDeque<Rec>,
    first: u32,
    /// list → its operations since it was last submitted
    recording: HashMap<usize, Vec<u32>>,
    /// list → the operations of the recording it last submitted
    recorded: HashMap<usize, Arc<[u32]>>,
    subs: VecDeque<Submission>,
    /// queue → submissions so far
    seqs: HashMap<usize, u64>,
    /// LOCAL: `XCL_LOCAL_GPU_TIMES`
    times: Option<super::gpu_times::Times>,
}
// SAFETY: `cpu` is the persistent map of `_buf`, read only under the mutex
unsafe impl Send for Markers {}

impl Markers {
    fn get(&self, id: u32) -> Option<&Rec> {
        id.checked_sub(self.first).and_then(|k| self.ops.get(k as usize))
    }

    fn push(&mut self, r: Rec) -> u32 {
        if self.ops.len() >= REMEMBERED {
            self.ops.pop_front();
            self.first += 1;
        }
        self.ops.push_back(r);
        self.first + self.ops.len() as u32 - 1
    }

    /// The last numbers in and out of `list`'s slot.
    fn words(&self, list: usize) -> Option<(u32, u32)> {
        let s = *self.slots.get(&list)? as usize;
        // SAFETY: slots index the mapped buffer of SLOTS pairs (see `readable_after_removal`)
        unsafe { Some((std::ptr::read_volatile(self.cpu.add(s * 2)), std::ptr::read_volatile(self.cpu.add(s * 2 + 1)))) }
    }
}

static MARKERS: Mutex<Option<Markers>> = Mutex::new(None);

unsafe fn make_markers(dev: &ID3D12Device) -> Option<Markers> {
    let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: (SLOTS * 8) as u64,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    let mut buf: Option<ID3D12Resource> = None;
    if let Err(e) = dev.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut buf) {
        crate::xlog_warn!("GPU culprit: no readback buffer for the markers: {e}");
        return None;
    }
    let buf = buf?;
    let mut p: *mut c_void = std::ptr::null_mut();
    buf.Map(0, None, Some(&mut p)).ok()?;
    crate::xlog_info!("GPU culprit: markers around dispatches, draws, ExecuteIndirect, copies, resolves and clears ({SLOTS} list slots)");
    Some(Markers {
        gpu: buf.GetGPUVirtualAddress(),
        _buf: buf,
        cpu: p as *const u32,
        slots: HashMap::new(),
        ops: VecDeque::new(),
        first: 1,
        recording: HashMap::new(),
        recorded: HashMap::new(),
        subs: VecDeque::new(),
        seqs: HashMap::new(),
        times: if super::gpu_times::enabled() { start_times(dev) } else { None },
    })
}

/// LOCAL: the timestamps (`gpu_times`) and the thread that adds them up.
unsafe fn start_times(dev: &ID3D12Device) -> Option<super::gpu_times::Times> {
    let t = super::gpu_times::Times::new(dev)?;
    let _ = std::thread::Builder::new().name("xcl-gpu-times".into()).spawn(|| loop {
        std::thread::sleep(std::time::Duration::from_millis(500));
        with_times(|t, _| {
            if t.due() {
                t.report("");
            }
        });
    });
    Some(t)
}

/// `f` with the timestamps, after adding up the resolved ones (`op` describes an operation).
fn with_times(f: impl FnOnce(&mut super::gpu_times::Times, &dyn Fn(u32) -> Option<Op>)) {
    let mut g = MARKERS.lock().unwrap();
    let Some(m) = g.as_mut() else { return };
    let Some(mut t) = m.times.take() else { return };
    let op = |id: u32| m.get(id).map(|r| r.op);
    t.collect(op);
    f(&mut t, &op);
    m.times = Some(t);
}

/// LOCAL: the process ends – the last `GPU times` report.
pub fn times_at_exit() {
    with_times(|t, _| t.report(" at exit"));
}

/// `op` recorded on `list`, with a marker in before it when `marked`: its number (0: none).
unsafe fn record(list: *mut c_void, op: Op, marked: bool) -> u32 {
    let Some(l) = ID3D12GraphicsCommandList::from_raw_borrowed(&list) else { return 0 };
    // not on copy lists (the DMA lists' copies): no shader runs there, and markers there are
    // not needed to find a hang
    if l.GetType() == D3D12_COMMAND_LIST_TYPE_COPY {
        return 0;
    }
    let Ok(l2) = l.cast::<ID3D12GraphicsCommandList2>() else { return 0 };
    let mut g = MARKERS.lock().unwrap();
    if g.is_none() {
        let mut dev: Option<ID3D12Device> = None;
        if l.GetDevice(&mut dev).is_err() {
            return 0;
        }
        *g = make_markers(&dev.unwrap());
    }
    let Some(m) = g.as_mut() else { return 0 };
    let key = list as usize;
    let Some(slot) = slot_for(&mut m.slots, key, SLOTS) else { return 0 };
    let id = m.push(Rec { list: key, op, marked });
    m.recording.entry(key).or_default().push(id);
    if marked {
        let p = D3D12_WRITEBUFFERIMMEDIATE_PARAMETER { Dest: m.gpu + slot as u64 * 8, Value: id };
        l2.WriteBufferImmediate(1, &p, Some(&D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_IN));
        if let Some(t) = m.times.as_mut() {
            t.start(l, id);
        }
    }
    id
}

/// Before `op` on `list`: a marker in with a new number, which the caller hands to `end` once
/// the operation is recorded (0: no marker; nothing when the diagnostic is off).
pub unsafe fn begin(list: *mut c_void, op: impl FnOnce() -> Op) -> u32 {
    if !enabled() {
        return 0;
    }
    record(list, op(), true)
}

/// After the operation: a marker out with its number.
pub unsafe fn end(list: *mut c_void, id: u32) {
    if id == 0 {
        return;
    }
    let Some(l) = ID3D12GraphicsCommandList::from_raw_borrowed(&list) else { return };
    let Ok(l2) = l.cast::<ID3D12GraphicsCommandList2>() else { return };
    let mut g = MARKERS.lock().unwrap();
    let Some(m) = g.as_mut() else { return };
    let Some(&slot) = m.slots.get(&(list as usize)) else { return };
    if let Some(t) = m.times.as_mut() {
        t.stop(l, id);
    }
    let p = D3D12_WRITEBUFFERIMMEDIATE_PARAMETER { Dest: m.gpu + slot as u64 * 8 + 4, Value: id };
    l2.WriteBufferImmediate(1, &p, Some(&D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_OUT));
}

/// The slot of `list`: its own, or the next free one; none once `cap` lists have one (the list is
/// then not recorded, so every recorded slot stays inside the readback buffer).
fn slot_for(slots: &mut HashMap<usize, u32>, list: usize, cap: usize) -> Option<u32> {
    if let Some(&s) = slots.get(&list) {
        return Some(s);
    }
    let n = slots.len();
    (n < cap).then(|| {
        slots.insert(list, n as u32);
        n as u32
    })
}

/// An operation the title recorded that XCL does not send to the GPU: numbered, no markers.
pub unsafe fn skipped(list: *mut c_void, op: impl FnOnce() -> Op) {
    if enabled() {
        record(list, op(), false);
    }
}

/// Before a dispatch with `pso` (the title's `title_pso`) on `list`: `begin`.
pub unsafe fn before_dispatch(list: &ID3D12GraphicsCommandList, title_pso: usize, pso: usize, groups: [u32; 3]) -> u32 {
    begin(list.as_raw(), || compute_op(if groups[..2] == [0, 0] { "ExecuteIndirect (dispatches)" } else { "Dispatch" }, title_pso, pso, groups))
}

/// After the dispatch: `end`.
pub unsafe fn after_dispatch(list: &ID3D12GraphicsCommandList, id: u32) {
    end(list.as_raw(), id)
}

/// `lists` went to the GPU on `queue` (one PC ExecuteCommandLists): what each recorded since
/// its last submission is this submission's.
pub fn note_submit(queue: *mut c_void, lists: &[*mut c_void]) {
    if !enabled() {
        return;
    }
    let mut g = MARKERS.lock().unwrap();
    let Some(m) = g.as_mut() else { return };
    let q = queue as usize;
    let seq = {
        let s = m.seqs.entry(q).or_default();
        *s += 1;
        *s
    };
    let mut sub = Submission { queue: q, seq, lists: Vec::with_capacity(lists.len()) };
    for &l in lists {
        let key = l as usize;
        if let Some(ids) = m.recording.remove(&key).filter(|v| !v.is_empty()) {
            m.recorded.insert(key, ids.into());
        }
        sub.lists.push((key, m.recorded.get(&key).cloned().unwrap_or_else(|| Arc::from([]))));
    }
    if let Some(t) = m.times.as_mut() {
        for (_, ids) in &sub.lists {
            // SAFETY: `queue` is the queue the caller just submitted to
            unsafe { t.submitted(queue, ids) };
        }
    }
    if m.subs.len() >= SUBMISSIONS {
        m.subs.pop_front();
    }
    m.subs.push_back(sub);
}

/// The lists with work started and not done: (list, last number in, last number out), from
/// each slot's pair of words. A marker in only says that the work before it has started, so a
/// list counts when its number in is above its number out (in and out written out of order
/// otherwise look open).
fn open_lists(slots: &HashMap<usize, u32>, words: impl Fn(u32) -> (u32, u32)) -> Vec<(usize, u32, u32)> {
    let mut open: Vec<(usize, u32, u32)> = slots
        .iter()
        .filter_map(|(&list, &slot)| {
            let (i, o) = words(slot);
            (i > o).then_some((list, i, o))
        })
        .collect();
    open.sort_by_key(|&(_, i, _)| i);
    open
}

/// Operations shown after the last marker out.
const AFTER: usize = 4;

/// Where a submitted recording got to.
#[derive(Debug, PartialEq, Eq)]
enum Progress {
    /// nothing marked in it
    Unmarked,
    /// its last marker out is there
    Done,
    /// it stopped: see `Frontier`
    Stopped(Frontier),
}

/// Where a recording stopped, in record order (indices into its `ids`): `done` the last
/// operation whose marker out is there (None: none of this recording's), `after` the next
/// `AFTER` operations, `in_flight` the first operation after `done` that was sent to the GPU.
///
/// Only the markers out are used. A marker in is not a reliable "started" on every driver: on
/// NVIDIA the operation that hung was the one after the last marker out, its marker in never
/// visible, and lists showed a number in below their number out.
#[derive(Debug, PartialEq, Eq)]
struct Frontier {
    done: Option<usize>,
    after: Vec<usize>,
    in_flight: Option<usize>,
}

/// `ids`: a recording's operations in record order (their numbers grow with it; `marked`:
/// which have markers); `out`: its list's last number out. Numbers grow over time, so a number
/// from an older recording of the list is below its first and one of a newer one above its last.
fn progress(ids: &[u32], marked: impl Fn(u32) -> bool, out: u32) -> Progress {
    let Some(last) = ids.iter().copied().filter(|&id| marked(id)).last() else { return Progress::Unmarked };
    if out >= last {
        return Progress::Done;
    }
    let done = ids.iter().rposition(|&id| marked(id) && id <= out);
    let start = done.map_or(0, |k| k + 1);
    Progress::Stopped(Frontier {
        done,
        after: (start..ids.len()).take(AFTER).collect(),
        in_flight: (start..ids.len()).find(|&k| marked(ids[k])),
    })
}

/// The operations of `list` numbered after `out` up to `in_` that are still remembered.
fn not_done(m: &Markers, list: usize, in_: u32, out: u32) -> Vec<(u32, Rec)> {
    ((out + 1)..=in_).filter_map(|id| m.get(id).filter(|r| r.list == list).map(|r| (id, *r))).collect()
}

fn describe(m: &Markers, id: u32) -> String {
    match m.get(id) {
        Some(r) if r.marked => format!("#{id} {}", r.op),
        Some(r) => format!("#{id} {} (not sent to the GPU)", r.op),
        None => format!("#{id} (forgotten)"),
    }
}

/// The lists with work started and not done, each with its operations after its last marker out.
fn report_open(m: &Markers, why: &str) {
    let open = open_lists(&m.slots, |s| unsafe { (std::ptr::read_volatile(m.cpu.add(s as usize * 2)), std::ptr::read_volatile(m.cpu.add(s as usize * 2 + 1))) });
    crate::xlog_warn!("GPU culprit{why}: {} of {} lists have work started and not done", open.len(), m.slots.len());
    for (list, i, o) in open.iter().take(64) {
        let left = not_done(m, *list, *i, *o);
        crate::xlog_warn!("GPU culprit{why}: list {list:#x}: marker in #{i}, last marker out #{o}; {} of its operations not known done:", left.len());
        for (id, _) in left.iter().take(64) {
            crate::xlog_warn!("GPU culprit{why}:   {}", describe(m, *id));
        }
    }
}

/// The watchdog released GPU waits: log the operations started and not done (once per burst,
/// 4 s apart, at most 64 times).
pub fn on_watchdog_release() {
    static BURSTS: AtomicU32 = AtomicU32::new(0);
    static LAST: Mutex<Option<std::time::Instant>> = Mutex::new(None);
    if !enabled() {
        return;
    }
    {
        let mut l = LAST.lock().unwrap();
        if l.is_some_and(|t| t.elapsed() < std::time::Duration::from_secs(4)) {
            return;
        }
        *l = Some(std::time::Instant::now());
    }
    if BURSTS.fetch_add(1, Ordering::Relaxed) >= 64 {
        return;
    }
    let g = MARKERS.lock().unwrap();
    let Some(m) = g.as_ref() else { return };
    report_open(m, "");
}

/// The readback buffer is still readable once the device is removed: it is CPU memory of a
/// READBACK heap, mapped for the process's life (`_buf` is never released), and a removal does
/// not unmap it – it holds what the GPU wrote before the reset. Checked anyway: the report must
/// not fault in the crash path.
fn readable_after_removal(m: &Markers) -> bool {
    crate::gpu::shader::readable(m.cpu as usize, SLOTS * 8)
}

/// The device was removed (called once, from `real::log_device_removed`): the lists with work in
/// flight, then for the last submissions of each queue where each of their lists got to.
pub fn on_device_removed() {
    static DONE: AtomicBool = AtomicBool::new(false);
    if !enabled() || DONE.swap(true, Ordering::AcqRel) {
        return;
    }
    // LOCAL: the times so far, and the operations that started and did not finish
    with_times(|t, op| {
        t.unfinished(op);
        t.report(" at removal");
    });
    let g = MARKERS.lock().unwrap();
    let Some(m) = g.as_ref() else {
        crate::xlog_error!("GPU culprit at removal: no operation was marked");
        return;
    };
    if !readable_after_removal(m) {
        crate::xlog_error!("GPU culprit at removal: the marker buffer is not readable");
        return;
    }
    let why = " at removal";
    report_open(m, why);
    let mut queues: Vec<usize> = m.subs.iter().map(|s| s.queue).collect();
    queues.sort_unstable();
    queues.dedup();
    for q in queues {
        let subs: Vec<&Submission> = m.subs.iter().filter(|s| s.queue == q).collect();
        let marked = |id: u32| m.get(id).is_some_and(|r| r.marked);
        let state = |s: &Submission| -> Vec<Progress> { s.lists.iter().map(|(l, ids)| m.words(*l).map_or(Progress::Unmarked, |(_, o)| progress(ids, marked, o))).collect() };
        let finished = |s: &Submission| state(s).iter().all(|p| matches!(p, Progress::Done | Progress::Unmarked));
        let last = subs.last().map_or(0, |s| s.seq);
        let done = subs.iter().rev().find(|s| finished(s)).map_or(0, |s| s.seq);
        let first_open = subs.iter().find(|s| s.seq > done && !finished(s)).map_or(0, |s| s.seq);
        crate::xlog_error!(
            "GPU culprit{why}: queue q{} ({q:#x}): submission #{last} the last sent, #{done} the last with all its marked work done, #{first_open} the first not done",
            super::list_x::queue_id(q)
        );
        // from the submission before the first one not done (the last ones when all are done)
        let from = if first_open == 0 { subs.len().saturating_sub(REPORTED_PER_QUEUE) } else { subs.iter().position(|s| s.seq == first_open).unwrap_or(0).saturating_sub(1) };
        for s in subs.iter().skip(from).take(REPORTED_PER_QUEUE) {
            for ((list, ids), p) in s.lists.iter().zip(state(s)) {
                let (in_, out) = m.words(*list).unwrap_or((0, 0));
                let n = ids.len();
                let head = format!("GPU culprit{why}:   q{} submission #{} list {list:#x} ({n} operations, out #{out}, in #{in_})", super::list_x::queue_id(q), s.seq);
                match p {
                    Progress::Unmarked => crate::xlog_error!("{head}: nothing marked"),
                    Progress::Done => crate::xlog_error!("{head}: done (last {})", ids.last().map_or(String::new(), |&id| describe(m, id))),
                    Progress::Stopped(f) => {
                        match f.done {
                            Some(k) => crate::xlog_error!("{head}: not finished; last out marker at op {} of {n}: {}", k + 1, describe(m, ids[k])),
                            None => crate::xlog_error!("{head}: not finished; no out marker of this recording (nothing of it known done)"),
                        }
                        let tag = |k: usize| if Some(k) == f.in_flight { "  <- in flight (first sent op after the last out marker)" } else { "" };
                        for &k in &f.after {
                            crate::xlog_error!("GPU culprit{why}:     after it, op {} of {n}: {}{}", k + 1, describe(m, ids[k]), tag(k));
                        }
                        match f.in_flight {
                            Some(k) if !f.after.contains(&k) => crate::xlog_error!("GPU culprit{why}:     op {} of {n}: {}{}", k + 1, describe(m, ids[k]), tag(k)),
                            None => crate::xlog_error!("GPU culprit{why}:     no sent op after the last out marker"),
                            _ => {}
                        }
                        // secondary: where the marker in points (not reliable on every driver)
                        match ids.iter().position(|&id| id == in_) {
                            Some(k) => crate::xlog_error!("GPU culprit{why}:     (marker in #{in_} = op {} of {n})", k + 1),
                            None => crate::xlog_error!("GPU culprit{why}:     (marker in #{in_}: not an op of this recording)"),
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Only a list whose number in is above its number out has work in flight.
    #[test]
    fn a_list_with_a_marker_in_above_its_marker_out_is_open() {
        let slots: HashMap<usize, u32> = [(0xA, 0), (0xB, 1), (0xC, 2), (0xD, 3)].into_iter().collect();
        // A: done; B: 12 started, 10 done; C: in and out written out of order; D: never ran
        let words = |s: u32| [(9, 9), (12, 10), (20, 21), (0, 0)][s as usize];
        assert_eq!(open_lists(&slots, words), vec![(0xB, 12, 10)]);
    }

    /// A submitted recording against its list's last marker out, in record order: done, nothing
    /// of it done (an older recording's number), or stopped after an operation – the next ones
    /// listed, the first sent one in flight. The marker in plays no part.
    #[test]
    fn a_recording_stops_after_its_last_marker_out_in_record_order() {
        // 23 was recorded but not sent to the GPU (no markers)
        let ids = [21, 22, 23, 24, 25, 26];
        let marked = |id: u32| id != 23;
        let stopped = |done, after: &[usize], in_flight| Progress::Stopped(Frontier { done, after: after.to_vec(), in_flight });
        assert_eq!(progress(&ids, marked, 26), Progress::Done);
        assert_eq!(progress(&ids, marked, 40), Progress::Done, "a newer recording ran after it");
        assert_eq!(progress(&ids, marked, 12), stopped(None, &[0, 1, 2, 3], Some(0)), "an older recording's number");
        assert_eq!(progress(&ids, marked, 0), stopped(None, &[0, 1, 2, 3], Some(0)));
        // stopped after 22: 23 (not sent) then 24, the first sent one
        assert_eq!(progress(&ids, marked, 22), stopped(Some(1), &[2, 3, 4, 5], Some(3)));
        assert_eq!(progress(&ids, marked, 25), stopped(Some(4), &[5], Some(5)));
        assert_eq!(progress(&[23], marked, 30), Progress::Unmarked);
        assert_eq!(progress(&[], marked, 30), Progress::Unmarked);
        // only unsent operations after the last marker out: none in flight
        assert_eq!(progress(&[21, 23], marked, 21), Progress::Done);
    }

    /// Operations are found by number until they are forgotten, and each is logged as its kind.
    #[test]
    fn operations_are_kept_by_number_and_described() {
        let mut ops = VecDeque::new();
        let mut first = 1u32;
        let push = |ops: &mut VecDeque<Rec>, first: &mut u32, r: Rec| {
            if ops.len() >= 3 {
                ops.pop_front();
                *first += 1;
            }
            ops.push_back(r);
            *first + ops.len() as u32 - 1
        };
        let draw = Op { kind: "DrawIndexedInstanced", pso: 0x40, shaders: [0x1234, 0x5678], args: [36, 2, 0], res: [0; 2], shape: Shape::Draw };
        let disp = Op { kind: "Dispatch", pso: 0x50, shaders: [0xabc, 0], args: [16, 16, 16], res: [0; 2], shape: Shape::Groups };
        let ids: Vec<u32> = [draw, disp, draw, disp].iter().map(|&op| push(&mut ops, &mut first, Rec { list: 7, op, marked: true })).collect();
        assert_eq!(ids, vec![1, 2, 3, 4]);
        assert_eq!(first, 2, "the oldest is forgotten");
        let get = |id: u32| id.checked_sub(first).and_then(|k| ops.get(k as usize));
        assert!(get(1).is_none());
        assert_eq!(get(4).map(|r| r.op), Some(disp));
        assert_eq!(draw.to_string(), "DrawIndexedInstanced: vs 0000000000001234 ps 0000000000005678, pipeline 0x40, 36 vertices or indices x 2 instances");
        assert_eq!(disp.to_string(), "Dispatch: cs 0000000000000abc, pipeline 0x50, groups 16x16x16");
        assert_eq!(compute_op("x", 0, 0, [0, 0, 8]).shape, Shape::Indirect);
        assert_eq!(bulk_op("CopyBufferRegion", 0x10, 0x20, 256).to_string(), "CopyBufferRegion: 0x10 <- 0x20, 256 bytes");
    }

    /// Once every slot is taken, a new list gets none and is not recorded, and the lists that
    /// have one keep it: no slot past the readback buffer is ever read.
    #[test]
    fn a_list_past_the_last_slot_gets_none() {
        let mut slots = HashMap::new();
        assert_eq!(slot_for(&mut slots, 0xA, 2), Some(0));
        assert_eq!(slot_for(&mut slots, 0xB, 2), Some(1));
        assert_eq!(slot_for(&mut slots, 0xC, 2), None);
        assert_eq!(slots.len(), 2, "the list without a slot is not recorded");
        assert_eq!(slot_for(&mut slots, 0xA, 2), Some(0));
        assert!(slots.values().all(|&s| (s as usize) < 2));
    }

    /// The markers reach the readback buffer: a list run to its end leaves its numbers in and
    /// out equal, so nothing is reported for it.
    #[test]
    fn gpu_markers_of_a_finished_list_match() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                crate::gpu::recomp::runner::skip_test("no GPU");
                return;
            }
            let dev = dev.unwrap();
            let Some(m) = make_markers(&dev) else { panic!("readback buffer") };
            let ty = D3D12_COMMAND_LIST_TYPE_COMPUTE;
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: ty, ..Default::default() }).unwrap();
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(ty).unwrap();
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, ty, &alloc, None).unwrap();
            let l2: ID3D12GraphicsCommandList2 = list.cast().unwrap();
            for (i, mode) in [(0u64, D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_IN), (4, D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_OUT)] {
                let p = D3D12_WRITEBUFFERIMMEDIATE_PARAMETER { Dest: m.gpu + 8 + i, Value: 77 };
                l2.WriteBufferImmediate(1, &p, Some(&mode));
            }
            list.Close().unwrap();
            queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            queue.Signal(&fence, 1).unwrap();
            let t = std::time::Instant::now();
            while fence.GetCompletedValue() < 1 {
                assert!(t.elapsed().as_secs() < 10, "the list did not finish");
                std::thread::yield_now();
            }
            let words = |s: u32| (std::ptr::read_volatile(m.cpu.add(s as usize * 2)), std::ptr::read_volatile(m.cpu.add(s as usize * 2 + 1)));
            assert_eq!(words(1), (77, 77));
            let slots: HashMap<usize, u32> = [(list.as_raw() as usize, 1)].into_iter().collect();
            assert!(open_lists(&slots, words).is_empty());
            assert!(readable_after_removal(&m));
        }
    }
}
