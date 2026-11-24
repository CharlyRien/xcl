//! LOCAL DIAGNOSTIC, not for upstream: `XCL_LOCAL_GPU_TIMES=1` – GPU time of each operation
//! `culprit` marks (it turns those markers on), from D3D12 timestamp queries: one before and one
//! after the operation, each resolved at once into a readback buffer. A thread adds up the
//! resolved ones every 500 ms; every 30 s, at the process's end and at a device removal the
//! programs that took the most GPU time are logged (`GPU times:`), and an operation of more than
//! 1 s at once (`GPU times: slow op`).
//!
//! Times are per queue: a timestamp says when that queue reached that point. On one queue the
//! operations do not overlap (the culprit's marker out after each waits for it to finish), so an
//! operation's time is how long its queue spent on it – including what other queues ran on the
//! GPU at the same time. Summed over queues, times can exceed the wall clock.
//!
//! Bounded: a ring of `OPS` query pairs; a pair is free again once both of its values are read.
//! An operation whose pair is still held is not timed (counted as "not timed"). Copy lists take
//! no timestamps here (`culprit::record` leaves them out).

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;

use super::culprit::{Op, Shape};

pub fn enabled() -> bool {
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| std::env::var("XCL_LOCAL_GPU_TIMES").is_ok_and(|v| v.trim() == "1"))
}

/// Operations timed at once (two queries, 16 bytes of readback each).
pub const OPS: usize = 1 << 17;
const TOP: usize = 25;
const SLOW_MS: f64 = 1000.0;
const SLOW_LINES: u32 = 200;
pub const PERIOD: Duration = Duration::from_secs(30);

/// One program's operations in a report period.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Stat {
    pub n: u64,
    pub total_ms: f64,
    pub max_ms: f64,
    /// the arguments (groups, counts) of the longest one
    pub max_args: [u32; 3],
}

/// GPU time by program: kind and console shaders (a CS, or a VS and a PS).
#[derive(Default)]
pub struct Agg {
    by: HashMap<(&'static str, [u64; 2]), (Op, Stat)>,
}

impl Agg {
    pub fn add(&mut self, op: &Op, ms: f64) {
        let (_, s) = self.by.entry((op.kind, op.shaders)).or_insert((*op, Stat::default()));
        s.n += 1;
        s.total_ms += ms;
        if ms >= s.max_ms {
            s.max_ms = ms;
            s.max_args = op.args;
        }
    }

    /// The `n` programs with the most time, most first.
    pub fn top(&self, n: usize) -> Vec<(Op, Stat)> {
        let mut v: Vec<(Op, Stat)> = self.by.values().copied().collect();
        v.sort_by(|a, b| b.1.total_ms.total_cmp(&a.1.total_ms));
        v.truncate(n);
        v
    }

    pub fn total_ms(&self) -> f64 {
        self.by.values().map(|(_, s)| s.total_ms).sum()
    }

    pub fn ops(&self) -> u64 {
        self.by.values().map(|(_, s)| s.n).sum()
    }
}

/// A program, for a report line.
pub fn label(op: &Op) -> String {
    let [h0, h1] = op.shaders;
    match op.shape {
        Shape::Groups => format!("{} cs {h0:016x}", op.kind),
        Shape::Draw => format!("{} vs {h0:016x} ps {h1:016x}", op.kind),
        Shape::Indirect => format!("{} shaders {h0:016x} {h1:016x}", op.kind),
        Shape::Bulk => op.kind.to_string(),
    }
}

/// The size of one operation of `op`'s shape with `args`.
pub fn size(shape: Shape, [a, b, c]: [u32; 3]) -> String {
    match shape {
        Shape::Groups => format!("groups {a}x{b}x{c}"),
        Shape::Draw => format!("{a} vertices or indices x {b} instances"),
        Shape::Indirect => format!("up to {a} records"),
        Shape::Bulk if a != 0 => format!("{a} bytes"),
        Shape::Bulk => String::new(),
    }
}

/// From intervals on one queue: (first start to last end, the time covered by intervals).
pub fn busy(spans: &mut [(u64, u64)]) -> (u64, u64) {
    spans.sort_unstable();
    let (Some(first), Some(last)) = (spans.first().map(|s| s.0), spans.iter().map(|s| s.1).max()) else { return (0, 0) };
    let (mut covered, mut at) = (0u64, first);
    for &(s, e) in spans.iter() {
        let s = s.max(at);
        if e > s {
            covered += e - s;
            at = e;
        }
    }
    (last - first, covered)
}

/// A queue's timestamp clock: ticks per second, and the log's time (`log::uptime`) of tick `gpu0`.
#[derive(Clone, Copy)]
struct Clock {
    freq: u64,
    gpu0: u64,
    t0: f64,
}

impl Clock {
    fn ms(&self, ticks: u64) -> f64 {
        ticks as f64 * 1000.0 / self.freq as f64
    }

    fn at(&self, tick: u64) -> f64 {
        self.t0 + (tick as f64 - self.gpu0 as f64) / self.freq as f64
    }
}

struct Pending {
    slot: u32,
    /// the queue it was submitted to (0: not yet)
    queue: usize,
}

pub struct Times {
    heap: ID3D12QueryHeap,
    buf: ID3D12Resource,
    cpu: *mut u64,
    /// pair → the operation holding it (0: free)
    owner: Vec<u32>,
    pending: HashMap<u32, Pending>,
    clocks: HashMap<usize, Option<Clock>>,
    agg: Agg,
    /// per queue, the timed intervals of this period (ticks)
    spans: HashMap<usize, Vec<(u64, u64)>>,
    not_timed: u64,
    slow: u32,
    since: Instant,
}
// SAFETY: `cpu` is the persistent map of `buf`; used under `culprit`'s mutex
unsafe impl Send for Times {}

impl Times {
    pub unsafe fn new(dev: &ID3D12Device) -> Option<Times> {
        let mut heap: Option<ID3D12QueryHeap> = None;
        let hd = D3D12_QUERY_HEAP_DESC { Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP, Count: (OPS * 2) as u32, NodeMask: 0 };
        if let Err(e) = dev.CreateQueryHeap(&hd, &mut heap) {
            crate::xlog_warn!("GPU times: no timestamp query heap: {e}");
            return None;
        }
        let props = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: (OPS * 16) as u64,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut buf: Option<ID3D12Resource> = None;
        if let Err(e) = dev.CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut buf) {
            crate::xlog_warn!("GPU times: no readback buffer: {e}");
            return None;
        }
        let buf = buf?;
        let mut p: *mut c_void = std::ptr::null_mut();
        buf.Map(0, None, Some(&mut p)).ok()?;
        crate::xlog_info!(
            "GPU times: timestamps around each marked operation ({OPS} at once, {} MB of readback); reports every {} s, slow ops over {SLOW_MS} ms",
            OPS * 16 >> 20,
            PERIOD.as_secs()
        );
        Some(Times {
            heap: heap?,
            buf,
            cpu: p as *mut u64,
            owner: vec![0; OPS],
            pending: HashMap::new(),
            clocks: HashMap::new(),
            agg: Agg::default(),
            spans: HashMap::new(),
            not_timed: 0,
            slow: 0,
            since: Instant::now(),
        })
    }

    unsafe fn stamp(&self, l: &ID3D12GraphicsCommandList, k: usize) {
        l.EndQuery(&self.heap, D3D12_QUERY_TYPE_TIMESTAMP, k as u32);
        l.ResolveQueryData(&self.heap, D3D12_QUERY_TYPE_TIMESTAMP, k as u32, 1, &self.buf, k as u64 * 8);
    }

    /// Before operation `id` on `l`: its pair and its first timestamp (none: the pair is held).
    pub unsafe fn start(&mut self, l: &ID3D12GraphicsCommandList, id: u32) {
        let slot = id as usize % OPS;
        if self.owner[slot] != 0 {
            self.not_timed += 1;
            return;
        }
        self.owner[slot] = id;
        // the CPU may write a readback buffer; 0 = not resolved yet
        std::ptr::write_volatile(self.cpu.add(slot * 2), 0);
        std::ptr::write_volatile(self.cpu.add(slot * 2 + 1), 0);
        self.pending.insert(id, Pending { slot: slot as u32, queue: 0 });
        self.stamp(l, slot * 2);
    }

    /// After operation `id`: its second timestamp.
    pub unsafe fn stop(&mut self, l: &ID3D12GraphicsCommandList, id: u32) {
        if let Some(p) = self.pending.get(&id) {
            self.stamp(l, p.slot as usize * 2 + 1);
        }
    }

    /// The operations `ids` went to `queue`.
    pub unsafe fn submitted(&mut self, queue: *mut c_void, ids: &[u32]) {
        let q = queue as usize;
        self.clocks.entry(q).or_insert_with(|| calibrate(queue));
        for id in ids {
            if let Some(p) = self.pending.get_mut(id) {
                if p.queue == 0 {
                    p.queue = q;
                }
            }
        }
    }

    fn values(&self, slot: u32) -> (u64, u64) {
        // SAFETY: slots index the mapped buffer of OPS pairs
        unsafe { (std::ptr::read_volatile(self.cpu.add(slot as usize * 2)), std::ptr::read_volatile(self.cpu.add(slot as usize * 2 + 1))) }
    }

    /// The operations with both timestamps resolved: added up, their pairs freed.
    pub fn collect(&mut self, op_of: impl Fn(u32) -> Option<Op>) {
        let mut done = Vec::new();
        for (&id, p) in &self.pending {
            let Some(Some(clock)) = self.clocks.get(&p.queue) else { continue };
            let (s, e) = self.values(p.slot);
            if s != 0 && e != 0 {
                done.push((id, p.slot, p.queue, *clock, s, e));
            }
        }
        for (id, slot, q, clock, s, e) in done {
            self.pending.remove(&id);
            self.owner[slot as usize] = 0;
            if e < s {
                continue;
            }
            let ms = clock.ms(e - s);
            let op = op_of(id).unwrap_or(Op { kind: "(forgotten op)", ..Default::default() });
            self.agg.add(&op, ms);
            self.spans.entry(q).or_default().push((s, e));
            if ms > SLOW_MS && self.slow < SLOW_LINES {
                self.slow += 1;
                crate::xlog_warn!(
                    "GPU times: slow op #{id}: {} ({}), {ms:.0} ms on q{}, from +{:.3} s",
                    label(&op),
                    size(op.shape, op.args),
                    super::list_x::queue_id(q),
                    clock.at(s)
                );
            }
        }
    }

    /// The period's programs, most time first; a new period starts.
    pub fn report(&mut self, why: &str) {
        let wall = self.since.elapsed().as_secs_f64();
        crate::xlog_info!(
            "GPU times{why}: {} ops timed in {wall:.1} s of wall clock, {:.1} ms of GPU time summed over queues, {} not timed (no free query pair), {} pending. Times are per queue (ops on one queue are serialised by the markers; queues run side by side, so an op's time includes what other queues ran meanwhile and the sum can exceed the wall clock)",
            self.agg.ops(),
            self.agg.total_ms(),
            self.not_timed,
            self.pending.len()
        );
        let mut queues: Vec<usize> = self.spans.keys().copied().collect();
        queues.sort_unstable();
        for q in queues {
            let Some(Some(clock)) = self.clocks.get(&q) else { continue };
            let (span, covered) = busy(self.spans.get_mut(&q).unwrap());
            let gap = span - covered;
            crate::xlog_info!(
                "GPU times{why}: queue q{}: {:.1} ms from the first timed op to the end of the last, {:.1} ms in timed ops, {:.1} ms between them (unattributed or idle, {:.0}%)",
                super::list_x::queue_id(q),
                clock.ms(span),
                clock.ms(covered),
                clock.ms(gap),
                if span > 0 { gap as f64 * 100.0 / span as f64 } else { 0.0 }
            );
        }
        for (k, (op, s)) in self.agg.top(TOP).iter().enumerate() {
            crate::xlog_info!(
                "GPU times{why}: {:>2}. {}: {} ops, {:.1} ms, mean {:.3} ms, max {:.1} ms ({})",
                k + 1,
                label(op),
                s.n,
                s.total_ms,
                s.total_ms / s.n.max(1) as f64,
                s.max_ms,
                size(op.shape, s.max_args)
            );
        }
        self.agg = Agg::default();
        self.spans.clear();
        self.not_timed = 0;
        self.since = Instant::now();
    }

    pub fn due(&self) -> bool {
        self.since.elapsed() >= PERIOD
    }

    /// At a device removal: the submitted operations whose first timestamp is there and not the
    /// second (started, not finished), and how many never started.
    pub fn unfinished(&self, op_of: impl Fn(u32) -> Option<Op>) {
        let mut started = Vec::new();
        let mut never = 0;
        for (&id, p) in &self.pending {
            let Some(Some(clock)) = self.clocks.get(&p.queue) else { continue };
            match self.values(p.slot) {
                (0, _) => never += 1,
                (s, 0) => started.push((clock.at(s), id, p.queue)),
                _ => {}
            }
        }
        started.sort_by(|a, b| a.0.total_cmp(&b.0));
        crate::xlog_error!("GPU times at removal: {} submitted ops started and not finished, {never} never started", started.len());
        for (t, id, q) in started.iter().take(64) {
            let op = op_of(*id).unwrap_or(Op { kind: "(forgotten op)", ..Default::default() });
            crate::xlog_error!("GPU times at removal: not finished: #{id} {} ({}) on q{}, started at +{t:.3} s", label(&op), size(op.shape, op.args), super::list_x::queue_id(*q));
        }
    }
}

/// `queue`'s timestamp frequency and a timestamp at a known log time (None: a queue without
/// timestamps, e.g. a copy queue).
unsafe fn calibrate(queue: *mut c_void) -> Option<Clock> {
    use windows::core::Interface;
    use windows_sys::Win32::System::Performance::{QueryPerformanceCounter, QueryPerformanceFrequency};
    let q = ID3D12CommandQueue::from_raw_borrowed(&queue)?;
    if q.GetDesc().Type == D3D12_COMMAND_LIST_TYPE_COPY {
        return None;
    }
    let freq = q.GetTimestampFrequency().ok().filter(|&f| f > 0)?;
    let (mut gpu, mut cpu) = (0u64, 0u64);
    q.GetClockCalibration(&mut gpu, &mut cpu).ok()?;
    let (mut now, mut qf) = (0i64, 0i64);
    QueryPerformanceCounter(&mut now);
    QueryPerformanceFrequency(&mut qf);
    let t0 = crate::log::uptime() - (now as f64 - cpu as f64) / qf.max(1) as f64;
    Some(Clock { freq, gpu0: gpu, t0 })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dispatch(cs: u64, groups: [u32; 3]) -> Op {
        Op { kind: "Dispatch", pso: 1, shaders: [cs, 0], args: groups, res: [0; 2], shape: Shape::Groups }
    }

    /// Operations add up by program; the longest one's groups are kept; the top is by total.
    #[test]
    fn programs_add_up_and_rank_by_total_time() {
        let mut a = Agg::default();
        a.add(&dispatch(0xa, [1, 1, 1]), 2.0);
        a.add(&dispatch(0xa, [8, 8, 1]), 10.0);
        a.add(&dispatch(0xa, [2, 2, 1]), 3.0);
        a.add(&dispatch(0xb, [4, 1, 1]), 12.0);
        let draw = Op { kind: "DrawIndexedInstanced", pso: 2, shaders: [0xc, 0xd], args: [36, 1, 0], res: [0; 2], shape: Shape::Draw };
        a.add(&draw, 0.5);
        let top = a.top(25);
        assert_eq!(top.len(), 3);
        assert_eq!(top[0].0.shaders[0], 0xa);
        assert_eq!(top[0].1, Stat { n: 3, total_ms: 15.0, max_ms: 10.0, max_args: [8, 8, 1] });
        assert_eq!(top[1].0.shaders[0], 0xb);
        assert_eq!(top[2].0.kind, "DrawIndexedInstanced");
        assert_eq!(a.top(1).len(), 1);
        assert_eq!(a.ops(), 5);
        assert_eq!(a.total_ms(), 27.5);
        assert_eq!(label(&top[0].0), "Dispatch cs 000000000000000a");
        assert_eq!(size(Shape::Groups, top[0].1.max_args), "groups 8x8x1");
        assert_eq!(label(&draw), "DrawIndexedInstanced vs 000000000000000c ps 000000000000000d");
    }

    /// A queue's time between its first start and last end, and the part inside timed operations
    /// (overlapping ones counted once).
    #[test]
    fn the_time_between_timed_ops_is_the_span_less_their_union() {
        let mut spans = vec![(100, 150), (10, 30), (20, 40), (200, 210), (205, 208)];
        assert_eq!(busy(&mut spans), (200, 30 + 50 + 10));
        assert_eq!(busy(&mut []), (0, 0));
        let c = Clock { freq: 1_000_000, gpu0: 2_000_000, t0: 5.0 };
        assert_eq!(c.ms(1500), 1.5);
        assert_eq!(c.at(3_000_000), 6.0);
    }
}
