//! LOCAL DIAGNOSTIC, not for upstream: `XCL_LOCAL_QJOURNAL=1` – per command queue, the GPU-side
//! operations XCL issues on it (waits, signals, submissions), each followed by a signal of a
//! progress fence of the queue's own, so the operation a queue stands at is known exactly.
//! When the sync watchdog is about to release GPU waits (`dump`, before it signals on the CPU),
//! for every queue: the first operation not passed, and for a wait the queue that signals that
//! fence value and whether that queue got past its signal, in its own order.
//! `XCL_LOCAL_QJOURNAL_EVERY_S=<s>` also dumps every <s> seconds (a stop with no watchdog).

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

pub fn enabled() -> bool {
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| std::env::var("XCL_LOCAL_QJOURNAL").is_ok_and(|v| v.trim() == "1"))
}

const RING: usize = 4096;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Wait,
    Signal,
    Exec,
}

struct Entry {
    /// the progress value the queue reaches once past this operation
    seq: u64,
    kind: Kind,
    fence: Option<ID3D12Fence>,
    fence_raw: usize,
    value: u64,
    tag: String,
    at: Instant,
}

struct QState {
    name: Option<String>,
    progress: ID3D12Fence,
    next: u64,
    ring: VecDeque<Entry>,
}

// (COM pointers used under the lock only; D3D12 fences are free-threaded)
unsafe impl Send for QState {}

static QUEUES: Mutex<Option<HashMap<usize, QState>>> = Mutex::new(None);
static NAMES: Mutex<Vec<(usize, &'static str)>> = Mutex::new(Vec::new());
static START: OnceLock<Instant> = OnceLock::new();

/// Name an XCL queue (not a title queue): title queues are named `q<n>` as in the sync logs.
pub fn name(queue: usize, n: &'static str) {
    if enabled() {
        NAMES.lock().unwrap().push((queue, n));
    }
}

/// `queue` just got `kind` (on `fence` >= / = `value`; for `Exec`, `value` = list count).
pub unsafe fn note(queue: &ID3D12CommandQueue, kind: Kind, fence: Option<&ID3D12Fence>, value: u64, tag: impl FnOnce() -> String) {
    if !enabled() {
        return;
    }
    START.get_or_init(Instant::now);
    start_periodic();
    let key = queue.as_raw() as usize;
    let mut g = QUEUES.lock().unwrap();
    let m = g.get_or_insert_with(HashMap::new);
    if !m.contains_key(&key) {
        let mut dev: Option<ID3D12Device> = None;
        let Some(dev) = queue.GetDevice(&mut dev).ok().and(dev) else { return };
        let Ok(progress) = dev.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE) else { return };
        let name = NAMES.lock().unwrap().iter().find(|e| e.0 == key).map(|e| e.1.to_string());
        m.insert(key, QState { name, progress, next: 0, ring: VecDeque::with_capacity(RING) });
    }
    let q = m.get_mut(&key).unwrap();
    q.next += 1;
    let _ = queue.Signal(&q.progress, q.next);
    if q.ring.len() >= RING {
        q.ring.pop_front();
    }
    q.ring.push_back(Entry {
        seq: q.next,
        kind,
        fence: fence.cloned(),
        fence_raw: fence.map_or(0, |f| f.as_raw() as usize),
        value,
        tag: tag(),
        at: Instant::now(),
    });
}

/// a title fence by its raw pointer
pub unsafe fn note_raw(queue: *mut std::ffi::c_void, kind: Kind, fence: usize, value: u64, tag: &'static str) {
    if !enabled() || queue.is_null() || fence == 0 {
        return;
    }
    let fp = fence as *mut std::ffi::c_void;
    let (Some(q), Some(f)) = (ID3D12CommandQueue::from_raw_borrowed(&queue), ID3D12Fence::from_raw_borrowed(&fp)) else { return };
    note(q, kind, Some(f), value, || format!("{tag} {fence:#x}"));
}

/// CPU-side waits of submitting threads: (count, total µs) per kind.
pub static CPU_W64: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
pub static CPU_SPARSE: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

pub fn cpu_wait(c: &[AtomicU64; 2], t: Instant) {
    if enabled() {
        c[0].fetch_add(1, Ordering::Relaxed);
        c[1].fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
    }
}

fn qname(key: usize, q: &QState) -> String {
    q.name.clone().unwrap_or_else(|| format!("q{}", super::list_x::queue_id(key)))
}

/// Log where each queue stands (at most `MAX` dumps; `why` says what asked for it).
pub fn dump(why: &str) {
    static N: AtomicU64 = AtomicU64::new(0);
    const MAX: u64 = 80;
    static LAST: Mutex<Option<Instant>> = Mutex::new(None);
    if !enabled() {
        return;
    }
    // (a burst of releases: the first one of it)
    if why == "watchdog" {
        let mut l = LAST.lock().unwrap();
        if l.is_some_and(|t| t.elapsed().as_millis() < 1000) {
            return;
        }
        *l = Some(Instant::now());
    }
    if N.fetch_add(1, Ordering::Relaxed) >= MAX {
        return;
    }
    let t0 = *START.get_or_init(Instant::now);
    let now = Instant::now();
    let g = QUEUES.lock().unwrap();
    let Some(m) = g.as_ref() else { return };
    // every signal still in the rings: fence → [(value, queue, seq)]
    let mut signals: HashMap<usize, Vec<(u64, usize, u64)>> = HashMap::new();
    let mut done: HashMap<usize, u64> = HashMap::new();
    for (&k, q) in m {
        done.insert(k, unsafe { q.progress.GetCompletedValue() });
        for e in q.ring.iter().filter(|e| e.kind == Kind::Signal) {
            signals.entry(e.fence_raw).or_default().push((e.value, k, e.seq));
        }
    }
    crate::xlog_warn!(
        "qjournal [{why}] at {:.3} s: cpu waits w64-fallback {} ({} ms), sparse wait_for {} ({} ms)",
        (now - t0).as_secs_f64(),
        CPU_W64[0].load(Ordering::Relaxed),
        CPU_W64[1].load(Ordering::Relaxed) / 1000,
        CPU_SPARSE[0].load(Ordering::Relaxed),
        CPU_SPARSE[1].load(Ordering::Relaxed) / 1000
    );
    let mut keys: Vec<usize> = m.keys().copied().collect();
    keys.sort_by_key(|k| qname(*k, &m[k]));
    for k in keys {
        let q = &m[&k];
        let c = done[&k];
        let pending: Vec<&Entry> = q.ring.iter().filter(|e| e.seq > c).collect();
        let last = q.ring.iter().rev().find(|e| e.seq <= c);
        let last_s = last.map_or("-".to_string(), |e| format!("#{} {:?} {} v={:#x} ({:.0} ms ago)", e.seq, e.kind, e.tag, e.value, (now - e.at).as_secs_f64() * 1000.0));
        if pending.is_empty() {
            crate::xlog_warn!("qjournal   {}: idle at #{c} (last {last_s})", qname(k, q));
            continue;
        }
        crate::xlog_warn!("qjournal   {}: at #{c}, {} ops not passed; last passed {last_s}", qname(k, q), pending.len());
        for e in pending.iter().take(6) {
            let mut s = format!("#{} {:?} {} v={:#x} (issued {:.0} ms ago)", e.seq, e.kind, e.tag, e.value, (now - e.at).as_secs_f64() * 1000.0);
            if e.kind == Kind::Wait {
                let fc = e.fence.as_ref().map_or(0, |f| unsafe { f.GetCompletedValue() });
                s += &format!(" fence done={fc:#x}");
                if fc < e.value {
                    // the first signal of that fence reaching the value
                    let mut sg: Vec<(u64, usize, u64)> = signals.get(&e.fence_raw).cloned().unwrap_or_default();
                    sg.sort_by_key(|x| (x.0, x.2));
                    match sg.iter().find(|x| x.0 >= e.value) {
                        Some(&(v, pq, pseq)) => {
                            let pdone = done.get(&pq).copied().unwrap_or(0);
                            s += &format!(
                                " <- signal v={v:#x} by {} at #{pseq} ({}; it is at #{pdone})",
                                qname(pq, &m[&pq]),
                                if pdone >= pseq { "passed" } else { "NOT passed" }
                            );
                        }
                        None => s += " <- no GPU signal of that value issued (CPU signal or none)",
                    }
                }
            }
            crate::xlog_warn!("qjournal     {s}");
        }
    }
}

/// `XCL_LOCAL_QJOURNAL_EVERY_S`: a thread that dumps every that many seconds.
pub fn start_periodic() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    let Some(s) = std::env::var("XCL_LOCAL_QJOURNAL_EVERY_S").ok().and_then(|v| v.trim().parse::<u64>().ok()).filter(|&s| s > 0) else { return };
    if !enabled() {
        return;
    }
    ONCE.call_once(|| {
        let _ = std::thread::Builder::new().name("xcl-qjournal".into()).spawn(move || loop {
            std::thread::sleep(std::time::Duration::from_secs(s));
            dump("periodic");
        });
    });
}
