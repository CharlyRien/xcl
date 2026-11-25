//! LOCAL agent P, not for upstream as is: `XCL_LOCAL_PSO_BUDGET_MB=<N>` (issue #2403).
//!
//! NVIDIA's driver keeps the machine code of each pipeline from its first use until it is
//! released, in one heap of at most 512 MB; a submission that needs more removes the device
//! (0x887a0005, no page fault). On an RTX 3090 (616.56) the test of #2403 is removed at 446 MB
//! of `GetCachedBlob` sizes; a retail title bound 472 MB of XCL's pipelines when it was removed,
//! and XCL never releases a pipeline it built for a stand-in (`pso_async`).
//!
//! With the switch, the code of those pipelines (the ones `pso_async` builds for a stand-in,
//! with the compute pass of a two-pass one) is summed from their first bind. Past N MB the least
//! recently bound ones that no recording references are released, down to 7/8 of N, and built
//! again by the next `resolve` of their stand-in (`pso_async::revive`: a compute program from
//! the driver's blob, `real::compute_blob` / `driver_cache`).
//!
//! What references a pipeline:
//! * a list's recording, from the bind that records it (`bind`: `SetPipelineState`, a Reset or a
//!   creation with an initial pipeline, `ClearState`) until that list's next Reset or its last
//!   Release (`list_reset`). A closed list may be submitted again, so a submit does not end it.
//! * the GPU's execution of it: a pipeline whose recordings are all gone is released only once
//!   every queue XCL knows (the title's, tile mappings, uploads) is past a fence value signalled
//!   after that (as `late_release`), on a thread of its own: the title's threads never wait.
//! * a thread between its `resolve` and its bind: a pipeline goes no sooner than `GRACE` after it
//!   was chosen, and a bind of one already chosen keeps it until its lists are reset again.
//!
//! Log: each release batch, each revive (first 20, then powers of two), and every 10 s the code
//! of the pipelines bound (`LOCAL pso budget:`).
use std::collections::{HashMap, HashSet, VecDeque};
use std::ffi::c_void;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

/// a pipeline chosen to go is released no sooner than this after (a thread that resolved it
/// before has bound it by then)
const GRACE: Duration = Duration::from_millis(250);
/// how often the thread here seals a batch / looks for batches the queues are past
const TICK: Duration = Duration::from_millis(4);
const SEAL_EVERY: Duration = Duration::from_millis(10);
const REPORT_EVERY: Duration = Duration::from_secs(10);

/// The budget in bytes (`XCL_LOCAL_PSO_BUDGET_MB`), None: off.
pub fn budget() -> Option<u64> {
    static B: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    *B.get_or_init(|| std::env::var("XCL_LOCAL_PSO_BUDGET_MB").ok().and_then(|v| v.trim().parse::<u64>().ok()).filter(|&mb| mb > 0).map(|mb| mb << 20))
}

pub fn enabled() -> bool {
    budget().is_some()
}

struct Entry {
    standin: usize,
    /// driver code (`GetCachedBlob`) with the two-pass compute pass
    bytes: u64,
    /// `pso_async` can build it again
    rebuildable: bool,
    /// bound since it was built (its code counts from then)
    bound: bool,
    /// `State::tick` of its last bind
    last: u64,
    /// lists whose recording binds it
    refs: u32,
    /// chosen to go (then not counted in `code` but in `awaiting`)
    gone: Option<Instant>,
    /// in `open` or a sealed batch
    queued: bool,
    /// binds since it was chosen: a batch sealed before the last one does not cover it
    epoch: u32,
    /// the two-pass compute pipeline that goes with it (0: none)
    alias: usize,
}

#[derive(Default)]
struct State {
    live: HashMap<usize, Entry>,
    alias: HashMap<usize, usize>,
    /// list → the pipelines its recording binds
    lists: HashMap<usize, HashSet<usize>>,
    /// chosen, no recording left: to seal into a batch (with its `epoch`)
    open: Vec<(usize, u32)>,
    tick: u64,
    code: u64,
    awaiting: u64,
    peak: u64,
    stats: Stats,
}

/// What the budget did so far (`stats`).
#[derive(Default, Clone, Copy, Debug)]
pub struct Stats {
    /// driver code of the pipelines bound since they were built and not chosen to go
    pub code: u64,
    /// chosen to go, not released yet
    pub awaiting: u64,
    pub peak: u64,
    pub pipelines: u64,
    pub evicted: u64,
    pub evicted_bytes: u64,
    pub released: u64,
    pub released_bytes: u64,
    pub revived: u64,
    pub revive_failed: u64,
    pub revive_us: u64,
    pub revive_max_us: u64,
    /// time spent choosing what goes, on the binding threads
    pub evict_us: u64,
    /// the budget was passed with too little to release (all in recordings)
    pub short: u64,
    /// binds of a pipeline already chosen to go (it then waits for its lists again)
    pub binds_of_gone: u64,
    /// batches sealed (a fence value a queue each)
    pub batches: u64,
}

static STATE: Mutex<Option<State>> = Mutex::new(None);

/// Queues a test submits on (the title's are `present::known_queues`).
static WATCHED: Mutex<Vec<usize>> = Mutex::new(Vec::new());

/// Tests: a queue whose work a released pipeline must be past.
#[doc(hidden)]
pub fn watch_queue(q: &ID3D12CommandQueue) {
    let raw = q.as_raw() as usize;
    let mut g = WATCHED.lock().unwrap();
    if !g.contains(&raw) {
        std::mem::forget(q.clone());
        g.push(raw);
    }
}

pub fn stats() -> Stats {
    let g = STATE.lock().unwrap();
    let Some(s) = g.as_ref() else { return Stats::default() };
    Stats { code: s.code, awaiting: s.awaiting, peak: s.peak, pipelines: s.live.values().filter(|e| e.bound && e.gone.is_none()).count() as u64, ..s.stats }
}

fn mb(b: u64) -> f64 {
    b as f64 / 1048576.0
}

/// The driver code of a pipeline: its `GetCachedBlob` size.
unsafe fn code_bytes(pso: usize) -> u64 {
    if pso == 0 {
        return 0;
    }
    super::real::driver_code_bytes(pso as *mut c_void)
}

/// `pso_async` built `real` for `standin` (`rebuildable`: it can build it again).
pub fn built(standin: usize, real: usize, rebuildable: bool) {
    if !enabled() || real == 0 {
        return;
    }
    let alias = super::real::two_pass_compute(real).unwrap_or(0);
    let bytes = unsafe { code_bytes(real) + code_bytes(alias) };
    start_thread();
    let mut g = STATE.lock().unwrap();
    let s = g.get_or_insert_with(State::default);
    if let Some(old) = s.live.remove(&real) {
        // (an address released without us: not counted any more)
        uncount(s, real, &old);
    }
    s.live.insert(real, Entry { standin, bytes, rebuildable, bound: false, last: 0, refs: 0, gone: None, queued: false, epoch: 0, alias });
    if alias != 0 {
        s.alias.insert(alias, real);
    }
}

fn uncount(s: &mut State, real: usize, e: &Entry) {
    if e.gone.is_some() {
        s.awaiting -= e.bytes;
    } else if e.bound {
        s.code -= e.bytes;
    }
    if e.alias != 0 {
        s.alias.remove(&e.alias);
    }
    if e.refs > 0 {
        for set in s.lists.values_mut() {
            set.remove(&real);
        }
    }
}

/// The title released the stand-in of `real`: `pso_async` releases `real` itself, now.
pub fn forget(real: usize) {
    if !enabled() {
        return;
    }
    let mut g = STATE.lock().unwrap();
    let Some(s) = g.as_mut() else { return };
    if let Some(e) = s.live.remove(&real) {
        uncount(s, real, &e);
    }
}

/// `list` records a bind of `pso` (any pipeline: only those built for a stand-in count).
pub unsafe fn bind(list: *mut c_void, pso: usize) {
    if !enabled() || pso == 0 || list.is_null() {
        return;
    }
    let budget = budget().unwrap_or(u64::MAX);
    let victims = {
        let mut g = STATE.lock().unwrap();
        let Some(s) = g.as_mut() else { return };
        let owner = s.alias.get(&pso).copied().unwrap_or(pso);
        if !s.live.contains_key(&owner) {
            return;
        }
        s.tick += 1;
        let tick = s.tick;
        let fresh = s.lists.entry(list as usize).or_default().insert(owner);
        let e = s.live.get_mut(&owner).unwrap();
        e.last = tick;
        if fresh {
            e.refs += 1;
        }
        let mut first = false;
        if e.gone.is_some() {
            // (a batch sealed before does not cover this recording)
            e.epoch += 1;
            s.stats.binds_of_gone += 1;
        } else if !e.bound {
            e.bound = true;
            first = true;
            s.code += e.bytes;
            s.peak = s.peak.max(s.code + s.awaiting);
        }
        if first && s.code > budget {
            choose(s, owner, budget)
        } else {
            Vec::new()
        }
    };
    for (standin, real) in victims {
        if !super::pso_async::evict(standin, real) {
            evict_failed(real);
        }
    }
}

/// The least recently bound pipelines no recording references go, down to 7/8 of `budget`.
fn choose(s: &mut State, keep: usize, budget: u64) -> Vec<(usize, usize)> {
    let t = Instant::now();
    let target = budget - budget / 8;
    let mut free: Vec<(u64, usize)> = s.live.iter().filter(|(&r, e)| r != keep && e.bound && e.gone.is_none() && e.refs == 0 && e.rebuildable).map(|(&r, e)| (e.last, r)).collect();
    free.sort_unstable();
    let mut out = Vec::new();
    let now = Instant::now();
    for (_, r) in free {
        if s.code <= target {
            break;
        }
        let e = s.live.get_mut(&r).unwrap();
        e.gone = Some(now);
        e.queued = true;
        s.code -= e.bytes;
        s.awaiting += e.bytes;
        s.stats.evicted += 1;
        s.stats.evicted_bytes += e.bytes;
        s.open.push((r, e.epoch));
        out.push((e.standin, r));
    }
    if s.code > target {
        s.stats.short += 1;
        let n = s.stats.short;
        if n <= 10 || n.is_power_of_two() {
            let held = s.live.values().filter(|e| e.bound && e.gone.is_none() && e.refs > 0).map(|e| e.bytes).sum::<u64>();
            crate::xlog_warn!(
                "LOCAL pso budget: {:.1} MB bound, {:.1} MB of it in recordings – cannot get under {:.0} MB ({} times)",
                mb(s.code),
                mb(held),
                mb(target),
                n
            );
        }
    }
    s.stats.evict_us += t.elapsed().as_micros() as u64;
    out
}

/// `pso_async` did not let `real` go (its stand-in was released meanwhile, or it is another
/// pipeline now): it is counted again.
fn evict_failed(real: usize) {
    let mut g = STATE.lock().unwrap();
    let Some(s) = g.as_mut() else { return };
    if let Some(e) = s.live.get_mut(&real) {
        if e.gone.take().is_some() {
            s.awaiting -= e.bytes;
            s.code += e.bytes;
            s.stats.evicted -= 1;
            s.stats.evicted_bytes -= e.bytes;
        }
    }
}

/// `list`'s recording is gone (Reset, or its last Release).
pub fn list_reset(list: *mut c_void) {
    if !enabled() || list.is_null() {
        return;
    }
    if let Some(s) = STATE.lock().unwrap().as_mut() {
        reset_locked(s, list as usize);
    }
}

fn reset_locked(s: &mut State, list: usize) {
    let Some(set) = s.lists.get_mut(&list) else { return };
    if set.is_empty() {
        return;
    }
    let set = std::mem::take(set);
    for r in set {
        if let Some(e) = s.live.get_mut(&r) {
            e.refs -= 1;
            if e.refs == 0 && e.gone.is_some() && !e.queued {
                e.queued = true;
                s.open.push((r, e.epoch));
            }
        }
    }
}

/// `list` is released for good.
pub fn list_release(list: *mut c_void) {
    if !enabled() || list.is_null() {
        return;
    }
    list_reset(list);
    if let Some(s) = STATE.lock().unwrap().as_mut() {
        s.lists.remove(&(list as usize));
    }
}

/// `pso_async::revive` built an evicted pipeline again (`real`, 0: failed) in `took`.
pub fn revived(standin: usize, real: usize, compute: bool, took: Duration) {
    if !enabled() {
        return;
    }
    let us = took.as_micros() as u64;
    let n = {
        let mut g = STATE.lock().unwrap();
        let s = g.get_or_insert_with(State::default);
        if real == 0 {
            s.stats.revive_failed += 1;
        } else {
            s.stats.revived += 1;
        }
        s.stats.revive_us += us;
        s.stats.revive_max_us = s.stats.revive_max_us.max(us);
        s.stats.revived + s.stats.revive_failed
    };
    if real == 0 {
        crate::xlog_error!("LOCAL pso budget: building the released pipeline of stand-in {standin:#x} again FAILED ({} ms) – its binds get the stand-in", us / 1000);
    } else if n <= 20 || n.is_power_of_two() {
        crate::xlog_info!("LOCAL pso budget: {} pipeline of stand-in {standin:#x} built again in {:.1} ms ({} so far)", if compute { "compute" } else { "graphics" }, us as f64 / 1000.0, n);
    }
}

/// For the device-removal report.
pub fn removal_line() -> Option<String> {
    enabled().then(|| format!("LOCAL pso budget at the removal: {}", summary()))
}

fn summary() -> String {
    let s = stats();
    format!(
        "{:.1} MB bound in {} pipelines (peak {:.1} MB with {:.1} MB awaiting release now), {} released ({:.1} MB), {} chosen, {} built again ({} failed, {:.1} ms in all, longest {:.1} ms), {:.1} ms choosing, {} times short, {} binds of chosen ones",
        mb(s.code),
        s.pipelines,
        mb(s.peak),
        mb(s.awaiting),
        s.released,
        mb(s.released_bytes),
        s.evicted,
        s.revived,
        s.revive_failed,
        s.revive_us as f64 / 1000.0,
        s.revive_max_us as f64 / 1000.0,
        s.evict_us as f64 / 1000.0,
        s.short,
        s.binds_of_gone
    )
}

fn start_thread() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        crate::xlog_info!("LOCAL pso budget: on, {} MB of driver code for the pipelines built for stand-ins (XCL_LOCAL_PSO_BUDGET_MB)", budget().unwrap_or(0) >> 20);
        let _ = std::thread::Builder::new().name("xcl-pso-budget".into()).spawn(|| {
            let mut sealer = Sealer::default();
            let mut report = Instant::now();
            loop {
                std::thread::sleep(TICK);
                unsafe { sealer.step() };
                if report.elapsed() >= REPORT_EVERY {
                    report = Instant::now();
                    crate::xlog_info!("LOCAL pso budget: {}", summary());
                }
            }
        });
    });
}

#[derive(Default)]
struct Sealer {
    /// queue → its fence here and the last value signalled on it
    fences: HashMap<usize, (ID3D12Fence, u64)>,
    sealed: VecDeque<(Vec<(usize, u32)>, Vec<(ID3D12Fence, u64)>, Instant)>,
    last_seal: Option<Instant>,
}

impl Sealer {
    unsafe fn queues() -> Vec<ID3D12CommandQueue> {
        let mut raws = super::present::known_queues();
        raws.extend(WATCHED.lock().unwrap().iter().copied());
        raws.sort_unstable();
        raws.dedup();
        let mut queues: Vec<ID3D12CommandQueue> = Vec::new();
        for raw in raws {
            let raw = raw as *mut c_void;
            if let Some(q) = ID3D12CommandQueue::from_raw_borrowed(&raw) {
                queues.push(q.clone());
            }
        }
        queues.extend(super::sparse::queue());
        queues.extend(super::title_textures::upload_queues());
        queues
    }

    unsafe fn step(&mut self) {
        // seal what was chosen: a fence value on every queue, after all submitted so far
        if self.last_seal.is_none_or(|t| t.elapsed() >= SEAL_EVERY) {
            let open = STATE.lock().unwrap().as_mut().map(|s| std::mem::take(&mut s.open)).unwrap_or_default();
            if !open.is_empty() {
                self.last_seal = Some(Instant::now());
                let mut marks = Vec::new();
                let mut ok = true;
                for q in Self::queues() {
                    let key = q.as_raw() as usize;
                    if !self.fences.contains_key(&key) {
                        let mut dev: Option<ID3D12Device> = None;
                        let made = q.GetDevice(&mut dev).ok().and(dev).and_then(|d| d.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE).ok());
                        let Some(f) = made else {
                            ok = false;
                            break;
                        };
                        self.fences.insert(key, (f, 0));
                    }
                    let (fence, value) = self.fences.get_mut(&key).unwrap();
                    if q.Signal(&*fence, *value + 1).is_err() {
                        ok = false;
                        break;
                    }
                    *value += 1;
                    marks.push((fence.clone(), *value));
                }
                if ok {
                    self.sealed.push_back((open, marks, Instant::now()));
                    if let Some(s) = STATE.lock().unwrap().as_mut() {
                        s.stats.batches += 1;
                    }
                } else {
                    // (no fence or no signal: the device is gone – nothing goes)
                    if let Some(s) = STATE.lock().unwrap().as_mut() {
                        s.open.extend(open);
                    }
                }
            }
        }
        // release the batches every queue is past
        let mut ready = Vec::new();
        while self.sealed.front().is_some_and(|(_, marks, _)| marks.iter().all(|(f, v)| f.GetCompletedValue() >= *v)) {
            ready.push(self.sealed.pop_front().unwrap().0);
        }
        if ready.is_empty() {
            return;
        }
        let mut go = Vec::new();
        let mut later = Vec::new();
        {
            let mut g = STATE.lock().unwrap();
            let Some(s) = g.as_mut() else { return };
            for (r, epoch) in ready.into_iter().flatten() {
                let Some(e) = s.live.get_mut(&r) else { continue };
                match e.gone {
                    Some(at) if e.refs == 0 && e.epoch == epoch && at.elapsed() >= GRACE => {
                        let e = s.live.remove(&r).unwrap();
                        s.awaiting -= e.bytes;
                        if e.alias != 0 {
                            s.alias.remove(&e.alias);
                        }
                        s.stats.released += 1;
                        s.stats.released_bytes += e.bytes;
                        go.push(r);
                    }
                    // (too soon, or bound again since the batch was sealed: sealed again)
                    Some(_) if e.refs == 0 => later.push((r, e.epoch)),
                    // (bound again: queued when its lists are reset)
                    _ => e.queued = false,
                }
            }
            s.open.extend(later);
        }
        if !go.is_empty() {
            let n = go.len();
            for r in go {
                super::pso_async::release(r);
            }
            let s = stats();
            crate::xlog_info!("LOCAL pso budget: released {n} pipelines, {:.1} MB bound, {:.1} MB awaiting release, {} released in all", mb(s.code), mb(s.awaiting), s.released);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(standin: usize, bytes: u64, last: u64, refs: u32) -> Entry {
        Entry { standin, bytes, rebuildable: true, bound: true, last, refs, gone: None, queued: false, epoch: 0, alias: 0 }
    }

    /// A pipeline a recording binds is not chosen; the others are, least recently bound first,
    /// down to 7/8 of the budget; the one being bound is kept.
    #[test]
    fn only_pipelines_no_recording_binds_are_chosen_least_recent_first() {
        let mut s = State::default();
        for (k, last) in [(1usize, 5u64), (2, 1), (3, 3), (4, 9)] {
            s.live.insert(0x1000 * k, entry(k, 40 << 20, last, 0));
            s.code += 40 << 20;
        }
        s.live.get_mut(&0x2000).unwrap().refs = 1; // in a recording
        let out = choose(&mut s, 0x4000, 100 << 20);
        // 160 MB > 100: down to 87.5 MB – 0x3000 (last 3), then 0x1000 (last 5)
        assert_eq!(out, vec![(3, 0x3000), (1, 0x1000)]);
        assert_eq!((s.code, s.awaiting), (80 << 20, 80 << 20));
        assert!(s.live[&0x2000].gone.is_none() && s.live[&0x4000].gone.is_none());
        assert_eq!(s.open, vec![(0x3000, 0), (0x1000, 0)]);
    }

    /// A chosen pipeline still in a recording goes into a batch only once that list is reset.
    #[test]
    fn a_chosen_pipeline_waits_for_the_reset_of_its_lists() {
        let mut s = State::default();
        let mut e = entry(0xA, 1, 1, 1);
        e.gone = Some(Instant::now());
        s.live.insert(0xA000, e);
        s.lists.entry(0x77).or_default().insert(0xA000);
        reset_locked(&mut s, 0x88);
        assert!(s.open.is_empty(), "another list");
        reset_locked(&mut s, 0x77);
        assert_eq!(s.live[&0xA000].refs, 0);
        assert_eq!(s.open, vec![(0xA000, 0)]);
        reset_locked(&mut s, 0x77);
        assert_eq!(s.open.len(), 1, "queued once");
    }
}
