//! LOCAL agent P (#2403): the driver's code heap through XCL's own pipeline path.
//!
//! cargo test --release -p xcl --test pso_budget_limit -- --ignored --nocapture
//!
//! As `shader_code_limit.rs` (`MODE=keep`): compute pipelines of unique code, ~1.7 MB of driver
//! code each, every one dispatched and kept – but each is a stand-in of `pso_async`, built by its
//! first `resolve_for_bind` (as a title's compute pipeline), and every bind and list Reset is told
//! to `pso_budget` as XCL's hooks do. Frames of `LISTS` lists (one dispatch each) go out without
//! waiting, `RING` frames in flight, each list Reset as soon as its frame is submitted.
//!
//! Without `XCL_LOCAL_PSO_BUDGET_MB` the device is removed near 446 MB (RTX 3090, 616.56).
//! With it, the least recently bound pipelines are released and built again from the driver's
//! blob when bound again: `COUNT` new pipelines, then every `REVISIT`-th of them again; each
//! dispatch's result is checked against the same program run on the CPU.
//!
//! `COUNT` (600), `STMTS` (3000), `THREADS` (5, DXC), `REVISIT` (7), `PSO_TEST_DXIL_DIR` (DXIL
//! cache of this test; default: a temp folder), `DXIL_ONLY=1` (compile the DXIL, no GPU).

use std::sync::{Arc, Mutex};
use windows::core::Interface;
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
use xcl::hle::d3d12x::{pso_async, pso_budget};

const RING: usize = 3;
const LISTS: usize = 4;
const WORDS: usize = 1024;

fn var(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn constant(k: usize, stmts: usize, i: usize) -> u32 {
    ((k * stmts + i) as u32).wrapping_mul(0x9E37_79B1) | 1
}

/// A program no other one shares code with (as `shader_code_limit.rs`).
fn program(k: usize, stmts: usize) -> String {
    let mut h = String::from("RWByteAddressBuffer u : register(u0);\n[numthreads(1, 1, 1)] void main() { uint x = u.Load(0);\n");
    for i in 0..stmts {
        h += &format!("{{ uint y = u.Load((x & 1023u) * 4u); x = (x * {}u) / (y | 1u) + (y >> {}) % (x | 1u); }}\n", constant(k, stmts, i), 1 + i % 13);
    }
    h + "u.Store(4, x); }"
}

/// What `program(k)` stores, run on the CPU over `words`.
fn expected(k: usize, stmts: usize, words: &[u32]) -> u32 {
    let mut x = words[0];
    for i in 0..stmts {
        let y = words[(x & 1023) as usize];
        x = (x.wrapping_mul(constant(k, stmts, i)) / (y | 1)).wrapping_add((y >> (1 + i % 13)) % (x | 1));
    }
    x
}

fn pattern() -> Vec<u32> {
    (0..WORDS as u32).map(|i| i.wrapping_mul(2_654_435_761) ^ 0x5BD1_E995).collect()
}

/// What builds pipeline k (again): the device, root signature, DXIL, and the driver's blob of
/// its first build (as `real::compute_blob`).
struct Build {
    dev: ID3D12Device,
    rs: ID3D12RootSignature,
    dxil: Vec<u8>,
    blob: Mutex<Option<Vec<u8>>>,
    first_bytes: Mutex<u64>,
}

impl Build {
    fn make(&self) -> usize {
        unsafe {
            let mut blob = self.blob.lock().unwrap();
            let mut desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&self.rs),
                CS: D3D12_SHADER_BYTECODE { pShaderBytecode: self.dxil.as_ptr() as _, BytecodeLength: self.dxil.len() },
                ..Default::default()
            };
            if let Some(b) = blob.as_ref() {
                desc.CachedPSO = D3D12_CACHED_PIPELINE_STATE { pCachedBlob: b.as_ptr() as _, CachedBlobSizeInBytes: b.len() };
            }
            let p = match self.dev.CreateComputePipelineState::<ID3D12PipelineState>(&desc) {
                Ok(p) => p,
                Err(e) if blob.is_some() => {
                    eprintln!("blob refused ({e:?}): compiled again");
                    desc.CachedPSO = D3D12_CACHED_PIPELINE_STATE::default();
                    let Ok(p) = self.dev.CreateComputePipelineState::<ID3D12PipelineState>(&desc) else { return 0 };
                    p
                }
                Err(_) => return 0,
            };
            if blob.is_none() {
                if let Ok(b) = p.GetCachedBlob() {
                    let bytes = std::slice::from_raw_parts(b.GetBufferPointer() as *const u8, b.GetBufferSize()).to_vec();
                    *self.first_bytes.lock().unwrap() = bytes.len() as u64;
                    *blob = Some(bytes);
                }
            }
            p.into_raw() as usize
        }
    }
}

unsafe fn buffer(dev: &ID3D12Device, heap: D3D12_HEAP_TYPE, size: u64, uav: bool, state: D3D12_RESOURCE_STATES) -> ID3D12Resource {
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: size,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: if uav { D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS } else { D3D12_RESOURCE_FLAG_NONE },
        ..Default::default()
    };
    let mut r: Option<ID3D12Resource> = None;
    dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: heap, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut r).unwrap();
    r.unwrap()
}

fn transition(list: &ID3D12GraphicsCommandList, r: &ID3D12Resource, before: D3D12_RESOURCE_STATES, after: D3D12_RESOURCE_STATES) {
    let b = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: unsafe { std::mem::transmute_copy(r) },
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    };
    unsafe { list.ResourceBarrier(&[b]) };
}

fn mb(b: u64) -> f64 {
    b as f64 / 1048576.0
}

#[test]
#[ignore = "takes minutes and ends with a removed device without XCL_LOCAL_PSO_BUDGET_MB"]
fn xcl_pipelines_past_the_drivers_code_heap() {
    let (stmts, count, threads, revisit) = (var("STMTS", 3000), var("COUNT", 600), var("THREADS", 5), var("REVISIT", 7).max(1));
    let dir = std::env::var("PSO_TEST_DXIL_DIR").map(std::path::PathBuf::from).unwrap_or_else(|_| std::env::temp_dir().join("xcl-pso-budget-dxil"));
    let budget = pso_budget::budget();
    println!("budget: {}", budget.map_or("off".into(), |b| format!("{} MB", b >> 20)));
    let words = pattern();

    // DXIL of every program first (cached in `dir`: the next run starts at once)
    let t0 = std::time::Instant::now();
    let next = std::sync::atomic::AtomicUsize::new(0);
    let dxil: Vec<Mutex<Vec<u8>>> = (0..count).map(|_| Mutex::new(Vec::new())).collect();
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| loop {
                let k = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if k >= count {
                    return;
                }
                let (code, _) = xcl::gpu::recomp::dxc::compile_cached_in(&dir, &program(k, stmts), "main", "cs_6_0").expect("DXC");
                *dxil[k].lock().unwrap() = code;
            });
        }
    });
    println!("DXIL of {count} programs in {:.1} s", t0.elapsed().as_secs_f64());
    if std::env::var("DXIL_ONLY").is_ok_and(|v| v == "1") {
        return;
    }

    unsafe {
        let mut dev: Option<ID3D12Device> = None;
        D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut dev).unwrap();
        let dev = dev.unwrap();
        let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
        pso_budget::watch_queue(&queue);
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
        let allocs: Vec<Vec<ID3D12CommandAllocator>> = (0..RING).map(|_| (0..LISTS).map(|_| dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap()).collect()).collect();
        let lists: Vec<ID3D12GraphicsCommandList> = (0..LISTS)
            .map(|j| {
                let l: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocs[0][j], None).unwrap();
                l.Close().unwrap();
                l
            })
            .collect();
        let bufs: Vec<Vec<ID3D12Resource>> = (0..RING).map(|_| (0..LISTS).map(|_| buffer(&dev, D3D12_HEAP_TYPE_DEFAULT, 65536, true, D3D12_RESOURCE_STATE_COMMON)).collect()).collect();
        let upload = buffer(&dev, D3D12_HEAP_TYPE_UPLOAD, (WORDS * 4) as u64, false, D3D12_RESOURCE_STATE_GENERIC_READ);
        {
            let mut p = std::ptr::null_mut();
            upload.Map(0, None, Some(&mut p)).unwrap();
            std::ptr::copy_nonoverlapping(words.as_ptr(), p as *mut u32, WORDS);
            upload.Unmap(0, None);
        }
        let slots = count + count.div_ceil(revisit);
        let readback = buffer(&dev, D3D12_HEAP_TYPE_READBACK, (slots * 4) as u64, false, D3D12_RESOURCE_STATE_COPY_DEST);
        let param = [D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            Anonymous: D3D12_ROOT_PARAMETER_0 { Descriptor: D3D12_ROOT_DESCRIPTOR { ShaderRegister: 0, RegisterSpace: 0 } },
        }];
        let mut sig = None;
        D3D12SerializeRootSignature(&D3D12_ROOT_SIGNATURE_DESC { NumParameters: 1, pParameters: param.as_ptr(), ..Default::default() }, D3D_ROOT_SIGNATURE_VERSION_1, &mut sig, None).unwrap();
        let sig = sig.unwrap();
        let rs: ID3D12RootSignature = dev.CreateRootSignature(0, std::slice::from_raw_parts(sig.GetBufferPointer() as *const u8, sig.GetBufferSize())).unwrap();

        // stand-ins: keys only (`pso_async` never touches them; a failed build fails the test)
        let standin = |k: usize| 0x5E00_0000_0000usize + k * 0x40;
        let builds: Vec<Arc<Build>> = (0..count)
            .map(|k| Arc::new(Build { dev: dev.clone(), rs: rs.clone(), dxil: std::mem::take(&mut *dxil[k].lock().unwrap()), blob: Mutex::new(None), first_bytes: Mutex::new(0) }))
            .collect();
        let mut submitted = vec![false; count];
        let mut bound_once = vec![false; count];
        // the order: every program once, then every `revisit`-th again
        let order: Vec<usize> = (0..count).chain((0..count).step_by(revisit)).collect();
        let (mut frame, mut n, mut first_total) = (0u64, 0usize, 0u64);
        let (mut slot_of, mut first_build_us) = (Vec::new(), 0u64);
        let t1 = std::time::Instant::now();
        // pipelines are handed to `pso_async` `AHEAD` dispatches before their first bind: its
        // workers build them meanwhile, as a title's compute pipelines are built ahead (a pipeline
        // built and not bound takes no code in the driver's heap: #2403)
        let ahead = var("AHEAD", 32);
        let mut submit = |k: usize| {
            if !submitted[k] {
                submitted[k] = true;
                let b = builds[k].clone();
                let again = b.clone();
                // (`LAZY=1`: to the low-priority workers that build ahead, as `submit_lazy`;
                // else to the bind workers at once, as `submit`: more at a time)
                pso_async::submit_rebuildable(standin(k), true, Box::new(move || b.make()), 0, var("LAZY", 0) == 1, Box::new(move || again.make()));
            }
        };
        for (c, chunk) in order.chunks(LISTS).enumerate() {
            for &k in order.iter().skip(c * LISTS).take(ahead + LISTS) {
                submit(k);
            }
            let r = (frame as usize) % RING;
            // this ring slot's allocators: the GPU is done with frame - RING
            if frame >= RING as u64 {
                let want = frame + 1 - RING as u64;
                let start = std::time::Instant::now();
                while fence.GetCompletedValue() < want && start.elapsed().as_secs() < 10 {
                    std::thread::yield_now();
                }
            }
            let mut run = Vec::new();
            for (j, &k) in chunk.iter().enumerate() {
                let new = !std::mem::replace(&mut bound_once[k], true);
                let tb = std::time::Instant::now();
                let real = pso_async::resolve_for_bind(standin(k));
                assert_ne!(real, standin(k), "pipeline {k} not built");
                if new {
                    // (the wait for it at its bind: built ahead, mostly)
                    first_build_us += tb.elapsed().as_micros() as u64;
                    first_total += *builds[k].first_bytes.lock().unwrap();
                }
                let alloc = &allocs[r][j];
                if j == 0 {
                    for a in &allocs[r] {
                        a.Reset().unwrap();
                    }
                }
                let list = &lists[j];
                list.Reset(alloc, None).unwrap();
                pso_budget::list_reset(list.as_raw());
                let a = &bufs[r][j];
                list.CopyBufferRegion(a, 0, &upload, 0, (WORDS * 4) as u64);
                transition(list, a, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                list.SetComputeRootSignature(&rs);
                pso_budget::bind(list.as_raw(), real);
                let raw = real as *mut std::ffi::c_void;
                list.SetPipelineState(ID3D12PipelineState::from_raw_borrowed(&raw).unwrap());
                list.SetComputeRootUnorderedAccessView(0, a.GetGPUVirtualAddress());
                list.Dispatch(1, 1, 1);
                transition(list, a, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
                let slot = slot_of.len();
                slot_of.push(k);
                list.CopyBufferRegion(&readback, (slot * 4) as u64, a, 4, 4);
                list.Close().unwrap();
                run.push(Some(list.cast::<ID3D12CommandList>().unwrap()));
            }
            queue.ExecuteCommandLists(&run);
            frame += 1;
            queue.Signal(&fence, frame).unwrap();
            let before = n;
            n += chunk.len();
            let state = dev.GetDeviceRemovedReason();
            if n / 32 != before / 32 || state.is_err() || n == order.len() {
                let s = pso_budget::stats();
                println!(
                    "{n} dispatches ({} pipelines, {:.0} MB of driver blobs built), bound now {:.0} MB + {:.0} MB awaiting release, {} released, {} built again, {:.0} s, device {}",
                    slot_of.iter().copied().collect::<std::collections::HashSet<_>>().len(),
                    mb(first_total),
                    mb(s.code),
                    mb(s.awaiting),
                    s.released,
                    s.revived,
                    t1.elapsed().as_secs_f64(),
                    state.as_ref().err().map_or("ok".into(), |e| format!("removed ({:#x})", e.code().0))
                );
            }
            if state.is_err() {
                std::process::exit(3);
            }
        }
        let start = std::time::Instant::now();
        while fence.GetCompletedValue() < frame && start.elapsed().as_secs() < 30 {
            std::thread::yield_now();
        }
        assert!(dev.GetDeviceRemovedReason().is_ok(), "device removed at the end");
        let mut p = std::ptr::null_mut();
        readback.Map(0, Some(&D3D12_RANGE { Begin: 0, End: slots * 4 }), Some(&mut p)).unwrap();
        let got = std::slice::from_raw_parts(p as *const u32, slots);
        let mut wrong = Vec::new();
        for (slot, &k) in slot_of.iter().enumerate() {
            if got[slot] != expected(k, stmts, &words) {
                wrong.push((slot, k));
            }
        }
        let s = pso_budget::stats();
        let built_first = builds.iter().filter(|b| *b.first_bytes.lock().unwrap() != 0).count().max(1);
        println!(
            "done: {} dispatches of {} pipelines, {:.0} MB of driver blobs built; {} wrong results; first binds waited {:.1} ms each on average for the build",
            slot_of.len(),
            count,
            mb(first_total),
            wrong.len(),
            first_build_us as f64 / 1000.0 / built_first as f64
        );
        println!("budget: {s:?}");
        if s.revived > 0 {
            println!("built again: {} pipelines, {:.2} ms each on average, longest {:.2} ms", s.revived, s.revive_us as f64 / 1000.0 / s.revived as f64, s.revive_max_us as f64 / 1000.0);
        }
        assert!(wrong.is_empty(), "wrong results (slot, program): {:?}", &wrong[..wrong.len().min(20)]);
        if budget.is_some() {
            assert!(s.revived > 0 && s.released > 0, "the budget released and built pipelines again");
        }
    }
}
