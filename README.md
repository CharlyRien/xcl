# Lab patches: XCL's self-made waits off the submit path (RTX 3090)

23 patches (`git am patches/*.patch`). 0001-0014 as first posted; 0015-0023 added on 27/11 (fixes from an independent review of 0005-0014, a mesh pipeline fix, and #3000 behind a switch). Every new behavior sits behind its own switch and is off by default, except 0021 (a fix, see below). Context and numbers: xcl-dev/xcl#505, comment 7215587476.

**Base they apply to:**
- MacieGx's lab tree `59986ef`;
- plus `47d0eac` (`XCL_CHAIN_WHOLE`, cherry-picked clean);
- plus #2934's `7692dd87` and `c230aa1b` (`XCL_WATCH_TRAP`). These had two conflicts, resolved by hand: the `pub mod watch_trap;` line in `mod.rs`, and `refresh_once` in `title_textures.rs`, where the lab's `XCL_LOOK_JOIN` / `XCL_WATCH_HOT` branches are kept.

On a different base, expect conflicts in `title_textures.rs` and `sparse.rs`.

| patches | switch | what it does |
|---|---|---|
| 0001-0004 | `XCL_THREAD_SAMPLE=<from s>,<ms>,<count>` (+ `XCL_THREAD_SAMPLE_FILE`), `XCL_CALL_STATS=1` | thread sampler (RIP + 7 return addresses, cycle time per thread) and per-call counters, a log line every 5 s; no admin needed |
| 0005 | `XCL_FENCE_DIRECT=1` | XCL's own fence probes read the fence directly instead of going through the title's `GetCompletedValue` hook and its 2 ms `try_state` spin |
| 0006 | `XCL_SPARSE_MAP_NOWAIT=1` | a sparse `Map` copy-back runs on its own copy queue, ordered after the sparse queue, instead of behind the title's 3D work |
| 0007, 0009 | `XCL_USE_RANGE_HIT=1` | `use_range` takes the sparse `STATE` lock only for an address in a lazily placed buffer; each V# is checked once per bind |
| 0008 | `XCL_FENCE_FAST=1` | `fence_completed` without a `VirtualQuery` per call (not measured in a run yet) |
| 0010 | `XCL_FILTER_INDEX=1` | texture filter through a sorted index of the memory textures are read from |
| 0011 | `XCL_FENCE_DIRECT=2` | variant that also lands finished copy-backs; slower here, kept for reference |
| 0012-0014 | `XCL_SPARSE_SCATTER=1` | sparse uploads as one scatter dispatch instead of one `CopyBufferRegion` per run. Its tables are created outside XCL's hooks and released outside the sparse lock, which fixes a lock-order hang seen in one run |
| 0015 | `XCL_FILTER_INDEX=1` | a texture release moves the index generation while the texture table is held, so a re-activated alias is not missed |
| 0016 | `XCL_SPARSE_SCATTER=1` | the scatter root signature and pipeline are made with the device's original creators, not through XCL's hooks; a failure falls back to the copies |
| 0017 | `XCL_SPARSE_SCATTER=1` | a slot's scatter tables that the GPU may still read (the 5 s wait gave up) are held until the sparse fence passes, not rewritten or released |
| 0018 | `XCL_SPARSE_MAP_NOWAIT=1` | an unmap makes the sparse queue wait on the GPU for the Map copies queued before it, so a copy cannot lose its tile or heap |
| 0019, 0020 | – | the activity thread's fence line only with `XCL_CALL_STATS`; tests that touch process-wide lab state are serialized |
| 0021 | none (a fix) | translated mesh pipelines no longer get the topology subobject twice. The runtime refused every one (E_INVALIDARG, 0 ms), so all were built "without topology" with no driver blob, and each rebuild after `XCL_PSO_UNUSED_S` gave it back was a full 4-7 s compile on the binding thread (a 75 s freeze after a move here). A stream without a topology gets the same bytes as before |
| 0022 | `XCL_PSO_KEEP_BLOBLESS=1` (log lines always) | `built again at bind:` line for a rebuild of 1 s or more, `given back without a blob` counter; the switch keeps blobless pipelines from the unused delay (not the budget). Not needed after 0021 |
| 0023 | `XCL_TEX_KEY_PAD=1` | #3000 (voiceovarian1-max) behind a switch: view-by-address textures keyed without the stack padding at desc bytes 4..8; a `lab tex key:` counter line every 10 s either way |

**Runs here with all 23 (one run each, stable window 316 s to the end, 420 s in the world):**
- MacieGx's #505 env + `XCL_PAGE_BUDGET_MB=4096 XCL_LOOK_JOIN=0 XCL_CHAIN_WHOLE=1 XCL_WATCH_TRAP=1 XCL_CALL_STATS=1` + `XCL_FENCE_DIRECT=1 XCL_SPARSE_MAP_NOWAIT=1 XCL_USE_RANGE_HIT=1 XCL_FILTER_INDEX=1 XCL_SPARSE_SCATTER=1`:
  - 0001-0020: 3.04 presents/s, green texels 0.000 %;
  - + 0021-0022: 3.12, no rebuild of 1 s or more, rebuilds 25.6 s in all (117 s in the frozen run);
  - + 0023 with `XCL_TEX_KEY_PAD=1`: 3.50, 74,007 pad-only duplicate textures served instead of made, GPU busy median 287 → 245 ms, 36 % fewer texture evictions, character heads drawn again (still blue-lit).
- 0 device removals and 0 GPU wait timeouts in every run with these patches here (about 15 runs); one earlier scatter run stalled on a lock-order deadlock, which 0014 fixed.
- Two runs of the same setup here differed by ±7 % in presents/s, so read single-run rates as a direction.

The patches touch none of build.rs `TRANSLATOR_SOURCES`, so the translated shader cache stays valid.

Measured on: RTX 3090 (Resizable BAR off), driver 616.56, Ryzen 9 5950X, 32 GB RAM, Windows 11. One run per setup.

AI use: Claude Code, model Claude Opus 5.5, wrote these patches and this file. The person running the tests asked for them to be shared.
