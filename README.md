# Lab patches: XCL's self-made waits off the submit path (RTX 3090)

14 patches (`git am patches/*.patch`). Every new behavior sits behind its own switch and is off by default. Context and numbers: xcl-dev/xcl#505, comment 7215587476.

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

**Fastest env here (3.15 stable presents/s):**
- MacieGx's #505 env;
- `XCL_PAGE_BUDGET_MB=4096 XCL_LOOK_JOIN=0 XCL_CHAIN_WHOLE=1 XCL_WATCH_TRAP=1`;
- `XCL_FENCE_DIRECT=1 XCL_SPARSE_MAP_NOWAIT=1 XCL_USE_RANGE_HIT=1 XCL_FILTER_INDEX=1 XCL_SPARSE_SCATTER=1`;
- `XCL_CALL_STATS=1` was on in the measured runs.

The patches touch none of build.rs `TRANSLATOR_SOURCES`, so the translated shader cache stays valid.

Measured on: RTX 3090 (Resizable BAR off), driver 616.56, Ryzen 9 5950X, 32 GB RAM, Windows 11. One run per setup.

AI use: Claude Code, model Claude Opus 5.5, wrote these patches and this file. The person running the tests asked for them to be shared.
