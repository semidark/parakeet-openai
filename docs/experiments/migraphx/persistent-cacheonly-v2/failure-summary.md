# Cache-only persistent benchmark v2: resource-guard stop

**Not a GPU performance pass. No warm GPU p50/p95 results were obtained.**

The unchanged pinned binary ran with fresh encoder/decoder processes, identical
2-, 6- and 10-second fixtures, and copies of all precompiled caches from v1.
The container memory cap stayed at 25,769,803,776 bytes (24 GiB); the launcher
used a sampled stop threshold of 23,622,320,128 bytes (22 GiB).

## Observed outcome

- CPU validation and all steady/mixed CPU measurements completed.
- GPU numerical and exact transcript/trajectory checks completed for requests
  2 s, 6 s, 10 s, then 2 s again. Four validation gob records were persisted.
- The guard stopped the owned coordinator during the fifth encoder request,
  switching back to 6 s. Exit was 143 (SIGTERM), not an OOM-kill exit.
- Peak sampled container usage was 24,047,341,568 bytes (22.396 GiB), observed
  52.76 seconds after coordinator launch. The one-second sampling guard is not
  a continuous high-water measurement and cannot guarantee prevention of spikes.
- `oom_kill` remained 1, the pre-existing count from v1: no new OOM kill.
- Encoder and decoder logs contain no `Model Compile: Begin` markers. The
  launcher verified cache hashes unchanged from the pinned seed copies.
- GPU profiling files are zero bytes because normal session teardown did not
  occur. Full provider attribution and the GPU timing phase did not complete.
- First cache-only encoder calls for the completed sequence took approximately
  3.663, 3.822, 3.999 and 4.078 seconds. These include cache loading on shape
  changes and are not same-shape warm inference latency measurements.

## Interpretation and limits

Fresh-process cache-only execution reproduced memory growth sufficient to hit
the guard. Cold compilation is therefore not required for this observed failure
mode. This does not exclude compiler-memory retention as a contributor in v1.

The inspected ORT MIGraphX provider holds one active program in its function
state and attempts a disk-cache load on shape mismatch. It does not retain an
in-memory per-shape program pool. Loading a replacement constructs the new
program before assignment, so transient old/new overlap is a hypothesis.
Container-aggregate `memory.current` samples alone do not identify the responsible
worker/allocation, prove a leak, or distinguish live allocations, allocator
retention, file cache and replacement overlap at the time of the peak.

After shutdown, a follow-up snapshot showed roughly 3.65 GiB remaining, almost
entirely file cache: anonymous memory was 98,304 bytes. Both terminated worker
entries were confirmed zombies under container PID 1; no owned worker was
running. The container was not restarted or otherwise modified to reap them.

## Production and evidence

Launcher and follow-up checks confirmed production health `{"status":"ok"}`
and the unchanged tracked-diff SHA256:
`99a820cff4f5bde49facc238419004e6f3ee7729549cb3b2e956a11e15c735b9`.
No application, model, provider library, binary, or deployment changes occurred.

Evidence: `launcher-result.json`, `launcher-failure.json`, `memory-samples.jsonl`,
`failure-investigation.json`, coordinator/worker logs, validation gob records,
and provenance/cache hashes. The independent performance report was deliberately
not generated. The failed v1 evidence remains intact.

## Proposed next diagnostic (not launched)

Compare repeated fixed-shape requests with controlled shape switches in separate
fresh workers. Record per-worker RSS/anonymous memory, container anonymous/file
memory, and request boundaries before/during/after cache loading and inference.
Preserve compiler-environment isolation, existing memory cap/guard, cache hashes,
and correctness gates. Do not raise the cap, repeat the identical benchmark, or
claim general persistent-worker performance until the memory behavior is bounded.
