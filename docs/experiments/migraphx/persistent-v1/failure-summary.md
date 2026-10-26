# Persistent benchmark v1: failed, not a performance pass

CPU validation and all CPU latency measurements completed. Initial GPU numeric
and transcript checks passed for 2-, 6- and 10-second excerpts, but the second
validation round failed on the 10-second-to-2-second encoder shape switch with
`EOF`. The GPU timing phase never began; GPU profile files were not flushed
and are zero bytes. No full-GPU attribution or warm p50/p95 claim is valid.

The validation container retained its 24 GiB memory limit. `memory.events`
changed from `oom_kill 0` to `oom_kill 1`; peak usage equalled 25,769,803,776 bytes.
Kernel logs identify the victim as host PID 1639131, the GPU encoder worker,
in the same libpod cgroup. Its anonymous RSS at termination was 21,965,988 KiB,
plus 676,944 KiB file-backed RSS. This confirms a memory-cgroup OOM kill rather
than a socket-protocol error. See `kernel-oom-investigation.log` and
`failure-investigation.json`. The initial investigation lacked the killed PID;
the kernel log subsequently resolved it.

First native encoder calls: 2 s fixture 278.711 s including compilation;
6 s fixture 3.723 s using an existing cache; 10 s fixture 904.602 s including
compilation. These are first-call costs, not warm latencies.

The inspected provider stores a single active program in its function state.
On shape mismatch it loads the appropriate disk cache into that program or
compiles on miss. It does not implement an in-memory per-shape program pool.
`prog = migraphx::load(...)` constructs the replacement before assignment;
potential transient old/new allocation overlap and compiler-memory retention
are hypotheses, not established allocation-level root causes.

Next diagnostic: fresh processes with all three precompiled caches, same binary
and audio, unchanged 24 GiB cap. Monitor memory and stop at 22 GiB or unexpected
compilation. Preserve this failed evidence and use a separate output directory.
Do not label the cache-only retry as cold compilation performance.

Launcher confirmed unchanged application diff fingerprint and production
health `{"status":"ok"}`. No application/provider/library/deployment changes.
