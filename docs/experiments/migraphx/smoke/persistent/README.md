# Persistent two-process latency diagnostic

**Diagnostic only; not linked to the production application.** Unchanged
`onnxruntime_go v1.30.1`, isolated ORT 1.25.1 cachefix-v1 build, original FP32
models, gfx1151 validation container. No binding downgrade, model edits,
application worker/IPC changes, or deployment changes.

## Isolation and correctness

The coordinator starts separate persistent encoder and decoder OS processes.
Each owns one ORT session, reused for all requests and shapes. Encoder alone
receives `MIGRAPHX_DISABLE_PASSES=simplify_reshapes`; the decoder variable must be
unset. Known precision/cache/pass overrides are cleared before worker startup.
Private Unix sockets carry gob requests/responses with checked request IDs.
Workers exit with the coordinator, and inference requests have 20-minute limits.

Preprocessing and greedy TDT behavior snapshot the validated real-audio probe.
Natural 2-, 6- and 10-second speech excerpts use source interval start 1 second
from the local `learn-case-interviews.mp3`: no padding, bucketing or truncation.
Encoded-length/channel-major stride equality is mandatory. Decoder resets
target and recurrent state for every transcription, retaining output state only
on nonblank tokens, and runs to full encoded-length completion.

Profiling/numerical validation uses its own persistent process pair before the
timing pair. Two mixed-length rounds compare every encoder value, decoder logit
and state output against CPU at `atol=1e-3`, `rtol=1e-4`, requiring exact
transcripts and token/duration/frame trajectories. The six-second CPU trace
also must match the earlier validated diagnostic. Profile checks require the
exact ORT model-run count and exclusively the expected provider's attribution;
this is ORT attribution, not a hardware trace.

Timed requests require exact CPU transcript/trajectory equality. Full numerical
trace copying and profiling are disabled in timing workers. The GPU timing
phase must leave cache hashes unchanged and have no compilation markers.

## Timing methodology

For each provider and fixture: one first-shape request, three warmups, then 30
steady requests. Then ten mixed-length cycles (2,6,10 seconds) provide ten
observations per fixture under shape switching. One sequential request at a
time; CPU and GPU measured separately on the same machine with four ORT
intra-op threads per worker. CPU runs first; there is no randomized/repeated
trial design, confidence interval, or concurrency/throughput test.

The end-to-end clock starts with an **in-memory WAV** and ends when the full
transcript/trajectory response returns. It includes WAV parsing, mel extraction,
tensor setup, float feature transfers over both Unix sockets, inference, output
handling and transcript rendering. Input disk reads, correctness comparisons,
measurement persistence, worker startup and profiling are outside that clock.
It is not HTTP/service latency. RPC timing includes worker execution; the
reported residual is IPC plus worker/coordinator overhead, not IPC alone.

First-shape, warmup and profiled-validation observations are excluded from warm
p50/p95. Nearest-rank percentiles are computed separately for each fixture,
provider and schedule. Thirty steady and ten mixed samples give descriptive
percentiles only; mixed p95 is effectively the maximum observed value.

Worker startup/session creation and first native inference are reported
separately. Previously validated encoder/decoder caches are copied to dedicated
benchmark directories without modifying their originals. The 2- and 10-second
shapes may compile during validation; compilation does not count as warm
request latency. Switching shapes in the timed phase is deliberately included.

## Build / evidence

```sh
cd docs/experiments/migraphx/smoke
GOPROXY=off GOSUMDB=off go test ./...
GOPROXY=off GOSUMDB=off go vet ./...
GOPROXY=off GOSUMDB=off go test -race ./persistent
GOPROXY=off GOSUMDB=off go build -o ../ort125-persistent-probe ./persistent
```

Never rebuild an active binary. Runner: `diagnostics/run-persistent-v1.py`.
Fresh, one-shot evidence directory: `/work/persistent-v1/`, including provenance,
worker logs/profiles, validation gob records, per-phase timing files,
`result.json` or `failure.json`, and launcher/application-health notes.

**v1 status: FAILED, not a GPU performance pass.** CPU validation and all CPU
measurements completed. Initial GPU numerical/transcript checks passed for all
three lengths, but the encoder was OOM-killed on the subsequent 10-to-2-second
shape switch at the container's unchanged 24 GiB cap. GPU timing never began;
GPU profile files were not flushed. See `persistent-v1/failure-summary.md`.
The interrupted foreground preflight remains preserved in
`/work/persistent-v1/interrupted-preflight-1/`; it never started inference.

**Cache-only v2 retry:** `diagnostics/run-persistent-cacheonly-v2.py` uses the
unchanged pinned binary, identical fixtures, fresh processes, and copies of all
three precompiled encoder caches plus the decoder cache. Evidence goes to the
fresh `/work/persistent-cacheonly-v2/` directory. The 24 GiB container cap stays
unchanged; sampled memory reaching 22 GiB, a new OOM kill, or compilation markers
in either worker's logs stop the owned coordinator. One-second monitoring can
miss short spikes; the guard is not a guaranteed prevention of every OOM.
Six launcher-guard and thirteen summary-checker regression tests passed.

**v2 status: GUARD STOPPED, not a GPU performance pass.** Fresh cache-only GPU
workers completed the 2->6->10->2-second validation sequence, then reached
22.396 GiB sampled container usage during the next 6-second encoder request.
The guard terminated the coordinator; no new OOM kill occurred, cache hashes
stayed unchanged, and neither worker logged compilation. GPU timing never
started and profiling files were not flushed. Compilation is not required for
the observed memory growth; the responsible allocation/leak is not established.
See `persistent-cacheonly-v2/failure-summary.md` and `failure-investigation.json`.
Cleanup and production health/diff checks passed. No further retry was launched.

After the launcher completes, independently check the recorded sample counts,
percentiles, trajectories, timing boundaries, profiles, cache hashes, and
production-unchanged gate using:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 diagnostics/test-persistent-summary.py
PYTHONDONTWRITEBYTECODE=1 python3 diagnostics/test-persistent-cacheonly-v2.py
PYTHONDONTWRITEBYTECODE=1 python3 diagnostics/summarize-persistent-v1.py --root /path/to/complete-run-evidence
```

Run these commands from `docs/experiments/migraphx`. The summary command
requires completed evidence and creates `performance-summary.md` in the supplied root
exclusively; it cannot overwrite an existing report. Tests use synthetic data
and never launch inference. No GPU performance or production-readiness pass is
claimed before completion. Cache-only results do not establish cold-compilation
performance or erase the failed v1 run. The background launch wrapper runs the
checker only after a successful retry and retains `independent-check.log`.
