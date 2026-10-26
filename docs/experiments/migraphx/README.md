# Paused MIGraphX experiment — handoff, 2026-10-26 UTC

**Archive only; not production-ready. Keep the working ROCm deployment as the
baseline. No further GPU experiments were run to prepare this branch.**

## Preserved state

This snapshot starts from application commit `436daa8` on `feat/migraphx-ort125`
and preserves its uncommitted provider/configuration, encoder/VAD serialization,
and tests. It also includes selected standalone probes, offline regressions,
runtime patch provenance, and failure summaries. CPU/CUDA behavior is retained;
the binding remains `github.com/yalue/onnxruntime_go v1.30.1`.

The separate ROCm worktree/deployment was not modified. No PR comment, PR closure,
new release, or production deployment is part of this archive.

## What worked and what failed

| Area | Finding | Limit |
| --- | --- | --- |
| Runtime | Local ORT 1.25.1 loaded with the unchanged binding; ROCm 7.2.4, MIGraphX 2.15.0, Radeon 8060S `gfx1151` | Older packaged ORT 1.23.2 is incompatible with the binding's API 25 |
| Original encoder | Unpatched provider rejects 48 `Pad` nodes with omitted optional inputs and falls back to CPU | Registration alone does not establish GPU coverage |
| Provider patch | Omitted `Pad` values and complete versioned `shape-v2` keys fixed tested coverage/cache defects | No production runtime replacement |
| Synthetic encoder | 18 T128/T256 comparisons across sessions/processes passed; exclusively MIGraphX profile events | Encoder requires process-local `MIGRAPHX_DISABLE_PASSES=simplify_reshapes` |
| Cache behavior | Tested cross-process replay retained hashes; warm 21–23 ms; shape loads roughly 3–6 s | Encoder caches about 2.36/2.38 GB; cold compilation minutes; not arbitrary-shape validation |
| Decoder | Default-pass checks and independent-process replay passed | Encoder workaround instead triggered concat compiler errors and a native crash |
| Real audio | Six-second excerpt (598 natural mel frames) passed numerical checks, exact full transcript/trajectory, GPU attribution, and replay | Separate diagnostic OS processes; not integrated into the application |
| Persistent v1 | CPU phases and initial 2/6/10-second GPU checks completed; encoder OOM-killed on 10→2 switch | Unchanged 24 GiB cap; GPU timing never started |
| Cache-only v2 | Fresh workers passed 2→6→10→2, stopped on next 6-second encoder request at 22.396 GiB sampled usage after 52.76 s | Exit 143; unchanged caches; no compilation markers/new OOM kill; GPU timing never started |

The earlier approximately 4.8× warm six-second **encoder-only** comparison is a
narrow observation. There is no completed persistent GPU latency distribution,
HTTP/service benchmark, concurrency/throughput result, or deployable end-to-end
performance improvement. Profiles establish ORT attribution, not hardware tracing.

v2 container-aggregate memory is not per-worker RSS or a continuous high-water
measurement. It shows compilation is not necessary for the observed growth, but
does not prove a leak, responsible allocation/process, or old/new-program overlap.
Cleanup confirmed no active owned benchmark workers (remaining entries were
zombies); about 3.65 GiB remaining usage was almost entirely file cache. Historical
production health returned `{"status":"ok"}`.

## Snapshot map

- `runtime/`: exact validated provider patch/header, later working patch,
  build identity/library hashes, and original build procedures.
- `smoke/`: separate Go module with native-loading, encoder, decoder, real-audio,
  and persistent probes. Not imported into the production application.
- `diagnostics/`: cache regression, strict profile checker, one-shot historical
  launchers, independent summary checker, offline tests, and staged findings.
- `persistent-v1/failure-summary.md` and
  `persistent-cacheonly-v2/failure-summary.md`: final failure findings.
- `persistent-cacheonly-v2/failure-investigation.json`: selected investigation
  facts, not a complete raw run/evidence bundle.
- `snapshot-manifest.json`: original copied source hashes **before** path and
  archival documentation edits. It is not a final-tree checksum manifest.

Generated binaries, runtime libraries, ONNX models/external weights, compiled
`.mxr` caches, audio excerpts, gob traces, raw profiles/logs, and benchmark output
are intentionally excluded. Existing reference audio already tracked by the
application was not added or removed. Full raw evidence remains in the original
external experiment workspace; this branch is not a complete portable rerun.

Historical `/work` paths denote the container-mounted experiment workspace,
`/models` denotes read-only model mounts, and `/repo` in provenance denotes the
application checkout. Copied narrative reports describe stages at their original
time; this document and the failure summaries supersede their “next” suggestions.
The experiment-invariance tracked-diff fingerprint
`99a820cff4f5bde49facc238419004e6f3ee7729549cb3b2e956a11e15c735b9`
predates archival edits and is not a fingerprint of this branch.

## Offline checks only

From the application root, with Go dependencies already cached:

```sh
GOPROXY=off GOSUMDB=off go test ./...
GOPROXY=off GOSUMDB=off go vet ./...
cd docs/experiments/migraphx/smoke
GOPROXY=off GOSUMDB=off go test ./...
GOPROXY=off GOSUMDB=off go vet ./...
GOPROXY=off GOSUMDB=off go test -race ./persistent
```

From `docs/experiments/migraphx`, run each offline Python test:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 diagnostics/test-realaudio-coverage.py
PYTHONDONTWRITEBYTECODE=1 python3 diagnostics/test-realaudio-runner.py
PYTHONDONTWRITEBYTECODE=1 python3 diagnostics/test-persistent-summary.py
PYTHONDONTWRITEBYTECODE=1 python3 diagnostics/test-persistent-cacheonly-v2.py
```

These tests use synthetic/temporary evidence and do not launch GPU inference.
For the C++ regression, use the explicit paths in `runtime/README.md`.

Archival checks completed: application tests/vet/race and build; standalone
module tests/vet and persistent race test; all 31 offline Python regressions;
Go formatting and Git whitespace checks; JSON/source/artifact audit; both
provider patches applied against pristine base source with the exact difference
verified. The C++ cache regression passed in the existing GPU-less build
container because the host lacked `g++`. Build outputs went outside the checkout.
Opt-in native/GPU tests were not enabled, and no GPU inference was launched.

**Do not run historical launchers merely to test this archive.** Their host root
is now derived from their location (or `MIGRAPHX_EXPERIMENT_ROOT`), but they retain
the original container layout, exact binary/model/fixture/cache hashes, and
one-shot labels. `MIGRAPHX_APP_REPO` and `MIGRAPHX_SOURCE_AUDIO` configure the
application/source-audio paths for the historical persistent launcher. A rebuilt
binary requires a new run label, fresh directories, and reviewed updated hashes;
do not relax checks just to pass. The summary checker requires `--root` and
rejects incomplete evidence; it cannot produce a valid report from these failures.

## Resuming later requires a separate decision

1. Attribute shape-switch allocations using per-process and native allocation
   diagnostics, with the same resource cap and explicit stop guards.
2. Resolve the opposing encoder/decoder compiler environments; do not toggle
   process-global environment state around application sessions or runs.
3. Broaden shape/audio/VAD/concurrency correctness before proposing integration.
4. Complete independently checked GPU timings before making performance claims.

Historical upstream leads, **not confirmed root causes**:

- AMDMIGraphX issue #4164 / PR #4621: constant-convolution expansion; reported
  `.mxr` reduction 2.0 GiB→2.3 MiB with a later correctness regression noted.
- legacy-rocm-build issue #6573: approximately 0.9 KiB/copy HIP leak on 7.1.1,
  not reproduced on 7.2.4/7.13+; not directly established for this stack.
- AMDMIGraphX PRs #4443 and #5360: possible shape-transform leads only.

No exact matching upstream cache-only Parakeet failure or allocation-level cause
was confirmed. These references were recorded during the experiment, not
revalidated while archiving it.
