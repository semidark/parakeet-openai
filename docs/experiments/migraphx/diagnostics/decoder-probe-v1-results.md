# Decoder comparison probe — initial result, 2026-10-26 UTC

## Delivered

- Go source/tests/documentation: `/work/smoke/decoder/`
- Binary: `/work/ort125-decoder-probe`
- GPU diagnostic runner: `/work/run-gpu-decoder-v1.sh`
- Strict provider-attribution checker: `/work/diagnostics/check-decoder-profiles.py`
- CPU fixture: `/work/references/decoder-v1.gob`

Stack: unchanged Go binding v1.30.1, ORT 1.25.1 with isolated `cachefix-v1`
provider; original FP32 decoder; GPU diagnostic uses the process-local
`MIGRAPHX_DISABLE_PASSES=simplify_reshapes` workaround. Model mounts are read-only.
Encoder features are from deterministic synthetic mel input, **not real audio**.

## Verified

Offline `go test ./...`, `go vet ./...`, and build passed. Unit tests cover
per-element tolerance, non-finite values, output decisions, channel-major
feature extraction, blank/nonblank recurrent-state feedback and frame-advance
policy, strict vocabulary parsing, and refusal to overwrite CPU fixtures.
Profile-checker regression accepted exclusively GPU-attributed test events and
rejected fallback, mixed/missing attribution, wrong run counts, and missing files.

CPU: 16 forced-target steps plus 13 greedy steps. Replay and recurrent self-checks
had **zero differences** in logits and both state outputs. All token/duration
decisions matched. Profile: 87 model runs, 2001 CPU kernel events, no other
provider attribution. Representative recurrent native-Run p50/p95:
619.745/658.147 microseconds (profiling enabled, not end-to-end latency).

## GPU blocked

Artifact/model/feature hashes verified. MIGraphX session creation succeeded in
675.691639 ms. On first replay Run, generated vectorized `concat` kernel
compilation failed for gfx1151:

```text
integral_constant.hpp:140:15: error: non-type template argument is not a constant expression
concat.hpp:41:26: note: read of dereferenced one-past-the-end pointer is not allowed in a constant expression
1 error generated when compiling for gfx1151.
SIGSEGV: segmentation violation
signal arrived during cgo execution
```

Runner exit 2. No GPU decoder output or successful comparison. One zero-byte
profile; no compiled decoder cache file was found. This is a native GPU-stack
compilation/crash blocker, not evidence of a floating-point accuracy mismatch.
The exact crash cause after the compiler errors has not been independently
isolated. No repair or additional workaround was applied.

Evidence: `/work/decoder-cpu-v2.log`, `/work/decoder-gpu-v1.log`,
`/work/decoder-gpu-v1.exit`, and `/work/profiles/decoder-cpu-v1/`.
The earlier `/work/decoder-cpu-v1.log` records a strict metadata-check rejection
of the extra `prednet_lengths` output; source was corrected to accept unused
outputs while validating the three application-consumed outputs.

Production `http://127.0.0.1:5093/health` returned `{"status":"ok"}` after the
isolated crash. Application source, deployment, model weights and production
configuration were not modified by this diagnostic task.

## Follow-up: default pass pipeline succeeds

Controlled comparison runner: `/work/run-gpu-decoder-no-workaround-v1.sh`.
Same probe binary, model/features, patched runtime, tolerances, session/repeat
counts and arena limit; `MIGRAPHX_DISABLE_PASSES` explicitly unset and fresh
cache/profile/log paths. Log: `/work/decoder-gpu-no-workaround-v1.log`.

- Numerical/decision checks passed: 29 samples, two sessions, two repetitions
  of replay and recurrent modes, 232 comparisons total. All token/duration
  decisions matched CPU references.
- Maximum absolute differences across comparisons: logits 9.15527e-05,
  state1 7.18236e-06, state2 2.09808e-05.
- Both profiles passed strict attribution: 116 model runs and 116 MIGraphX
  kernel events per session; no CPU-attributed kernels. ORT attribution,
  not a hardware trace.
- First native Run including compilation: 5.846604228 seconds. Subsequent
  native-Run p50 values approximately 0.410–0.418 ms, profiling enabled.
- A 72,608,207-byte decoder cache artifact was produced. Second-session
  success is not independent-process proof of cache reuse.
- Production health remained `{"status":"ok"}`. No deployment changes.

The crash was **not reproduced with default passes**. This controlled comparison
strongly implicates disabling `simplify_reshapes` as the trigger for the failing
decoder compilation path on this stack; the generated concat defect and ensuing
native crash mechanism remain unisolated. It does not test an unpatched runtime
or real-audio transcription. Do not apply the encoder workaround globally to
decoder processes based on the initial run.

## Independent-process decoder cache replay

Runner: `/work/run-gpu-decoder-cache-replay-v1.sh`. A new process reused the
default-pass disk cache with the same binary, runtime, model, fixture and settings;
one session, two repetitions of replay/recurrent modes. Fresh profile/log/exit
paths preserve the cold-run evidence.

- All 116 numerical/decision comparisons passed.
- Strict profile check passed: 116 model runs and 116 MIGraphX kernel events,
  no CPU-attributed kernels (ORT attribution, not hardware tracing).
- No `Model Compile: Begin` marker appeared. Inspection of the pinned provider
  source shows both compilation paths use `compile_program`, which always logs
  this marker at WARNING. A successful precompiled-model load bypasses it.
  Together with the fresh process and existing disk artifact, this verifies
  decoder disk-cache reuse for the tested fixture/settings.
- First native Run: 143.265159 ms, versus cold 5.846604228 s. Session creation:
  669.529046 ms. Warm native-Run p50 values: 0.410–0.419 ms. These are profiling-
  enabled stage timings, not end-to-end transcription latency.
- The sole 72,608,207-byte MXR retained SHA256
  `af2d0e8419b94deabfd42c7124bf84d694738c7a7a2eae51cc883c49567a74d4`.
- Evidence: `/work/decoder-gpu-cache-replay-v1.log`,
  `/work/decoder-gpu-cache-replay-v1.exit`,
  `/work/decoder-gpu-cache-replay-v1.cache-before.sha256`, and
  `/work/profiles/decoder-gpu-cache-replay-v1/`.
- Production health remained `{"status":"ok"}`. No application source or
  production deployment was changed.

## Next (updated after independent-process replay)

Default-pass GPU decoder numerical correctness, ORT provider attribution,
and independent-process disk-cache reuse are now verified for this synthetic
fixture. Next evaluate an isolated real-audio pipeline with separate compiler
environments. The application-boundary inspection and safe diagnostic plan are
in `/work/diagnostics/encoder-decoder-isolation-v1.md`. Investigate the concat bug
separately if disabling `simplify_reshapes` for the decoder is required.
