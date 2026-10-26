# Standalone decoder CPU/MIGraphX comparison probe

Diagnostic only. This command is outside the application source tree and does
not change production, model weights, provider defaults, or the Go binding.
Requires ONNX Runtime **1.25.1** and `onnxruntime_go` **v1.30.1** unchanged.

## What it checks

- Original FP32 decoder interface and vocabulary, including symbolic model
  dimensions resolved to the application's fixed batch-one input shapes.
- Every output logit and both `[2,1,640]` recurrent-state outputs. The model's
  additional `prednet_lengths` output is not requested, matching the application.
- Exact token and duration argmax decisions (first index wins ties).
- Per-element error: `abs(got-ref) <= atol + rtol*abs(ref)`, defaults
  `atol=1e-3`, `rtol=1e-4`; non-finite values always fail.
- CPU-reference replay with identical inputs, separately from GPU-state
  recurrent rollout so drift cannot be hidden by teacher-forced CPU states.
- A forced-target sequence with state feedback on every step, and a greedy
  sequence matching the application's blank/state, frame-advance, and
  ten-tokens-per-frame policy. The forced sequence is deliberately **not**
  ordinary transcription: it exercises state when synthetic features emit blank.
- Model, vocabulary, and encoder-feature SHA256 identity stored in fixtures.
- Separate native-Run timings and ORT profiling. A numerical pass **does not**
  establish GPU execution. Check profiles before claiming GPU coverage.

All inference is serialized. Input/output tensors are reused, as in the
application decoder worker. Fixture creation refuses to overwrite an existing
file. Each invocation requires a fresh profiling directory. Cache directories
must be dedicated to the decoder and runtime under test.

## Build and unit tests (host)

```sh
cd docs/experiments/migraphx/smoke
GOPROXY=off GOSUMDB=off go test ./...
GOPROXY=off GOSUMDB=off go vet ./...
GOPROXY=off GOSUMDB=off go build -o ../ort125-decoder-probe ./decoder
```

## CPU fixture (validation container)

The existing `encoder-T256.gob` contains CPU encoder output from deterministic
**synthetic mel inputs**, not real audio. The probe also accepts other encoder
probe gob files with shape `[1,1024,T]` and a valid encoded length. The actual
source/provenance of new features must be documented separately.

For a new run, choose unused fixture/profile paths:

```sh
podman exec \
  -e LD_LIBRARY_PATH=/work/artifacts/cachefix-v1/lib:/opt/rocm/lib \
  -e ONNXRUNTIME_LIB=/work/artifacts/cachefix-v1/lib/libonnxruntime.so.1.25.1 \
  ort125-migraphx-validate nice -n 19 \
  timeout --signal=TERM --kill-after=10s 120s /work/ort125-decoder-probe \
  -model /models/decoder_joint-model.onnx \
  -features /work/references/encoder-T256.gob -vocab /models/vocab.txt \
  -provider cpu -references /work/references/decoder-v1.gob \
  -profiles /work/profiles/decoder-cpu-v1 -steps 16
```

CPU writes two sequences then self-checks replay and recurrent modes. If the
fixture contains N samples, the CPU profile must contain **3*N** model runs.
Greedy decoding may reach the end before the step limit, so obtain N from
`CPU FIXTURE: PASS samples=N` rather than assuming twice `-steps`.

## MIGraphX comparison

The preferred cold-run diagnostic is
`/work/run-gpu-decoder-no-workaround-v1.sh`. It verifies hashes, refuses existing
result/cache/profile paths, clears known precision/cache environment overrides,
and explicitly unsets `MIGRAPHX_DISABLE_PASSES`. It uses the isolated patched
provider, not an unpatched runtime. The original `/work/run-gpu-decoder-v1.sh`
disables `simplify_reshapes` and reproduced a native crash; retain it as failure
evidence, not the recommended decoder configuration. The arena limit is not a
total GPU-memory cap. Run only when no other GPU compiler workload is active.
The supplied v1 paths already exist; choose fresh paths for new cold runs.

```sh
podman exec ort125-migraphx-validate sh /work/run-gpu-decoder-no-workaround-v1.sh
```

The runner performs two sessions, two repetitions of both comparison modes.
For N fixture samples, each session produces **4*N** model runs. GPU uses the
step count stored in the fixture; the `-steps` flag controls only CPU generation.
Session recreation alone does not prove disk-cache reuse; inspect artifacts and
perform an independent-process replay before claiming it.

## Provider coverage

For the initial fixture, N=29:

```sh
podman exec ort125-migraphx-build python3 /work/diagnostics/check-decoder-profiles.py \
  /work/profiles/decoder-cpu-v1 --provider cpu --sessions 1 --runs-per-session 87
podman exec ort125-migraphx-build python3 /work/diagnostics/check-decoder-profiles.py \
  /work/profiles/decoder-gpu-no-workaround-v1 --provider migraphx --sessions 2 --runs-per-session 116
podman exec ort125-migraphx-build python3 /work/diagnostics/check-decoder-profiles.py \
  /work/profiles/decoder-gpu-cache-replay-v1 --provider migraphx --sessions 1 --runs-per-session 116
```

The checker reports counts for every provider, checks the exact number of
`model_run` events, and fails full-GPU validation on CPU fallback, mixed or
missing provider attribution, missing profiles, or missing kernel events.
Kernel event counts can exceed native Run counts when a graph is partitioned.
ORT attribution is not a hardware trace. Compare CPU/GPU latency cautiously:
profiling is enabled and measurements are native Run only, not transcript latency.

## Current validation

- Build, unit tests, vet, and profile-checker syntax: passed.
- CPU original-model fixture: passed, 29 samples (16 forced, 13 greedy).
- CPU replay/recurrent self-checks: zero differences; profile contains 87 native
  model runs and 2001 CPU-attributed kernel events.
- MIGraphX with `simplify_reshapes` disabled: **blocked** on first native Run. Session creation
  succeeded (~676 ms), then generated `concat` kernel compilation for `gfx1151`
  failed with `non-type template argument is not a constant expression`
  (`concat.hpp`: one-past-the-end dereference), followed by native `SIGSEGV`
  during cgo execution. Runner exit code: 2. No GPU decoder output was obtained;
  the only profile is zero bytes. No numerical, coverage, cache-reuse, or GPU
  latency pass can be claimed for that run. Full log: `/work/decoder-gpu-v1.log`.
- Production health endpoint `:5093/health` remained `{"status":"ok"}` after
  the isolated GPU crash. No production deployment/configuration change was made.

### Follow-up without the workaround

`/work/run-gpu-decoder-no-workaround-v1.sh` uses the same stack and fixture with
`MIGRAPHX_DISABLE_PASSES` explicitly unset and fresh diagnostic paths. **Passed**:
232 numerical/decision comparisons, and two profiles with 116 MIGraphX-attributed
model/kernel runs each, no CPU-attributed kernels. Maximum absolute differences:
logits 9.15527e-05, state1 7.18236e-06, state2 2.09808e-05. First native Run including
compilation: 5.8466 s; subsequent p50 approximately 0.410–0.418 ms. No crash.
Log: `/work/decoder-gpu-no-workaround-v1.log`. Full results and qualifications:
`/work/diagnostics/decoder-probe-v1-results.md`.

This strongly implicates the disabled pass as the trigger for the failed decoder
compilation path, not a universal decoder failure. Keep encoder/decoder pass
settings separate; production is unchanged. Default passes here still use the
patched runtime, not an unpatched provider. Real-audio end-to-end correctness
remains unverified.

### Independent-process cache replay

`/work/run-gpu-decoder-cache-replay-v1.sh` reused the default-pass cache in a new
process: **116 numerical/decision comparisons passed**, with 116 model runs and
116 exclusively MIGraphX-attributed kernel events. No compilation marker was
logged; pinned provider source logs every compilation call at WARNING. The MXR
hash stayed unchanged. First native Run: 143.265159 ms; warm p50 approximately
0.410–0.419 ms. This verifies disk-cache reuse for the tested fixture/settings,
not arbitrary inputs or production readiness. Log:
`/work/decoder-gpu-cache-replay-v1.log`.

Application encoder and decoder sessions currently share one process and the
same provider options. Do not toggle the process-wide compiler environment
between sessions, or mix compiler configurations in one cache directory.
Boundary findings and the staged real-audio test plan:
`/work/diagnostics/encoder-decoder-isolation-v1.md`.

If investigating the workaround-specific failure, isolate the generated concat/compiler issue
with this fixture and pinned stack. That is separate from building the probe;
no compiler or provider repair has been applied here. For an end-to-end test
before that repair, evaluate an **explicit GPU-encoder/CPU-decoder** configuration
in an isolated instance rather than calling silent fallback GPU success.

No real-audio, end-to-end transcript, arbitrary-shape, concurrent-request,
unpatched-runtime, or production-readiness claim is made by this probe.
