# Staged real-audio CPU/MIGraphX diagnostic

Diagnostic workspace only. No production application import, worker IPC rewrite,
model rewrite, binding downgrade, or provider-default change. Stack: unchanged
`onnxruntime_go v1.30.1`, ORT 1.25.1 with isolated `cachefix-v1` provider, original
FP32 encoder/decoder, gfx1151 validation container. Models are mounted read-only.

## Input and provenance

Historical local speech source (already tracked by the application; not copied
into this snapshot): `testdata/reference/learn-case-interviews.mp3`.
Extracted source interval 1.000–7.000 seconds using:

```sh
ffmpeg -nostdin -v error -n -i SOURCE.mp3 -ss 1.000 -t 6.000 \
  -ac 1 -ar 16000 -c:a pcm_s16le excerpt.wav
```

Six-second, mono, 16 kHz PCM16 WAV; no network needed. Source SHA256:
`182475483bb82f294ed1f5603f23773df29677138c523a34a38c5832750c7eeb`.
WAV SHA256:
`8790a74bcac06efd07296feb1c5d78b78e6c41d68a56b01f9381c50cf6dbabc9`.

`audio.go` and `mel.go` snapshot the application's preprocessing, changing only
the package declaration. `source-provenance.json` records exact source hashes.
WAV parsing uses application PCM scaling; mel extraction uses its 128-feature
filterbank and normalization. Six seconds yields **598 natural mel frames**.
No zero padding, frame bucketing, truncation, or cache-driven length adjustment.

Decoder worker and advance logic snapshot the tested standalone decoder probe.
The new decode loop runs until the actual encoded length is consumed, applying
the application's blank-token state policy, duration argmax, target feedback,
and ten-token-per-frame limit. It does not use the synthetic probe's 16-step
limit. Feature shape/encoded-length equality is required to preserve the
application's channel-major stride. Transcript rendering follows application
word-boundary/special-token/whitespace handling. Long-audio seam logic, VAD,
streaming callbacks and HTTP behavior are outside this short-window test.

## Build and checks

```sh
cd docs/experiments/migraphx/smoke
GOPROXY=off GOSUMDB=off go test ./...
GOPROXY=off GOSUMDB=off go vet ./...
GOPROXY=off GOSUMDB=off go build -o ../ort125-realaudio-probe ./realaudio
```

Do not overwrite a binary used by an active diagnostic. The v1 runner pins the
binary hash, so a rebuilt/different binary requires a new run label and hash.

## Process isolation and comparisons

The orchestrator is `diagnostics/run-realaudio-v1.py`. Each stage launches a
**new OS process** in `ort125-migraphx-validate` with known cache/precision
overrides cleared. GPU encoder alone receives
`MIGRAPHX_DISABLE_PASSES=simplify_reshapes`; decoder processes require the
variable to be **unset**, not merely empty. Encoder and decoder cache directories
are separate, fresh, and explicitly associated with their compiler settings.

1. Prepare one common mel artifact from the WAV.
2. CPU encode and complete CPU decode, with repeat self-checks.
3. GPU encode; compare every feature against CPU at `atol=1e-3`, `rtol=1e-4`.
4. CPU decode of GPU features; require the same full transcript and every
   token/duration/frame decision as the CPU pipeline.
5. GPU decode of CPU features; compare every logit and both state outputs
   against the matching CPU trace at the same tolerances and exact decisions.
6. GPU decode of GPU features; numerical comparison against CPU decoding those
   exact features, plus transcript/trajectory comparison against the CPU pipeline.
7. Independent-process encoder/decoder cache replay; require unchanged MXR
   hashes, no provider compilation markers, and repeated numerical/transcript
   checks. Replay feature artifacts must be byte-identical to their originals
   before reusing the saved numerical trace.

Each stage requires fresh output/report/profile paths and records log/exit status.
CPU self-checks and GPU rollouts carry their own recurrent outputs forward;
comparisons are not hidden behind teacher-forced reference states. Non-finite
values, incomplete trajectories, empty speech transcripts, different argmax
decisions, cache changes, or mixed/missing provider attribution fail the run.

Profiles must contain the reported exact number of `model_run` events and
exclusively the expected provider's kernel events. This is **ORT attribution**,
not a hardware trace. A numerical stage pass alone is insufficient.

The v1 runner is deliberately one-shot. Its paths already become occupied at
start; rerunning is refused. Preserve evidence and use a new labelled runner,
fresh output/cache/profile directories, and updated pinned hashes for a new run.

## Timing boundaries

Per-run `NativeRunNS` measures native ORT Run only. Session creation, the first
native run and subsequent p50/p95 are recorded separately. Encoder first-run
timing can include minutes of new-shape compilation. The decoder first cold
compile occurs in the GPU-on-CPU-features control, before the full GPU path.

`StageWallNS` includes setup, session creation, two repeats, comparison and gob
output writing. Host `process_wall_ns` additionally includes Podman/process
startup, report writing and teardown. Their sums are diagnostic pipeline wall
times with repetitions and file-transfer overhead, **not production latency or
a single-request service benchmark**. Preparation, reference controls and
profile checking are reported separately, not silently counted as GPU inference.

## Evidence and status

All run artifacts live in `/work/realaudio-v1/` (host workspace of the same name):
`orchestrator.log`, per-stage logs/exit files/reports/profiles, mel/features/full
trace gob files, cache manifests, and `result.json` or `failure.json`.

**Completed: PASS**, 2026-10-26 UTC. Offline tests/vet, full real-audio
transcript/trajectory comparisons, per-element numerical checks, strict provider
attribution, and independent-process cache replay passed. Encoder profiles each
contain two exclusively MIGraphX-attributed runs; decoder GPU profiles each
contain 84 exclusively MIGraphX-attributed runs (42 complete decisions, repeated
twice). CPU and staged GPU transcripts match exactly.

Maximum absolute differences: encoder 6.18165e-07; GPU decoder on GPU features:
logits 1.22070e-04, state1 5.25056e-06, state2 2.09808e-05. GPU decoder on CPU
features also passed, with logits maxabs 1.37329e-04. All values were checked
against the declared per-element tolerances; all decisions matched exactly.

Cold encoder first native Run including compilation: 647.297 s. Its second Run
was 29.289 ms. Independent-process cache replay first Run: 3.929 s; second Run:
29.077 ms. The second encoder Run is one warm observation, not a statistically
robust latency distribution. Decoder cold first Run: 5.168 s; cached-process
first Run: 121.497 ms; subsequent decoder native-Run p50: 0.426 ms.

Cached encoder+decoder subprocess wall sum: 6.058 s, including startup, repeats,
profiling and diagnostic I/O. This is not a single-transcription service latency
claim. Both caches retained their hashes, replay features were byte-identical,
and neither replay logged a compilation marker.

Production health returned `{"status":"ok"}` after completion. Application diff
fingerprint matched its pre-run value. No deployment changes. Next investigate
persistent process isolation and startup/cache-load costs before considering
production integration; broader audio/shape/concurrency validation remains open.
