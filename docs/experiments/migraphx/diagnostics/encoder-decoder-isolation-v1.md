# Encoder/decoder compiler isolation — inspection, 2026-10-26 UTC

## Verified application boundary

Inspected application `internal/asr/transcriber.go`:
`NewTranscriber` initializes one in-process ORT environment, creates one shared
dynamic encoder session, then creates decoder worker sessions in the same
process. All receive the same provider SessionOptions from `buildSessionOptions`.
Optional VAD also receives these options. The encoder Run gate is a concurrency
guard, not process/compiler-environment isolation. No encoder/decoder IPC or
per-model compiler-pass option exists in the inspected path.

No application code sets `MIGRAPHX_DISABLE_PASSES`; prior overrides came from
isolated diagnostic launchers. A launcher-level override would affect both
models in the current application process. Installed MIGraphX `env.hpp` also
contains cached environment accessors. Do not assume setting/unsetting the
variable around session creation or Run reliably scopes compiler behavior;
the specific pass-variable read implementation has not been inspected here.
Deferred compilation and concurrent requests further make environment toggling
an unsafe proposed integration strategy.

## Cache configuration boundary

Pinned patched provider filenames use MIGraphX version, graph identity, GPU
architecture and `shape-v2` input identity. The inspected shape key encodes input
names/ranks/dimensions. There is no compiler-pass override in the inspected
provider cache-name construction. Use separate cache directories for different
compiler environments and pinned stacks. Do not use a default-pass decoder cache
to test whether compiling with the encoder workaround succeeds: a cache hit
could bypass the failing compilation path.

## Completed diagnostics

- Encoder: earlier independent-process cache replay passed for mel lengths
  128 and 256 with `simplify_reshapes` disabled, on the patched runtime.
- Decoder: default-pass cold run and independent-process cache replay now pass
  numerical checks and strict ORT MIGraphX attribution. Disabling
  `simplify_reshapes` instead caused concat compilation errors then SIGSEGV.
- These inputs were synthetic; this does not establish real-audio transcription.
- Existing uncommitted application changes were inspected but left untouched.
  Production remains unchanged and its health check succeeded.

## Smallest next isolated real-audio diagnostic

Use a staged local diagnostic, not a production worker-process rewrite:

1. Extract a short speech excerpt from the existing checked-in
   `testdata/reference/learn-case-interviews.mp3`; preserve source and record
   excerpt offsets, conversion settings and hashes. No network is required.
2. Compute mel features using the application's existing preprocessing. Record
   the actual frame count; do not silently truncate or pad to 128/256 just to
   hit the validated caches. An unseen shape may require lengthy compilation.
3. Launch an encoder-only diagnostic process with the encoder workaround and
   its own cache directory; save named output tensors and encoded length.
   Compare against CPU encoding of identical mel input and inspect profiles.
4. Exit that process, then launch a decoder-only process with the override
   explicitly unset and a decoder-specific cache. Decode saved CPU/GPU features
   using the application's full TDT stopping and state-update policy.
5. Compare the resulting transcripts and token/duration sequences against a
   full CPU pipeline for identical audio. Do not substitute the current 29-step
   synthetic decoder fixture for complete transcription. Check GPU attribution
   for both stages and report stage/total times with subprocess/file overhead
   separated from native inference. Repeat for cached/warm behavior.

Only after this test passes should persistent worker IPC or a scoped compiler
fix be considered for application integration. An explicit GPU-encoder/CPU-
decoder baseline is also possible, but it is not full-GPU execution. No such
production architecture change was made.

## Implementation follow-up

The staged diagnostic is now implemented in `/work/smoke/realaudio/`, with the
one-shot orchestrator `/work/diagnostics/run-realaudio-v1.py`. It uses a local
six-second speech excerpt, exact preprocessing snapshots, full TDT completion,
separate OS processes/compiler caches, strict profile checks, and cold/replay
comparisons. Offline tests and vet passed. The complete staged real-audio run
and independent-process encoder/decoder cache replay **passed**. CPU and GPU
transcripts and all token/duration/frame decisions matched; strict profiles
showed exclusively MIGraphX attribution for both GPU stages. See
`/work/realaudio-v1/result.json` and `/work/smoke/realaudio/README.md` for timings,
numerical errors and qualifications. This is not application integration or
production-readiness validation. Production health and application diff
fingerprint were unchanged after completion.
