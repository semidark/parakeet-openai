# ORT 1.25.1 MIGraphX cache findings

These are local diagnostic results, not a claim of production compatibility.
The Go binding remains `github.com/yalue/onnxruntime_go v1.30.1`.

## Confirmed source defects

In `migraphx_execution_provider.cc`:

1. `make_hash` supplies `v.size()` as the byte length to `MurmurHash3::x86_128`.
   For a vector of four int64 dimensions, this hashes four bytes instead of
   32. Distinct sequence lengths with the same leading dimension can collide.
2. In the `no_input_shape` branch of `compute_func`, shapes are passed to
   the parser but are not appended to `input_shapes`. The first dynamic run
   consequently uses the empty-vector hash `0` for every initial length.
3. Session creation constructs a different key: it reads model dimensions,
   skips the first dimension of each input, and treats symbolic dimensions
   as zero. A complete fix needs consistent input ordering, every dimension,
   rank/input boundaries, and no cache lookup using unresolved dimensions.

## Regression using ORT's actual hash implementation

Build inside the local build container, without network or GPU access:

```sh
g++ -std=c++20 -O2 \
  -I/work/runtime \
  -I/work/onnxruntime/onnxruntime -I/work/onnxruntime/include/onnxruntime \
  /work/diagnostics/cache-hash-regression.cc \
  /work/onnxruntime/onnxruntime/core/framework/murmurhash3.cc \
  -o /work/diagnostics/cache-hash-regression
/work/diagnostics/cache-hash-regression
```

Observed hashes for illustrative input-shape vectors `[1,128,128,1]` and
`[1,128,256,1]` (not a reconstruction of provider parameter ordering):

| Hash implementation | T128 | T256 |
| --- | --- | --- |
| Current element-count length | `519c70b1cf0f20ca` | `519c70b1cf0f20ca` |
| Corrected byte-count length | `cf2462c854e68f44` | `8bd432c8d2a42b41` |

The empty first-run shape-vector hash is `0`. All regression assertions passed.
Correcting only the byte count does not fix the empty first-run key.

## Runtime evidence

- A diagnostic encoder copy with only trailing omitted Pad inputs removed
  matched the original CPU references byte-for-byte at T128 and T256.
- On that copy, ORT 1.25.1 with process-local
  `MIGRAPHX_DISABLE_PASSES=simplify_reshapes` passed all four T128 comparisons
  across two sessions: maximum absolute error `9.05246e-7`, warm runs about
  22 ms. Profiles showed MIGraphX kernel events only.
- A fresh T256 session using the T128 disk cache returned features shaped
  `[1,1024,16]`, but lengths `[32]`. The reference expected `[1,1024,32]`.
  That run exited 1; it was not a correctness pass.
- A cached T128-to-T256 shape-change run timed out after five minutes.
- The no-cache T256-to-T128-to-T256 run completed with exit 0 and all six
  numerical comparisons passing. It used the diagnostic Pad-input model copy
  and process-local `MIGRAPHX_DISABLE_PASSES=simplify_reshapes` override.
  Maximum absolute errors were `3.07336e-7` (first T256), `5.81145e-7`
  (T128), and `2.75671e-7` (return to T256). Shapes and lengths matched CPU
  references. Warm runs took 21-22 ms; shape-change runs took about 5m12s,
  2m57s, and 5m06s respectively, so returning to a prior shape still recompiled.
  Profile inspection found exactly six kernel events, all attributed to
  `MIGraphXExecutionProvider`, with no CPU or unattributed kernel events.
  This establishes GPU correctness only for that diagnostic sequence,
  not the original model, native disk caching, decoder, or real audio.

## Isolation

The production service on `:5093` and the original model were not changed.
Original encoder SHA256:
`98a74b21b4cc0017c1e7030319a4a96f4a9506e50f0708f3a516d02a77c96bb1`.
Do not reuse old `.mxr` files when validating a corrected key scheme.

## Isolated cachefix-v1 build

The modified provider built successfully with two compiler jobs. Artifacts
are in `/work/artifacts/cachefix-v1/lib`; the original `/work/artifacts/lib`
manifest still verifies. The ORT core and shared-provider libraries are
byte-identical to the baseline copies; only the MIGraphX provider differs.

- Patched provider SHA256:
  `f72ffc4bfa0818efc87677d68aab52709e364056c4297126a577e5ac83fd3597`.
- `ldd` resolved all provider dependencies against the existing ROCm stack.
- Native initialization and session-options creation passed with the unchanged
  Go binding and runtime version `1.25.1`.
- In the GPU-less build container (`/dev/kfd` absent), generic MIGraphX
  registration reached HIP device discovery and failed with HIP error 100,
  `no ROCm-capable device is detected`, exit 1. This is not GPU-validation
  success, but no library-loading or API-version error was observed.
- The build emitted one new unused-variable warning. That unused local was
  subsequently removed from the working source; the cachefix-v1 binary and
  its captured source snapshot were not changed or rebuilt.

The versioned shape-key regression passed for T128 versus T256, input order,
rank boundaries, scalar versus rank-one tensors, and renamed inputs. Neither
this regression nor the provider build proves native cache correctness.
## Original-model cachefix-v1 GPU validation

Completed diagnostic: `/work/run-gpu-encoder-cachefix-v1.sh`. It requires
the baseline completion marker, verifies artifact and original-model hashes,
and refuses existing cache/profile directories. It compares T128/T256/T128
against CPU references across two sessions using the original model and
a process-local `simplify_reshapes` override. The run completed with exit 0.

- All 12 numerical comparisons passed, including exact output shapes and
  lengths. Maximum absolute feature error was `2.43075e-7`.
- Both session profiles contained exactly six kernel events each, all
  attributed to `MIGraphXExecutionProvider`; no CPU or unattributed kernel
  events were present. The original model executed on the GPU without rewriting.
- Initial T128 and T256 compile/run times were 3m14.86s and 5m12.88s.
  Returning to T128 in session 1 took 3.83s. Session 2 shape-load runs took
  3.38s (T128), 4.76s (T256), and 5.71s (T128). Warm runs took 21-23 ms.
- Two distinct `shape-v2` cache artifacts were generated, approximately
  2.36 and 2.38 GB each. The shape-change source path attempts `migraphx::load`
  before compilation, using the same versioned key on each run. Cache-load
  messages are VERBOSE and were not captured at this probe's INFO level.
  Independent-process reversed-shape replay also passed; details follow below.
- Baseline and patched artifact manifests still verified, the original model
  SHA256 was unchanged, and the production health endpoint returned `ok`.

These results validate the original encoder for the tested synthetic T128/T256
inputs with the isolated patched provider and process-local compiler workaround.
They do not establish arbitrary-length, decoder, VAD, concurrent native request,
real-audio transcript, or default-compiler compatibility. No production runtime
or application default was changed.

## Independent-process cache replay

`/work/run-gpu-encoder-cachefix-v1-replay.sh` completed with exit 0 using
the original encoder, the same patched provider and process-local workaround,
and the existing cache. It started a new process with T256/T128/T256, one
session, and a two-minute timeout.

- All six comparisons passed; maximum absolute error was `2.43075e-7`,
  with output shapes and lengths matching CPU references.
- The profile contained six kernel events, all attributed to
  `MIGraphXExecutionProvider`, with no CPU or unattributed kernel events.
- Shape-load runs took 3.71s, 3.34s, and 3.17s; warm runs took 21-22 ms.
- Both cache artifacts retained identical filenames, sizes, modification
  timestamps, change timestamps, and SHA256 hashes. No new artifacts appeared.
  This verifies disk-cache reuse across processes for the tested two shapes,
  including a reversed first shape, without reproducing the old collision.

Verified cache artifact SHA256 values (filename suffix -> digest):

- `shape-v2-6795ef77f55be484.mxr` ->
  `e000c4da1247dff2bdedb4e0e18c58186977c07e1c9b58d2bb0c75e9b9112b11`
- `shape-v2-40198bebed36fdef.mxr` ->
  `6ababea9dc66c016731fb39af387d178b98ebf07215363f3f154b1c0d4594189`

Across the initial two sessions and independent replay, all 18 comparisons
passed and all 18 profiled kernel events used MIGraphX. The production health
endpoint still returned `ok`. The validation boundaries above remain unchanged.
