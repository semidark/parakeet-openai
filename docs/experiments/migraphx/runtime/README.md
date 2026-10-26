# Runtime source and build provenance

Base: ONNX Runtime **1.25.1**, commit
`8a77e459420f58fb946fd9067285cfa719f10bdd`. See `base-build-identity.txt`
for compiler and ROCm package versions, and `validated-library-SHA256SUMS` for
the original binary identities. Libraries are not included.

`validated-provider.patch` plus `migraphx_cache_key.h` are the exact captured
sources used for the tested cachefix-v1 provider:

- Patch SHA256: `93491bf3b33a079416e3e7bb80447be6079bc6f3edf3605928206bb7741c553e`
- Header SHA256: `e29149fd655e2e8e03a59ea58cec046f5b0b13f510fd9614ce12ac02e34d1ff1`
- Built provider SHA256: `f72ffc4bfa0818efc87677d68aab52709e364056c4297126a577e5ac83fd3597`

`working-provider.patch` is the later nested checkout's current tracked diff.
It only additionally removes an unused `index` local. It was **not rebuilt**
into the pinned validated binary and must not be presented as that binary's
exact source. Both patches require the same separate new header.

The root `.gitattributes` exempts only these patch files from blank-at-end-of-line
whitespace checks: unified diffs require a space prefix on empty context lines.
Removing that prefix would change the exact captured patch and its SHA256.

The patch corrects optional omitted `Pad` values and consistent complete input
shape identity (names, ranks, all dimensions and version prefix), including
first dynamic execution and session initialization. Old caches are not compatible.
Compiler-pass settings are not part of the key; use separate cache directories
for distinct compiler environments and pinned stacks.

## Applying the validated source (do not run as part of archiving)

In a **fresh disposable** checkout at the base commit, set `SNAPSHOT` to this
directory's absolute path, then:

```sh
git apply --check "$SNAPSHOT/validated-provider.patch"
git apply "$SNAPSHOT/validated-provider.patch"
cp "$SNAPSHOT/migraphx_cache_key.h" onnxruntime/core/providers/migraphx/
```

`build-runtime.sh` and `build-cachefix-v1.sh` preserve the original build
procedures and `/work` container layout. Build the pristine baseline first,
then apply the validated source and build the isolated provider; the latter
retains the baseline core/shared-provider libraries. They are provenance,
not a complete container/bootstrap recipe. They require dependencies, compiler
environment and mounts prepared separately, and may install/download build
dependencies. No build or GPU execution is automatic.

## Offline C++ cache regression

With `ORT_SRC` pointing to the base ORT checkout, `SNAPSHOT` to this directory,
and a C++20 compiler available:

```sh
g++ -std=c++20 -O2 \
  -I"$SNAPSHOT" \
  -I"$ORT_SRC/onnxruntime" -I"$ORT_SRC/include/onnxruntime" \
  "$SNAPSHOT/../diagnostics/cache-hash-regression.cc" \
  "$ORT_SRC/onnxruntime/core/framework/murmurhash3.cc" \
  -o /tmp/opencode/migraphx-cache-hash-regression
/tmp/opencode/migraphx-cache-hash-regression
```

This checks hash/key regressions without loading ORT or accessing a GPU; it does
not validate native model coverage or performance. The ORT source/headers are
not vendored; their Microsoft/MIT notices remain intact in the preserved patch
and header.
