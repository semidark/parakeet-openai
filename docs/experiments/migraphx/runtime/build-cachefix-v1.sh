#!/bin/sh
set -eu
dest=/work/artifacts/cachefix-v1
test ! -e "$dest"
mkdir -p "$dest/lib"
cp -a /work/artifacts/lib/libonnxruntime.so* "$dest/lib/"
cp -a /work/artifacts/lib/libonnxruntime_providers_shared.so "$dest/lib/"
nice -n 19 cmake --build /work/build/Release --target onnxruntime_providers_migraphx --parallel 2
cp -a /work/build/Release/libonnxruntime_providers_migraphx.so "$dest/lib/"
git -C /work/onnxruntime diff --check
git -C /work/onnxruntime diff --binary > "$dest/provider-tracked-changes.patch"
cp /work/onnxruntime/onnxruntime/core/providers/migraphx/migraphx_cache_key.h "$dest/"
cp /work/artifacts/build-identity.txt "$dest/base-build-identity.txt"
find "$dest/lib" -type f -exec sha256sum {} + > "$dest/SHA256SUMS"
printf 'CACHEFIX PROVIDER BUILD: PASS\n'
