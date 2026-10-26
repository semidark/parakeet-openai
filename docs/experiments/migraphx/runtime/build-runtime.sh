#!/usr/bin/env bash
set -euo pipefail

cd /work/onnxruntime
expected=8a77e459420f58fb946fd9067285cfa719f10bdd
actual=$(git rev-parse HEAD)
test "$actual" = "$expected"
test "$(cat VERSION_NUMBER)" = 1.25.1

mkdir -p /work/artifacts
{
  printf 'ORT_VERSION=1.25.1\nORT_COMMIT=%s\n' "$actual"
  cmake --version | head -1
  gcc --version | head -1
  dpkg-query -W -f='${binary:Package} ${Version}\n' migraphx migraphx-dev rocm-core hip-dev
} | tee /work/artifacts/build-identity.txt

git submodule update --init --depth 1 cmake/external/onnx
python3 -m pip install --break-system-packages 'setuptools>=68.2.2' wheel packaging psutil

nice -n 19 python3 tools/ci_build/build.py \
  --build_dir /work/build \
  --config Release \
  --update --build \
  --build_shared_lib --build_wheel \
  --use_migraphx --migraphx_home /opt/rocm \
  --cmake_generator Ninja \
  --parallel 8 \
  --skip_submodule_sync --skip_pip_install --skip_tests \
  --compile_no_warning_as_error --allow_running_as_root

mkdir -p /work/artifacts/lib /work/artifacts/wheels
cp -a /work/build/Release/libonnxruntime.so* /work/artifacts/lib/
cp -a /work/build/Release/libonnxruntime_providers_shared.so /work/artifacts/lib/
cp -a /work/build/Release/libonnxruntime_providers_migraphx.so /work/artifacts/lib/
cp /work/build/Release/dist/*.whl /work/artifacts/wheels/
find /work/artifacts -type f ! -name SHA256SUMS -exec sha256sum {} + > /work/artifacts/SHA256SUMS
printf '\nORT 1.25.1 MIGraphX build: PASS\n'
