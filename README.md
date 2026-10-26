<div align="center">
  <img src="docs/img/parakeet.png" width="120">
  <h1>Parakeet</h1>
  <p>A high-performance automatic speech recognition (ASR) server written in Go.<br>
  It uses NVIDIA's Parakeet TDT 0.6B model through ONNX Runtime to provide speech-to-text transcription via an OpenAI Whisper-compatible API.</p>
</div>

<br>

## Table of Contents

- [Overview](#overview)
- [Model Architecture](#model-architecture)
- [Parakeet vs Whisper](#parakeet-vs-whisper)
- [Requirements](#requirements)
  - [Installing ONNX Runtime](#installing-onnx-runtime)
- [Installation](#installation)
  - [From Release Binary](#from-release-binary)
  - [From Source](#from-source)
  - [Using Docker](#using-docker)
- [Configuration](#configuration)
  - [Command Line Flags](#command-line-flags)
  - [Environment Variables](#environment-variables)
  - [Model Files](#model-files)
- [API Reference](#api-reference)
  - [Transcribe Audio](#transcribe-audio)
  - [Streaming](#streaming)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Overview

Parakeet ASR Server provides a lightweight, production-ready speech recognition service without Python dependencies. It exposes an API compatible with OpenAI's Whisper, making it a drop-in replacement for applications already using that interface.

Key features:

- OpenAI Whisper-compatible REST API
- API key authentication (optional, via environment variable)
- Streaming transcriptions via Server-Sent Events (OpenAI-compatible `transcript.text.delta` / `transcript.text.done`)
- ONNX Runtime inference (CPU)
- No Python dependency at runtime (ffmpeg is an optional system dependency for non-WAV audio)
- Structured logging with `slog` (text and JSON formats, configurable log level)
- Support for multiple response formats (JSON, text, SRT, VTT)
- Multilingual support (English and 25+ languages)
- Quantized model support for reduced memory footprint
- Automatic audio conversion for non-WAV formats (MP3, OGG, WebM, FLAC, M4A, AAC, Opus, ...) when ffmpeg is installed

## Model Architecture

This server uses the [NVIDIA Parakeet TDT 0.6B](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) model, converted to ONNX format by [istupakov](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx).

The architecture consists of:

- **Encoder**: Conformer-based encoder with 1024-dimensional output. Processes 128-dimensional mel filterbank features with 8x temporal subsampling.
- **Decoder**: Token-and-Duration Transducer (TDT) decoder that jointly predicts tokens and their durations. Uses a 2-layer LSTM with 640-dimensional hidden state.
- **Vocabulary**: 8193 SentencePiece tokens including a blank token for CTC-style decoding.

The int8 quantized models require approximately 670MB of disk space and 2GB of RAM during inference.

## Parakeet vs Whisper

| Aspect        | Parakeet TDT                    | OpenAI Whisper                        |
| ------------- | ------------------------------- | ------------------------------------- |
| Architecture  | Conformer encoder + TDT decoder | Transformer encoder-decoder           |
| Decoding      | Non-autoregressive (parallel)   | Autoregressive (sequential)           |
| Speed         | Faster inference due to TDT     | Slower due to autoregressive decoding |
| Model size    | 0.6B parameters                 | 0.04B - 1.5B parameters               |
| Training data | NeMo ASR datasets               | 680K hours web audio                  |
| Primary focus | Accuracy and speed balance      | Multilingual robustness               |
| Timestamps    | Duration-based prediction       | Attention-based alignment             |

Parakeet TDT uses Token-and-Duration Transducer decoding, which predicts both the token and how many encoder frames to advance in a single step. This allows for faster inference compared to traditional autoregressive decoders while maintaining competitive accuracy.

## Requirements

- **ONNX Runtime 1.25.x or later** (required at runtime)
- Parakeet TDT ONNX models (downloaded separately)
- **ffmpeg** (optional) — enables automatic conversion of MP3, OGG, WebM, FLAC, M4A, AAC, Opus and any other ffmpeg-supported format. When ffmpeg is not present, only WAV input is accepted and non-WAV uploads return a 400 error. The official Docker image already ships with ffmpeg.

For building from source:

- Go 1.25 or later

### Installing ONNX Runtime

ONNX Runtime is required to run the inference. Choose the installation method for your Linux distribution:

#### Ubuntu / Debian

```bash
# Option 1: Download from GitHub releases (recommended)
curl -L -o onnxruntime.tgz "https://github.com/microsoft/onnxruntime/releases/download/v1.25.1/onnxruntime-linux-x64-1.25.1.tgz"
tar xzf onnxruntime.tgz
sudo cp onnxruntime-linux-x64-1.25.1/lib/* /usr/local/lib/
sudo ldconfig

# Option 2: Using apt (if available in your version)
sudo apt update
sudo apt install libonnxruntime-dev
```

#### Fedora / RHEL / CentOS

```bash
# Download from GitHub releases
curl -L -o onnxruntime.tgz "https://github.com/microsoft/onnxruntime/releases/download/v1.25.1/onnxruntime-linux-x64-1.25.1.tgz"
tar xzf onnxruntime.tgz
sudo cp onnxruntime-linux-x64-1.25.1/lib/* /usr/local/lib/
sudo ldconfig
```

#### Arch Linux

```bash
# From AUR
yay -S onnxruntime

# Or manually
curl -L -o onnxruntime.tgz "https://github.com/microsoft/onnxruntime/releases/download/v1.25.1/onnxruntime-linux-x64-1.25.1.tgz"
tar xzf onnxruntime.tgz
sudo cp onnxruntime-linux-x64-1.25.1/lib/* /usr/local/lib/
sudo ldconfig
```

#### Alpine Linux

```bash
apk add onnxruntime
```

#### Manual Installation (any distro)

```bash
# Download and extract
curl -L -o onnxruntime.tgz "https://github.com/microsoft/onnxruntime/releases/download/v1.25.1/onnxruntime-linux-x64-1.25.1.tgz"
tar xzf onnxruntime.tgz

# Install to /usr/local
sudo cp -r onnxruntime-linux-x64-1.25.1/lib/* /usr/local/lib/
sudo cp -r onnxruntime-linux-x64-1.25.1/include/* /usr/local/include/
sudo ldconfig

# Or set environment variable to use from current directory
export ONNXRUNTIME_LIB=$(pwd)/onnxruntime-linux-x64-1.25.1/lib/libonnxruntime.so
```

#### ARM64 (Raspberry Pi, etc.)

```bash
curl -L -o onnxruntime.tgz "https://github.com/microsoft/onnxruntime/releases/download/v1.25.1/onnxruntime-linux-aarch64-1.25.1.tgz"
tar xzf onnxruntime.tgz
sudo cp onnxruntime-linux-aarch64-1.25.1/lib/* /usr/local/lib/
sudo ldconfig
```

#### Verify Installation

```bash
# Check if library is found
ldconfig -p | grep onnxruntime

# Or find the library manually
find /usr -name "libonnxruntime.so*" 2>/dev/null
```

If the library is installed in a non-standard location, set the `ONNXRUNTIME_LIB` environment variable:

```bash
export ONNXRUNTIME_LIB=/path/to/libonnxruntime.so
```

## Installation

### From Release Binary

Download the latest release for your platform from the [Releases](https://github.com/achetronic/parakeet/releases) page.

```bash
# Linux (amd64)
curl -L -o parakeet https://github.com/achetronic/parakeet/releases/latest/download/parakeet-linux-amd64
chmod +x parakeet

# Download models (using Makefile)
make models           # int8 quantized models (recommended, ~670MB)
# Or for full precision:
make models-fp32      # fp32 models (~2.5GB)

# Run (requires ONNX Runtime - see Installing ONNX Runtime section above)
./parakeet -port 5092 -models ./models
```

### From Source

```bash
# Clone the repository
git clone https://github.com/achetronic/parakeet.git
cd parakeet

# Download models
make models           # int8 quantized models (recommended)
# Or: make models-fp32  # full precision models

# Build
make build

# Run
./bin/parakeet
```

### Using Docker

The Docker image that is published in the repository is ready to use. It includes ONNX Runtime and models.

```bash

# Run the container
docker run -d \
  --name parakeet \
  -p 5092:5092 \
  ghcr.io/achetronic/parakeet:latest
```

#### Docker Compose

```yaml
version: "3.8"
services:
  parakeet:
    image: ghcr.io/achetronic/parakeet:latest
    ports:
      - "5092:5092"
    environment:
      - PARAKEET_API_KEY=your-secret-key # optional
      - PARAKEET_WORKERS=2
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5092/health"]
      interval: 30s
      timeout: 3s
      retries: 3
```

### GPU Inference (CUDA)

Parakeet can offload inference to an NVIDIA GPU through the ONNX Runtime CUDA
execution provider. GPU support ships as a dedicated image with the `-cuda`
tag suffix; it bundles the GPU build of ONNX Runtime and the fp32 models on a
CUDA base image.

**Prerequisites:** an NVIDIA GPU with up-to-date drivers and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
installed on the host. The CUDA image is `linux/amd64` only.

```bash
docker run -d \
  --name parakeet \
  --gpus all \
  -p 5092:5092 \
  ghcr.io/achetronic/parakeet:latest-cuda
```

The CUDA image enables the GPU by default (`-gpu cuda`). For a non-Docker
binary, or to select a specific device, use the `-gpu`/`-gpu-device` flags or
the `PARAKEET_GPU`/`PARAKEET_GPU_DEVICE` environment variables:

```bash
./parakeet -gpu cuda -gpu-device 0
# or, equivalently, via environment:
PARAKEET_GPU=cuda PARAKEET_GPU_DEVICE=0 ./parakeet
```

If the requested provider cannot be initialized (missing CUDA libraries, driver
or version mismatch), the server fails at startup rather than silently falling
back to CPU, so misconfiguration is visible immediately.

> [!NOTE]
> fp32 is the GPU-appropriate precision and is the default for the CUDA image.
> CPU images and CPU mode are unaffected by these flags.

**Memory and audio length.** The encoder processes the whole file in a single
pass, so peak memory scales with audio duration. On GPU this is bounded by VRAM:
very long inputs (roughly an hour or more on a 24 GB card) can exceed device
memory and fail with an ONNX Runtime allocation error. Segment long audio (for
example with `ffmpeg -i in.wav -f segment -segment_time 300 -ar 16000 -ac 1
chunk_%03d.wav`) or use a CPU image, which is bounded by system RAM instead.

### GPU Inference (MIGraphX / AMD, experimental)

MIGraphX provider selection uses the existing `onnxruntime_go v1.30.1`
binding and targets **ONNX Runtime 1.25.1**, matching the CPU/CUDA runtime
version. No binding downgrade, API-version fallback, or additional C shim is
required: ORT 1.25.1 supports generic registration with the name `MIGraphX`.

**Prerequisites:** an AMD GPU, compatible ROCm/MIGraphX libraries, and an
**ORT 1.25.1 build with the MIGraphX execution provider enabled**. The standard
CPU and CUDA archives do not provide MIGraphX. The checked AMD ROCm 7.2.4
manylinux index publishes ORT 1.23.2 MIGraphX wheels, not a matching 1.25.1
wheel; those older wheels cannot be used with this binding unchanged.
A local source build of ORT 1.25.1 against ROCm 7.2.4 / MIGraphX 2.15.0
completed, and native loading with the unchanged Go binding passed. This
does not establish end-to-end inference compatibility or provide a published image.

```bash
LD_LIBRARY_PATH=/opt/onnxruntime/lib:/opt/rocm/lib \
ONNXRUNTIME_LIB=/opt/onnxruntime/lib/libonnxruntime.so.1.25.1 \
./parakeet -gpu migraphx -gpu-device 0
```

Disk caching is disabled by default. **Do not enable `-migraphx-cache-dir`
for variable-length inputs with the unpatched ORT 1.25.1 provider:** a local
probe reproduced reuse of a 128-frame compiled program for a 256-frame input,
producing the wrong encoder output shape. The provider's first dynamic run
uses an empty shape key, and its shape-vector hash counts elements rather
than bytes. A no-cache diagnostic model probe passed shape changes, but
disabling the cache is not verified as a complete fix for varying real-audio
lengths with the original models; shape changes also required minutes of
recompilation in that probe.
For a runtime with verified cache fixes, create the directory beforehand and
make it writable by the server user; the provider does not create it. Treat
caches as runtime/GPU/compiler-specific artifacts and do not assume portability
across stack upgrades. `-migraphx-mem-limit` optionally
limits the provider's allocation arena in bytes, **not total GPU memory**.
The arena grows with `kSameAsRequested` rather than power-of-two increments.
The application does not request fp16/int8 quantization; runtime environment
settings may override precision defaults. Existing model selection is unchanged
and prefers `.int8.onnx` files when present. Use a directory containing only
the intended fp32 encoder/decoder models for initial MIGraphX validation;
quantized-model support is not established here. CPU and CUDA configuration
remain unchanged.

Provider-registration failure is fatal rather than silently choosing CPU.
However, successful registration does not prove GPU coverage: unsupported
graph nodes may still execute on CPU. Profiling and output comparisons are
required to establish acceleration and correctness.

MIGraphX calls to the shared encoder are serialized because the 1.25.1
provider updates shape-dependent compiler state before its execution lock.
Requests waiting for the encoder can be canceled, but an already-started
native Run is not interrupted by this guard. CPU/CUDA encoder concurrency
and the decoder worker pool are unchanged.
The shared VAD's native Run calls are also serialized in MIGraphX mode,
including first-run compilation; its existing API has no cancellable wait.

**Validation status (standalone probes, not end-to-end application validation):**

- Native loading of CPU-only and locally built MIGraphX-enabled ORT 1.25.1
  libraries is verified with the unchanged binding. The CPU-only runtime
  rejects MIGraphX as expected. GPU registration passed on AMD Radeon 8060S
  (`gfx1151`) with ROCm 7.2.4 / MIGraphX 2.15.0.
- With the unpatched provider, the original FP32 encoder falls back entirely
  to CPU: the provider rejects 48 `Pad` nodes with omitted optional inputs,
  exceeding its unsupported-node cutoff. Successful registration alone
  therefore does not prove acceleration.
- A separate diagnostic model copy with only trailing empty `Pad` inputs
  removed produced byte-identical CPU references at 128 and 256 frames.
  Its first GPU compilation failed in `simplify_reshapes` at default settings.
- With `MIGRAPHX_DISABLE_PASSES=simplify_reshapes` set only for the diagnostic
  process, the copy passed a 128-frame CPU/GPU comparison: maximum absolute
  error `9.05e-7`, approximately 22 ms warm inference, and successful same-shape
  cache reuse across two sessions. Both profiles recorded only MIGraphX kernel
  events. The first run including compilation took approximately 3 min 19 s.
- With the unpatched provider, variable-length cached inference failed:
  a 128-to-256-frame run
  timed out after five minutes, and a fresh 256-frame session using the
  128-frame cache returned `[1, 1024, 16]` instead of `[1, 1024, 32]`.
- With disk caching disabled and the same process-local compiler workaround,
  the diagnostic copy passed all six comparisons for 256 → 128 → 256 frames
  in one session. Output shapes and lengths matched CPU references; maximum
  absolute error was `5.82e-7` or less. The profile recorded six MIGraphX kernel
  events and no CPU kernel events. Warm runs took approximately 21–22 ms,
  but each shape change, including the return to 256 frames, required roughly
  3–5 minutes of compilation. This does not validate disk caching or the
  original encoder model.
- An isolated provider patch for complete, versioned shape keys and omitted
  optional `Pad` values built successfully. Shape-key regression and native
  loading checks passed. With the same process-local compiler workaround,
  the original encoder passed all 12 comparisons for 128 → 256 → 128 frames
  across two sessions: maximum absolute error `2.44e-7` or less, with 12
  MIGraphX kernel events and no CPU kernel events. Distinct shape-specific
  cache files were generated; cached shape changes took 3.4–5.7 s rather than
  minutes, and warm runs took 21–23 ms. A new-process 256 → 128 → 256 replay
  also passed all six comparisons with only MIGraphX kernel events, 3.2–3.7 s
  shape-load runs, and unchanged cache hashes and metadata. This verifies cache
  reuse for the tested two shapes. It is limited synthetic-input validation,
  not proof of
  arbitrary-length or default-compiler compatibility.
- Decoder default-pass numerical checks and independent-process cache replay
  passed with exclusively MIGraphX profile events. Applying the encoder's
  `simplify_reshapes` workaround to the decoder instead caused compiler errors
  followed by a native crash; the two stages need separate compiler environments.
- A standalone six-second real-audio pipeline with separate encoder/decoder
  processes passed per-element numerical checks, exact transcript and full
  token/duration/frame-trajectory equality, strict MIGraphX attribution, and
  independent-process cache replay. Its warm encoder observation was about
  29 ms; this is not an end-to-end application or service benchmark.
- Persistent two-process validation failed under the unchanged 24 GiB container
  cap: v1's encoder was OOM-killed on a 10-to-2-second shape switch. Fresh
  cache-only v2 workers passed 2 → 6 → 10 → 2 seconds, then reached 22.396 GiB
  sampled container usage on the next 6-second encoder request and were stopped
  by the guard. No compilation markers, changed cache hashes, or new OOM kill
  occurred in v2. Neither run reached GPU timing; no GPU p50/p95 report exists.
  Aggregate memory samples do not establish a leak or an allocation-level cause.
- The experiment is paused. The working ROCm deployment remains the baseline.
  Concurrent native requests, VAD, arbitrary-length behavior, and production
  integration remain unverified. The original models and production service
  were not modified by the probes. See the [handoff snapshot](docs/experiments/migraphx/README.md)
  for sources, runtime provenance, findings, and offline checks.

No compiler-pass workaround, model rewrite, or static-shape override is
enabled automatically by this application. The diagnostic results do not
establish end-to-end deployable compatibility with the original models.
No published MIGraphX container image is claimed by this experimental change.

To run the CPU-only native compatibility test:

```bash
PARAKEET_TEST_ORT125_CPU_LIB=/path/to/cpu-only/libonnxruntime.so.1.25.1 \
  go test ./internal/asr -run TestMIGraphXCPUOnlyRuntime -v
```

## Configuration

### Command Line Flags

| Flag                          | Description                                                              | Default                    | Example                                |
| ----------------------------- | ------------------------------------------------------------------------ | -------------------------- | -------------------------------------- |
| `-port`                       | HTTP server port                                                         | `5092`                     | `-port 8080`                           |
| `-models`                     | Path to models directory                                                 | `./models`                 | `-models /opt/parakeet/models`         |
| `-log-level`                  | Log level: debug, info, warn, error                                      | `info`                     | `-log-level debug`                     |
| `-log-format`                 | Log output format: text or json                                          | `text`                     | `-log-format json`                     |
| `-workers`                    | Concurrent inference workers (each ~670MB RAM for int8)                  | `4`                        | `-workers 2`                           |
| `-ffmpeg`                     | Enable ffmpeg fallback for non-WAV audio                                 | `true`                     | `-ffmpeg=false`                        |
| `-ffmpeg-path`                | Path to the ffmpeg binary (empty = resolve from `PATH`)                  | ``                         | `-ffmpeg-path /usr/bin/ffmpeg`         |
| `-ffmpeg-timeout`             | Maximum wall-clock time for a single ffmpeg conversion                   | `60s`                      | `-ffmpeg-timeout 30s`                  |
| `-gpu`                        | Execution provider: `cpu`, `cuda`, or experimental `migraphx`             | `cpu`                      | `-gpu migraphx`                        |
| `-gpu-device`                 | GPU device index for `cuda`/`migraphx`                                   | `0`                        | `-gpu-device 1`                        |
| `-migraphx-mem-limit`          | MIGraphX arena limit in bytes (not total GPU memory)                      | `0` (runtime default)      | `-migraphx-mem-limit 6442450944`        |
| `-migraphx-cache-dir`          | Writable persistent MIGraphX compiled-model cache directory               | Empty (disabled)           | `-migraphx-cache-dir /cache/migraphx`   |
| `-long-audio`                 | Split audio over the model limit into chunks instead of rejecting it     | `false`                    | `-long-audio`                          |
| `-chunk-seconds`              | Sliding-window size for long audio, in seconds                           | `300`                      | `-chunk-seconds 240`                   |
| `-chunk-overlap-seconds`      | Overlap between consecutive chunks, in seconds                           | `15`                       | `-chunk-overlap-seconds 10`            |
| `-disable-vad-based-chunking` | Disable the Silero VAD chunk-boundary layer (falls back to mel energy)   | `false`                    | `-disable-vad-based-chunking`          |
| `-disable-mel-based-chunking` | Disable the mel-energy chunk-boundary layer (falls back to the midpoint) | `false`                    | `-disable-mel-based-chunking`          |
| `-vad-model-path`             | Path to the Silero VAD ONNX model                                        | `<models>/silero_vad.onnx` | `-vad-model-path /opt/silero_vad.onnx` |

**Examples:**

```bash
# Basic usage
./parakeet

# Custom port and models directory
./parakeet -port 8080 -models /opt/models

# Enable debug logging for troubleshooting
./parakeet -log-level debug

# JSON logs for production (e.g. log aggregation with ELK, Loki, CloudWatch)
./parakeet -log-format json

# JSON logs with debug level
./parakeet -log-format json -log-level debug

# Suppress ONNX Runtime schema warnings (stderr) while keeping debug logs
./parakeet -log-level debug 2>&1 | grep -v "Schema error"
```

### Long Audio

The model's encoder tops out at 400 seconds of audio in a single pass. By
default, longer input is rejected with a clear error (and a log line pointing
here). Pass `-long-audio` to split it into overlapping windows
(`-chunk-seconds`, `-chunk-overlap-seconds`), transcribe each, and stitch the
results, dropping the overlap so words at the seams are not duplicated. Files
under the chunk size are transcribed in one pass either way.

**How chunk boundaries are chosen.** A blind split in the middle of an overlap
can fall mid-word and make that word show up twice or vanish at the seam. To
avoid this, the overlap is split on silence using a cascade (each layer falls
through to the next when it cannot decide):

1. **Silero VAD** picks the centre of the longest silence in the overlap. This
   needs `silero_vad.onnx` in the models directory (downloaded by `make models`;
   see below). A missing model is not fatal: it warns once and falls back to the
   next layer.
2. **Mel energy** picks the quietest point of the already-extracted features.
3. **Midpoint** is the final fallback (the original behaviour).

A second, always-on safety net removes any duplicate or colliding tokens right
at each seam. You can turn off individual layers with
`-disable-vad-based-chunking` / `-disable-mel-based-chunking`.

### Environment Variables

Every command-line flag also reads from an environment variable: take the flag
name, uppercase it and replace dashes with underscores, then prefix it with
`PARAKEET_`. So `-log-level` maps to `PARAKEET_LOG_LEVEL`, `-ffmpeg-timeout` to
`PARAKEET_FFMPEG_TIMEOUT`, and so on. An explicit flag always overrides its env
var (precedence: **CLI flag > env var > default**); an invalid env value is
ignored with a warning and the default is kept.

The Docker images bake their operational defaults as `PARAKEET_*` env vars, so
they survive when you pass your own flags.

A few variables have no flag equivalent:

| Variable           | Description                                 | Default               |
| ------------------ | ------------------------------------------- | --------------------- |
| `ONNXRUNTIME_LIB`  | Path to libonnxruntime.so                   | Auto-detected         |
| `PARAKEET_API_KEY` | API key for `/v1/*` endpoint authentication | Empty (auth disabled) |

### Model Files

The following files are required in the models directory:

| File                            | Size   | Description                                              |
| ------------------------------- | ------ | -------------------------------------------------------- |
| `config.json`                   | 97 B   | Model configuration                                      |
| `vocab.txt`                     | 94 KB  | SentencePiece vocabulary                                 |
| `nemo128.onnx`                  | 140 KB | Preprocessor graph                                       |
| `encoder-model.int8.onnx`       | 652 MB | Quantized encoder                                        |
| `decoder_joint-model.int8.onnx` | 18 MB  | Quantized TDT decoder                                    |
| `silero_vad.onnx`               | 2.3 MB | Silero VAD (chunk boundaries, long-audio only; optional) |

For full precision models, use `encoder-model.onnx` (requires `encoder-model.onnx.data`, 2.5GB total) and `decoder_joint-model.onnx` (72MB).

`silero_vad.onnx` ([snakers4/silero-vad](https://github.com/snakers4/silero-vad), MIT, pinned to release v6.2.1) is downloaded and checksum-verified by `make models`. It is only used to place chunk boundaries on silence in long-audio mode; if it is missing the server logs a warning once and falls back to mel-energy boundaries.

## API Reference

### Authentication

When `PARAKEET_API_KEY` is set, all `/v1/*` endpoints require an `Authorization` header:

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" http://localhost:5092/v1/models
```

The `/health` endpoint is always unauthenticated.

### Transcribe Audio

```
POST /v1/audio/transcriptions
```

Transcribes audio into text. Compatible with OpenAI's Whisper API.

**Request**

Content-Type: `multipart/form-data`

| Parameter         | Type   | Required | Description                                                                            |
| ----------------- | ------ | -------- | -------------------------------------------------------------------------------------- |
| `file`            | file   | Yes      | Audio file (WAV always supported; MP3/OGG/WebM/FLAC/M4A/AAC/Opus via ffmpeg, max 25MB) |
| `model`           | string | No       | Model name (accepted but ignored)                                                      |
| `language`        | string | No       | ISO-639-1 language code (default: en)                                                  |
| `response_format` | string | No       | Output format: json, text, srt, vtt, verbose_json                                      |
| `stream`          | bool   | No       | When `true`, stream the transcription as Server-Sent Events (see Streaming below)      |
| `prompt`          | string | No       | Accepted but ignored                                                                   |
| `temperature`     | float  | No       | Accepted but ignored                                                                   |

**Response**

JSON format (default):

```json
{
  "text": "transcribed text here"
}
```

Verbose JSON format:

```json
{
  "task": "transcribe",
  "language": "en",
  "duration": 5.2,
  "text": "transcribed text here",
  "segments": [
    {
      "id": 0,
      "start": 0,
      "end": 5.2,
      "text": "transcribed text here"
    }
  ]
}
```

**Example**

```bash
curl -X POST http://localhost:5092/v1/audio/transcriptions \
  -H "Authorization: Bearer $PARAKEET_API_KEY" \
  -F file=@audio.wav \
  -F language=en \
  -F response_format=json
```

#### Streaming

Set `stream=true` to receive the transcription incrementally as
[Server-Sent Events](https://developer.mozilla.org/docs/Web/API/Server-sent_events),
following OpenAI's streaming transcription protocol. The audio is still
uploaded in full; the server emits each chunk of text as soon as it is
decoded, then a final event with the complete text.

Two event types are sent:

- `transcript.text.delta` — a piece of newly transcribed text.
- `transcript.text.done` — sent once at the end, with the full transcript.

**Example**

```bash
curl -N -X POST http://localhost:5092/v1/audio/transcriptions \
  -H "Authorization: Bearer $PARAKEET_API_KEY" \
  -F file=@audio.wav \
  -F stream=true
```

**Response** (`Content-Type: text/event-stream`):

```
event: transcript.text.delta
data: {"type":"transcript.text.delta","delta":" Ma"}

event: transcript.text.delta
data: {"type":"transcript.text.delta","delta":"ybe"}

event: transcript.text.done
data: {"type":"transcript.text.done","text":"Maybe next time, huh?"}
```

This is compatible with clients that speak OpenAI's streaming
transcription API, such as Wyoming OpenAI for Home Assistant.

### List Models

```
GET /v1/models
```

Returns available models. Returns `parakeet-tdt-0.6b` and `whisper-1` (alias for compatibility).

### Health Check

```
GET /health
```

Returns `{"status": "ok"}` if the server is running.

## Development

### Available Make Targets

```bash
make help          # Show all available targets

# Build
make build         # Build the binary

# Development
make run           # Build and run
make run-dev       # Run with custom port (5092) for development
make clean         # Remove build artifacts

# Code quality
make fmt           # Format code
make vet           # Run go vet
make lint          # Run all linters (vet + fmt)
make test          # Run tests
make test-coverage # Run tests with coverage report

# Models
make models        # Download int8 models (default)
make models-int8   # Download int8 quantized models
make models-fp32   # Download full precision models

# Docker
make docker-build-int8  # Build Docker image with int8 models
make docker-build-fp32  # Build Docker image with fp32 models
make docker-build-cuda  # Build CUDA/GPU image with fp32 models
make docker-run-int8    # Run Docker container with int8 models
make docker-run-fp32    # Run Docker container with fp32 models
make docker-run-cuda    # Run CUDA/GPU container (needs --gpus all)

# Release
make release       # Build binaries for all platforms
```

### Running Tests

```bash
make test

# With coverage
make test-coverage
open coverage.html
```

## Troubleshooting

### ONNX Runtime library not found

Install ONNX Runtime or set the library path:

```bash
export ONNXRUNTIME_LIB=/path/to/libonnxruntime.so
```

Common installation locations:

- `/usr/lib/libonnxruntime.so`
- `/usr/local/lib/libonnxruntime.so`
- `/opt/onnxruntime/lib/libonnxruntime.so`

### Encoder model not found

Download the models:

```bash
make models
```

### Out of memory errors

Use the int8 quantized models (default) instead of fp32. The int8 models require approximately 2GB of RAM versus 6GB for fp32.

### Unsupported audio format

WAV is always supported natively. Any other format (MP3, OGG, WebM, FLAC, M4A, AAC, Opus, ...) is transcoded on the fly to 16 kHz mono WAV using a local `ffmpeg` binary.

If the server responds with `400 Unsupported or malformed audio`:

1. Install `ffmpeg` and make sure it is in `PATH` (or pass `-ffmpeg-path /absolute/path/to/ffmpeg`). The official Docker image already includes ffmpeg.
2. Check the server logs. On startup you will see one of:
   - `ffmpeg conversion enabled binary=/usr/bin/ffmpeg timeout=60s` — ready.
   - `ffmpeg not found, non-WAV inputs will be rejected` — install it or disable conversion with `-ffmpeg=false` if you only need WAV.
3. As a manual alternative, convert client-side before uploading:

   ```bash
   ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
   ```

Audio is detected by content (magic bytes), not by filename extension, so clients that upload files without an extension still work.

## License

- Code: MIT License
- Parakeet Model: [CC-BY-4.0](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)

## Credits

- **NVIDIA** - Original [Parakeet TDT 0.6B](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) model
- **Ivan Stupakov** ([@istupakov](https://github.com/istupakov)) - [ONNX conversion](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx) of the Parakeet model
