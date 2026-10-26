#!/usr/bin/env python3
"""Isolated staged transcription; launches a new process per compiler environment."""
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(os.environ.get('MIGRAPHX_EXPERIMENT_ROOT', str(pathlib.Path(__file__).resolve().parents[1])))
OUT = ROOT / 'realaudio-v1'
CONTAINER = 'ort125-migraphx-validate'
LIB = '/work/artifacts/cachefix-v1/lib/libonnxruntime.so.1.25.1'
MODEL_HASHES = {
    '/models/encoder-model.onnx': '98a74b21b4cc0017c1e7030319a4a96f4a9506e50f0708f3a516d02a77c96bb1',
    '/models/decoder_joint-model.onnx': 'e978ddf6688527182c10fde2eb4b83068421648985ef23f7a86be732be8706c1',
    '/models/vocab.txt': 'd58544679ea4bc6ac563d1f545eb7d474bd6cfa467f0a6e2c1dc1c7d37e3c35d',
    '/work/ort125-realaudio-probe': '61bc3d1fde1a0e8ee9de3dd838a488210cffa00b034fcf931b747dbe3ebe8c8b',
    '/work/realaudio-v1/excerpt.wav': '8790a74bcac06efd07296feb1c5d78b78e6c41d68a56b01f9381c50cf6dbabc9',
}
CLEAR = [
    'MIGRAPHX_DISABLE_PASSES', 'ORT_MIGRAPHX_MODEL_CACHE_PATH',
    'ORT_MIGRAPHX_CACHE_PATH', 'ORT_MIGRAPHX_DUMP_MODEL_OPS',
    'ORT_MIGRAPHX_FP16_ENABLE', 'ORT_MIGRAPHX_BF16_ENABLE',
    'ORT_MIGRAPHX_FP8_ENABLE', 'ORT_MIGRAPHX_INT8_ENABLE',
    'ORT_MIGRAPHX_EXHAUSTIVE_TUNE',
]
stage_results = []


def exclusive_json(path, data):
    with path.open('x') as f:
        json.dump(data, f, indent=2)
        f.write('\n')


def check_profiles(label, provider, runs):
    cmd = [sys.executable, str(ROOT / 'diagnostics/check-decoder-profiles.py'),
           str(OUT / f'{label}-profiles'), '--provider', provider,
           '--sessions', '1', '--runs-per-session', str(runs)]
    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    with (OUT / f'{label}-coverage.log').open('x') as f:
        f.write(result.stdout)
    print(result.stdout, end='', flush=True)
    if result.returncode:
        raise RuntimeError(f'{label}: strict provider attribution failed')


def stage(label, kind, provider, source, output, reference=None, cache=None):
    print(f'START {label} stage={kind} provider={provider}', flush=True)
    if (OUT / f'{label}.log').exists() or (OUT / f'{label}.exit').exists():
        raise RuntimeError(f'{label}: existing evidence; refusing overwrite')
    env = ['env']
    for name in CLEAR:
        env += ['-u', name]
    env += [f'LD_LIBRARY_PATH=/work/artifacts/cachefix-v1/lib:/opt/rocm/lib',
            f'ONNXRUNTIME_LIB={LIB}']
    if kind == 'encode' and provider == 'migraphx':
        env += ['MIGRAPHX_DISABLE_PASSES=simplify_reshapes']
    args = ['/work/ort125-realaudio-probe', '-stage', kind, '-provider', provider,
            '-input', '/work/realaudio-v1/' + source,
            '-output', '/work/realaudio-v1/' + output,
            '-report', f'/work/realaudio-v1/{label}.json', '-repeats', '2']
    if kind != 'prepare':
        model = 'encoder-model.onnx' if kind == 'encode' else 'decoder_joint-model.onnx'
        args += ['-model', '/models/' + model,
                 '-profiles', f'/work/realaudio-v1/{label}-profiles']
    if kind == 'decode':
        args += ['-vocab', '/models/vocab.txt']
    if reference:
        args += ['-reference', '/work/realaudio-v1/' + reference]
    if cache:
        args += ['-cache', '/work/realaudio-v1/' + cache]
    cmd = ['podman', 'exec', CONTAINER] + env + [
        'nice', '-n', '19', 'timeout', '--signal=TERM', '--kill-after=10s',
        '1200s' if kind == 'encode' else '180s'] + args
    start = time.monotonic_ns()
    with (OUT / f'{label}.log').open('x') as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
    wall = time.monotonic_ns() - start
    with (OUT / f'{label}.exit').open('x') as f:
        f.write(str(rc) + '\n')
    if rc:
        print('\n'.join((OUT / f'{label}.log').read_text().splitlines()[-60:]), flush=True)
        raise RuntimeError(f'{label}: exit {rc}; stopped without claiming success')
    report = json.loads((OUT / f'{label}.json').read_text())
    stage_results.append({'label': label, 'process_wall_ns': wall, 'report': report})
    if kind != 'prepare':
        check_profiles(label, provider, report['Runs'])
    print(f'PASS {label} process_wall_s={wall / 1e9:.6f} runs={report["Runs"]} text={report["Text"]!r}', flush=True)
    return report


def cache_manifest(dirname):
    directory = OUT / dirname
    files = sorted(directory.glob('*.mxr'))
    if not files:
        raise RuntimeError(f'{dirname}: missing compiled cache')
    return {p.name: {'size': p.stat().st_size,
                     'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in files}


def same_transcription(got, ref, label):
    for key in ['InputHash', 'ModelHash', 'VocabHash', 'Lengths', 'Text', 'Decisions']:
        if got[key] != ref[key]:
            raise RuntimeError(f'{label}: full-pipeline mismatch in {key}')
    if not got['Text'] or not got['Decisions']:
        raise RuntimeError(f'{label}: empty transcript is not a speech validation pass')


def main():
    if (OUT / 'started.json').exists():
        raise RuntimeError('v1 already started; use fresh paths for another run')
    exclusive_json(OUT / 'started.json', {'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                                          'container': CONTAINER, 'diagnostic_only': True})
    subprocess.run(['podman', 'exec', '-w', '/work/artifacts/cachefix-v1', CONTAINER,
                    'sha256sum', '-c', 'SHA256SUMS'], check=True)
    checks = ''.join(f'{h}  {p}\n' for p, h in MODEL_HASHES.items())
    subprocess.run(['podman', 'exec', '-i', CONTAINER, 'sha256sum', '-c', '-'],
                   input=checks, text=True, check=True)
    stage('prepare', 'prepare', 'cpu', 'excerpt.wav', 'mel.gob')
    cpu_encoder = stage('cpu-encoder', 'encode', 'cpu', 'mel.gob', 'cpu-features.gob')
    cpu = stage('cpu-decode', 'decode', 'cpu', 'cpu-features.gob', 'cpu-trace.gob')
    gpu_encoder = stage('gpu-encoder-cold', 'encode', 'migraphx', 'mel.gob',
                        'gpu-features.gob', 'cpu-features.gob', 'encoder-simplify-disabled-cache')
    encoder_cache = cache_manifest('encoder-simplify-disabled-cache')
    # CPU control for exact GPU encoder output; separates encoder/decoder effects.
    cpu_gpu_features = stage('cpu-decode-gpu-features', 'decode', 'cpu',
                             'gpu-features.gob', 'cpu-gpu-features-trace.gob')
    same_transcription(cpu_gpu_features, cpu, 'CPU decoder on GPU features')
    gpu_cpu_features = stage('gpu-decode-cpu-features', 'decode', 'migraphx',
                             'cpu-features.gob', 'gpu-cpu-features-trace.gob',
                             'cpu-trace.gob', 'decoder-default-cache')
    same_transcription(gpu_cpu_features, cpu, 'GPU decoder on CPU features')
    gpu = stage('gpu-decode-gpu-features', 'decode', 'migraphx',
                'gpu-features.gob', 'gpu-trace.gob',
                'cpu-gpu-features-trace.gob', 'decoder-default-cache')
    same_transcription(gpu, cpu, 'staged GPU pipeline')
    decoder_cache = cache_manifest('decoder-default-cache')
    exclusive_json(OUT / 'cache-before-replay.json', {'encoder': encoder_cache, 'decoder': decoder_cache})
    warm_encoder = stage('gpu-encoder-replay', 'encode', 'migraphx', 'mel.gob',
                         'gpu-replay-features.gob', 'cpu-features.gob', 'encoder-simplify-disabled-cache')
    if cache_manifest('encoder-simplify-disabled-cache') != encoder_cache:
        raise RuntimeError('encoder cache changed during independent-process replay')
    # A byte-identical feature artifact is required for the saved numerical trace.
    if hashlib.sha256((OUT / 'gpu-replay-features.gob').read_bytes()).hexdigest() != hashlib.sha256((OUT / 'gpu-features.gob').read_bytes()).hexdigest():
        raise RuntimeError('replay feature artifact changed; generate an explicit matching CPU trace before proceeding')
    warm_gpu = stage('gpu-decode-replay', 'decode', 'migraphx',
                     'gpu-replay-features.gob', 'gpu-replay-trace.gob',
                     'cpu-gpu-features-trace.gob', 'decoder-default-cache')
    same_transcription(warm_gpu, cpu, 'independent-process GPU pipeline replay')
    if cache_manifest('decoder-default-cache') != decoder_cache:
        raise RuntimeError('decoder cache changed during independent-process replay')
    for label in ['gpu-encoder-replay', 'gpu-decode-replay']:
        if 'Model Compile: Begin' in (OUT / f'{label}.log').read_text():
            raise RuntimeError(f'{label}: replay unexpectedly compiled')
    cpu_time = sum(s['process_wall_ns'] for s in stage_results if s['label'] in ['cpu-encoder', 'cpu-decode'])
    gpu_time = sum(s['process_wall_ns'] for s in stage_results if s['label'] in ['gpu-encoder-cold', 'gpu-decode-gpu-features'])
    # Decoder compilation occurs first on CPU features; include that separately.
    warm_time = sum(s['process_wall_ns'] for s in stage_results if s['label'] in ['gpu-encoder-replay', 'gpu-decode-replay'])
    exclusive_json(OUT / 'result.json', {
        'status': 'PASS', 'scope': 'six-second staged diagnostic; not production/HTTP/concurrency/long-audio validation',
        'transcript': cpu['Text'], 'natural_mel_frames': cpu_encoder['Frames'],
        'encoder_maxabs': gpu_encoder['MaxAbs'], 'gpu_decoder_maxabs': gpu['MaxAbs'],
        'cpu_encode_decode_process_sum_ns': cpu_time,
        'gpu_cold_encoder_plus_cached_decoder_process_sum_ns': gpu_time,
        'gpu_cached_encode_decode_process_sum_ns': warm_time,
        'timing_note': 'Sequential subprocess wall sums include startup, repeats, profiling and diagnostic I/O; not service latency. See per-run native times.',
        'cache_replay': 'PASS: independent processes, unchanged hashes, no compile markers',
        'stages': stage_results,
    })
    print(f'REAL-AUDIO STAGED PIPELINE: PASS transcript={cpu["Text"]!r}', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'REAL-AUDIO STAGED PIPELINE: FAIL: {exc}', file=sys.stderr, flush=True)
        if not (OUT / 'failure.json').exists():
            exclusive_json(OUT / 'failure.json', {'error': str(exc), 'completed_stages': stage_results})
        sys.exit(1)
