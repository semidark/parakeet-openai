#!/usr/bin/env python3
"""One-shot persistent benchmark launcher, diagnostic container only."""
import hashlib
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(os.environ.get('MIGRAPHX_EXPERIMENT_ROOT', str(pathlib.Path(__file__).resolve().parents[1])))
APP_REPO = pathlib.Path(os.environ.get('MIGRAPHX_APP_REPO', str(pathlib.Path(__file__).resolve().parents[4])))
SOURCE_AUDIO = pathlib.Path(os.environ.get('MIGRAPHX_SOURCE_AUDIO', str(APP_REPO / 'testdata/reference/learn-case-interviews.mp3')))
OUT = ROOT / 'persistent-v1'
spec = importlib.util.spec_from_file_location('staged', ROOT / 'diagnostics/run-realaudio-v1.py')
staged = importlib.util.module_from_spec(spec)
spec.loader.exec_module(staged)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def health():
    diff = subprocess.check_output(['git', '-C', str(APP_REPO), 'diff', '--binary'])
    return {'application_diff_sha256': hashlib.sha256(diff).hexdigest(),
            'health': json.loads(subprocess.check_output(['curl', '--max-time', '5', '--fail', '--silent', '--show-error', 'http://127.0.0.1:5093/health'], text=True))}


def main():
    if (OUT / 'launch.json').exists():
        raise RuntimeError('one-shot launcher already started; use fresh labelled paths')
    before = health()
    staged.exclusive_json(OUT / 'launch.json', {'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'before': before})
    subprocess.run(['podman', 'exec', '-w', '/work/artifacts/cachefix-v1', staged.CONTAINER, 'sha256sum', '-c', 'SHA256SUMS'], check=True)
    hashes = {p: h for p, h in staged.MODEL_HASHES.items() if p.startswith('/models/')}
    hashes['/work/ort125-persistent-probe'] = digest(ROOT / 'ort125-persistent-probe')
    if hashes['/work/ort125-persistent-probe'] != '52ea4de039cf7037f8c863f4cbf8f8adf2e7e690bf430f00e6a37c267b467dfd':
        raise RuntimeError('binary differs from reviewed build')
    for name in ['audio-2s.wav', 'audio-6s.wav', 'audio-10s.wav']:
        hashes['/work/persistent-v1/' + name] = digest(OUT / name)
    staged.exclusive_json(OUT / 'provenance.json', {
        'hashes': hashes, 'source_audio': str(SOURCE_AUDIO),
        'source_sha256': digest(SOURCE_AUDIO),
        'source_start_seconds': 1, 'duration_seconds': [2, 6, 10],
        'fixture_command': 'ffmpeg -nostdin -v error -n -i SOURCE -ss 1.000 -t SECONDS -ac 1 -ar 16000 -c:a pcm_s16le WAV',
        'snapshot_provenance': json.loads((ROOT / 'smoke/persistent/source-provenance.json').read_text()),
        'source_hashes': {str(p.relative_to(ROOT)): digest(p) for p in sorted((ROOT / 'smoke/persistent').glob('*.go'))},
        'cache_copies': {str(p.relative_to(OUT)): digest(p) for p in sorted(OUT.glob('*-cache/*.mxr'))},
    })
    checks = ''.join(f'{h}  {p}\n' for p, h in hashes.items())
    subprocess.run(['podman', 'exec', '-i', staged.CONTAINER, 'sha256sum', '-c', '-'], input=checks, text=True, check=True)
    env = ['env']
    for name in staged.CLEAR:
        env += ['-u', name]
    env += ['LD_LIBRARY_PATH=/work/artifacts/cachefix-v1/lib:/opt/rocm/lib', f'ONNXRUNTIME_LIB={staged.LIB}']
    command = ['podman', 'exec', staged.CONTAINER] + env + [
        'nice', '-n', '19', 'timeout', '--signal=TERM', '--kill-after=15s', '3600s',
        '/work/ort125-persistent-probe', '-root', '/work/persistent-v1', '-models', '/models', '-repeats', '30', '-mixed', '10']
    start = time.monotonic_ns()
    with (OUT / 'coordinator.log').open('x') as log:
        rc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
    after = health()
    staged.exclusive_json(OUT / 'launcher-result.json', {'exit': rc, 'wall_ns': time.monotonic_ns() - start, 'after': after, 'application_unchanged': before == after})
    print('\n'.join((OUT / 'coordinator.log').read_text().splitlines()[-65:]), flush=True)
    if rc:
        raise RuntimeError(f'benchmark exit {rc}; retained partial evidence, not a pass')
    result = json.loads((OUT / 'result.json').read_text())
    if result['status'] != 'PASS' or before != after:
        raise RuntimeError('benchmark/status or application-health gate failed')
    print('PERSISTENT BENCHMARK AND PRODUCTION-UNCHANGED CHECK: PASS', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        if not (OUT / 'launcher-failure.json').exists():
            staged.exclusive_json(OUT / 'launcher-failure.json', {'error': str(exc)})
        print('PERSISTENT BENCHMARK LAUNCHER: FAIL:', exc, file=sys.stderr, flush=True)
        sys.exit(1)
