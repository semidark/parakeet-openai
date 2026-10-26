#!/usr/bin/env python3
"""Fresh-process cache-only retry. Preserve v1; keep the 24 GiB cap unchanged."""
import hashlib
import importlib.util
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

ROOT = pathlib.Path(os.environ.get('MIGRAPHX_EXPERIMENT_ROOT', str(pathlib.Path(__file__).resolve().parents[1])))
OUT = ROOT / 'persistent-cacheonly-v2'
SEED = ROOT / 'persistent-v1'
CONTAINER = 'ort125-migraphx-validate'
BINARY_HASH = '52ea4de039cf7037f8c863f4cbf8f8adf2e7e690bf430f00e6a37c267b467dfd'
CAP = 24 * 1024 ** 3
STOP_AT = 22 * 1024 ** 3
CACHE_HASHES = {
    'decoder-cache/20f00-c066bc13b3060973-aafa5570125740ec-shape-v2-579010d7ae22e966.mxr': 'f9f3434fce34363f5d2916192acbb2fa312725fee43b83aa798630df29c0da8a',
    'encoder-cache/20f00-f9539faac1b99a81-aafa5570125740ec-shape-v2-2b606fd7df14fde0.mxr': 'e360aac35e3b149afcf3845ce70d7bfc5981ecee786ad38f0f397ca7b1c7ac14',
    'encoder-cache/20f00-f9539faac1b99a81-aafa5570125740ec-shape-v2-52cb9dd7ec2e4db1.mxr': 'f429f354e6111cad7bacb52b32ff6a7a2e2bc4d21d05611310c9fff73ea41e04',
    'encoder-cache/20f00-f9539faac1b99a81-aafa5570125740ec-shape-v2-6a408d3b5465a523.mxr': '965c609c534b2373ead46cd33a17b22e2c52738b2bddf467bdd3ff2eb21ee0be',
}
FIXTURE_HASHES = {
    'audio-2s.wav': 'ec8e57621847fcdb6d0c48fe3fdd3e4fc4630e7b76566d4272547a28e7e28950',
    'audio-6s.wav': '8790a74bcac06efd07296feb1c5d78b78e6c41d68a56b01f9381c50cf6dbabc9',
    'audio-10s.wav': '8ba3944f4a115cca0c58332ade547b1a0a3dc9b714a1b1300fbd253b29f361dc',
}
spec = importlib.util.spec_from_file_location('persistent_v1_launcher', ROOT / 'diagnostics/run-persistent-v1.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)


def note(name, data):
    previous.staged.exclusive_json(OUT / name, data)


def memory_snapshot():
    command = ['podman', 'exec', CONTAINER, 'sh', '-c',
               'cat /sys/fs/cgroup/memory.current\ncat /sys/fs/cgroup/memory.events\ncat /sys/fs/cgroup/memory.max']
    lines = subprocess.check_output(command, text=True, timeout=10).splitlines()
    return {'current_bytes': int(lines[0]), 'limit_bytes': int(lines[-1]),
            'events': {line.split()[0]: int(line.split()[1]) for line in lines[1:-1]}}


def guard_reason(snapshot, baseline, compilation):
    if snapshot['limit_bytes'] != CAP:
        return 'container memory cap changed'
    if snapshot['events'].get('oom_kill', 0) > baseline['events'].get('oom_kill', 0):
        return 'new container OOM kill detected'
    if snapshot['current_bytes'] >= STOP_AT:
        return '22 GiB memory guard reached; do not raise the 24 GiB cap'
    if compilation:
        return 'unexpected compilation in cache-only retry'
    return None


def stop_owned_coordinator():
    # The pidfile is written by the shell which execs our coordinator. Validate
    # the exact unique root argument in /proc before signalling that PID only.
    # Encoder/decoder children have Pdeathsig=SIGTERM in the unchanged binary.
    script = '''
test -r /work/persistent-cacheonly-v2/coordinator.pid || exit 0
pid=$(cat /work/persistent-cacheonly-v2/coordinator.pid)
case "$pid" in ''|*[!0-9]*) exit 2 ;; esac
test -r "/proc/$pid/cmdline" || exit 0
cmd=$(tr '\\000' ' ' < "/proc/$pid/cmdline")
case "$cmd" in
  *"/work/ort125-persistent-probe -root /work/persistent-cacheonly-v2 -models /models -repeats 30 -mixed 10"*) kill -TERM "$pid" ;;
  *) exit 3 ;;
esac
'''
    subprocess.run(['podman', 'exec', CONTAINER, 'sh', '-c', script], check=True, timeout=10)


def checked_caches():
    return {str(p.relative_to(OUT)): previous.digest(p) for p in sorted(OUT.glob('*-cache/*.mxr'))}


def compilation_seen(root):
    return any('Model Compile: Begin' in p.read_text(errors='replace')
               for role in ['encoder', 'decoder'] for p in root.glob(f'*-{role}.log'))


def main():
    OUT.mkdir(mode=0o700)  # Fresh directory required; never clear/reuse v1.
    before = previous.health()
    baseline = memory_snapshot()
    reason = guard_reason(baseline, baseline, False)
    if reason:
        raise RuntimeError(reason)
    note('launch.json', {'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'before': before,
                         'cache_only': True, 'baseline_memory': baseline, 'stop_at_bytes': STOP_AT,
                         'memory_limit_unchanged_bytes': CAP, 'seed_run': str(SEED)})
    binary = ROOT / 'ort125-persistent-probe'
    if previous.digest(binary) != BINARY_HASH:
        raise RuntimeError('binary differs from pinned v1 build')
    for relative, expected in CACHE_HASHES.items():
        source = SEED / relative
        if previous.digest(source) != expected:
            raise RuntimeError('seed cache changed: ' + relative)
        destination = OUT / relative
        destination.parent.mkdir(exist_ok=True)
        subprocess.run(['cp', '--reflink=auto', '--no-clobber', str(source), str(destination)], check=True)
    for name, expected in FIXTURE_HASHES.items():
        if previous.digest(SEED / name) != expected:
            raise RuntimeError('seed fixture changed: ' + name)
        shutil.copyfile(SEED / name, OUT / name)
    if checked_caches() != CACHE_HASHES:
        raise RuntimeError('cache copies differ from seeds')
    hashes = {p: h for p, h in previous.staged.MODEL_HASHES.items() if p.startswith('/models/')}
    hashes['/work/ort125-persistent-probe'] = BINARY_HASH
    hashes.update({'/work/persistent-cacheonly-v2/' + k: v for k, v in FIXTURE_HASHES.items()})
    provenance = json.loads((SEED / 'provenance.json').read_text())
    provenance.update(hashes=hashes, cache_copies=CACHE_HASHES, seed_run=str(SEED),
                      retry_note='same binary/methodology; all shapes precompiled in v1; fresh processes')
    note('provenance.json', provenance)
    subprocess.run(['podman', 'exec', '-w', '/work/artifacts/cachefix-v1', CONTAINER,
                    'sha256sum', '-c', 'SHA256SUMS'], check=True)
    checks = ''.join(f'{h}  {p}\n' for p, h in hashes.items())
    subprocess.run(['podman', 'exec', '-i', CONTAINER, 'sha256sum', '-c', '-'],
                   input=checks, text=True, check=True)
    env = ['env']
    for name in previous.staged.CLEAR:
        env += ['-u', name]
    env += ['LD_LIBRARY_PATH=/work/artifacts/cachefix-v1/lib:/opt/rocm/lib',
            f'ONNXRUNTIME_LIB={previous.staged.LIB}']
    owned_script = '''
set -eu
test ! -e /work/persistent-cacheonly-v2/coordinator.pid
printf '%s\\n' "$$" > /work/persistent-cacheonly-v2/coordinator.pid
exec /work/ort125-persistent-probe -root /work/persistent-cacheonly-v2 -models /models -repeats 30 -mixed 10
'''
    command = ['podman', 'exec', CONTAINER] + env + [
        'nice', '-n', '19', 'timeout', '--signal=TERM', '--kill-after=15s', '900s', 'sh', '-c', owned_script]
    start = time.monotonic_ns()
    reason = None
    peak = baseline['current_bytes']
    with (OUT / 'coordinator.log').open('x') as log, (OUT / 'memory-samples.jsonl').open('x') as samples:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                snapshot = memory_snapshot()
                peak = max(peak, snapshot['current_bytes'])
                record = {'elapsed_ns': time.monotonic_ns() - start, **snapshot}
                samples.write(json.dumps(record) + '\n')
                samples.flush()
                reason = guard_reason(snapshot, baseline, compilation_seen(OUT))
                if reason:
                    print('GUARD STOP:', reason, flush=True)
                    stop_owned_coordinator()
                    break
                time.sleep(1)
            rc = process.wait(timeout=45)
        except BaseException:
            stop_owned_coordinator()
            process.wait(timeout=45)
            raise
    after = previous.health()
    final_memory = memory_snapshot()
    final_reason = guard_reason(final_memory, baseline, compilation_seen(OUT))
    reason = reason or final_reason
    cache_unchanged = checked_caches() == CACHE_HASHES
    note('launcher-result.json', {'exit': rc, 'wall_ns': time.monotonic_ns() - start,
                                 'after': after, 'application_unchanged': before == after,
                                 'guard_stop': reason, 'peak_sampled_bytes': peak,
                                 'baseline_memory': baseline, 'final_memory': final_memory,
                                 'cache_unchanged': cache_unchanged})
    print('\n'.join((OUT / 'coordinator.log').read_text().splitlines()[-45:]), flush=True)
    if reason or rc or before != after or not cache_unchanged:
        raise RuntimeError(reason or f'exit={rc}, unchanged_health={before == after}, unchanged_cache={cache_unchanged}')
    if json.loads((OUT / 'result.json').read_text())['status'] != 'PASS':
        raise RuntimeError('benchmark did not record PASS')
    print('CACHE-ONLY RETRY: PASS; no new OOM, no compilation, unchanged cap and production', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        if OUT.exists() and not (OUT / 'launcher-failure.json').exists():
            note('launcher-failure.json', {'error': str(exc)})
        print('CACHE-ONLY RETRY: FAIL:', exc, file=sys.stderr, flush=True)
        sys.exit(1)
