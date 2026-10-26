#!/usr/bin/env python3
"""Independently verify completed persistent benchmark evidence and write a qualified report.

Does not launch inference, rebuild binaries, or overwrite benchmark artifacts.
"""
import argparse
import collections
import hashlib
import json
import math
import pathlib


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def percentile(values, p):
    require(bool(values), 'empty percentile input')
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * p) - 1)]


def verified_summaries(result, started):
    require(result['status'] == 'PASS', 'benchmark did not pass')
    require(started['timed_profiling'] is False, 'timed profiling enabled')
    require(result['percentile_method'] == 'nearest rank', 'unexpected percentile method')
    expected_counts = {'validation': 2, 'first-shape': 1, 'warmup': 3,
                       'steady': started['timed_repeats'], 'mixed': started['mixed_repeats']}
    require(expected_counts['steady'] >= 20 and expected_counts['mixed'] >= 10,
            'insufficient requested samples')
    groups = collections.defaultdict(list)
    references = {}
    for row in result['measurements']:
        provider, phase, fixture = row['Provider'], row['Phase'], row['Fixture']
        require(provider in ['cpu', 'migraphx'], 'unknown provider')
        require(phase in expected_counts, 'unknown phase')
        require(fixture in ['audio-2s.wav', 'audio-6s.wav', 'audio-10s.wav'], 'unknown fixture')
        seconds = int(fixture.split('-')[1].split('s')[0])
        require(row['Samples'] == seconds * 16000 and row['Frames'] == seconds * 100 - 2,
                f'{provider}/{phase}/{fixture}: unnatural input length')
        require(row['EncodedFrames'] == (row['Frames'] - 1) // 8 + 1,
                f'{fixture}: unexpected encoded length')
        require(row['Text'] and len(row['Decisions']) == row['DecodeSteps'] > 0,
                f'{fixture}: missing transcript/decisions')
        clocks = ['TotalNS', 'PreprocessNS', 'EncoderRPCNS', 'DecoderRPCNS',
                  'EncoderNativeNS', 'DecoderNativeNS', 'DecoderFirstNativeNS']
        require(all(isinstance(row[k], int) and row[k] > 0 for k in clocks), 'invalid timing')
        require(row['TotalNS'] >= sum(row[k] for k in ['PreprocessNS', 'EncoderRPCNS', 'DecoderRPCNS']),
                'total shorter than sequential timed stages')
        require(row['EncoderRPCNS'] >= row['EncoderNativeNS'] and
                row['DecoderRPCNS'] >= row['DecoderNativeNS'] >= row['DecoderFirstNativeNS'],
                'native inference exceeds RPC timing')
        trajectory = (row['Text'], row['EncodedFrames'], row['DecodeSteps'], row['Decisions'])
        if provider == 'cpu' and phase == 'validation' and row['Sequence'] == 0:
            require(fixture not in references, 'duplicate CPU baseline')
            references[fixture] = trajectory
        groups[(provider, phase, fixture)].append(row)
    require(len(references) == 3, 'missing CPU references')
    for provider in ['cpu', 'migraphx']:
        for fixture in references:
            for phase, count in expected_counts.items():
                rows = groups[(provider, phase, fixture)]
                require(len(rows) == count, f'{provider}/{phase}/{fixture}: expected {count} samples, got {len(rows)}')
                require(sorted(r['Sequence'] for r in rows) == list(range(count)), 'missing/duplicate sequence')
                for r in rows:
                    require((r['Text'], r['EncodedFrames'], r['DecodeSteps'], r['Decisions']) == references[fixture],
                            f'{provider}/{phase}/{fixture}: trajectory mismatch')
    supplied = {}
    for s in result['summary']:
        key = (s['Provider'], s['Schedule'], s['Fixture'])
        require(key not in supplied, 'duplicate supplied summary')
        supplied[key] = s
    calculated = []
    for provider in ['cpu', 'migraphx']:
        for schedule in ['steady', 'mixed']:
            for seconds in [2, 6, 10]:
                fixture = f'audio-{seconds}s.wav'
                key = (provider, schedule, fixture)
                rows = groups[key]
                s = {'Provider': provider, 'Schedule': schedule, 'Fixture': fixture, 'Count': len(rows)}
                total = [r['TotalNS'] for r in rows]
                s.update(TotalP50MS=percentile(total, .5) / 1e6,
                         TotalP95MS=percentile(total, .95) / 1e6,
                         TotalMinMS=min(total) / 1e6, TotalMaxMS=max(total) / 1e6)
                for field, source in [('EncoderNativeP50MS', 'EncoderNativeNS'),
                                      ('DecoderNativeP50MS', 'DecoderNativeNS'),
                                      ('PreprocessP50MS', 'PreprocessNS')]:
                    s[field] = percentile([r[source] for r in rows], .5) / 1e6
                s['IPCP50MS'] = percentile([r['TotalNS'] - r['PreprocessNS'] - r['EncoderNativeNS'] -
                                           r['DecoderNativeNS'] for r in rows], .5) / 1e6
                s['RealTimeFactorP50'] = s['TotalP50MS'] / (seconds * 1000)
                s['RealTimeFactorP95'] = s['TotalP95MS'] / (seconds * 1000)
                require(key in supplied, 'missing supplied summary')
                for field, value in s.items():
                    got = supplied[key][field]
                    require(math.isclose(got, value, rel_tol=1e-10, abs_tol=1e-10) if isinstance(value, float)
                            else got == value, f'{key}: inconsistent summary {field}')
                calculated.append(s)
    require(len(supplied) == 12, 'unexpected summary count')
    return calculated


def verify_files(root, result):
    for name in ['failure.json', 'launcher-failure.json']:
        require(not (root / name).exists(), f'failure evidence exists: {name}')
    launch = json.loads((root / 'launch.json').read_text())
    completed = json.loads((root / 'launcher-result.json').read_text())
    require(completed['exit'] == 0 and completed['application_unchanged'] is True and
            completed['after'] == launch['before'], 'launcher/application-health gate failed')
    require(completed['after']['health'] == {'status': 'ok'}, 'production health was not OK')
    if launch.get('cache_only'):
        verify_cacheonly_guard(root, launch, completed)
    provenance = json.loads((root / 'provenance.json').read_text())
    require(digest(root.parent / 'ort125-persistent-probe') ==
            provenance['hashes']['/work/ort125-persistent-probe'], 'active build changed')
    for fixture, expected in result['fixture_sha256'].items():
        require(digest(root / fixture) == expected == provenance['hashes'][f'/work/{root.name}/' + fixture],
                f'fixture changed: {fixture}')
    cache = json.loads((root / 'cache-after-validation.json').read_text())
    current = {role + '/' + p.name: digest(p) for role in ['encoder', 'decoder']
               for p in sorted((root / (role + '-cache')).glob('*.mxr'))}
    require(current == cache and cache, 'timed cache hashes changed')
    for provider in ['cpu', 'migraphx']:
        for role in ['encoder', 'decoder']:
            label = f'{provider}-validation-{role}'
            files = list((root / (label + '-profiles')).glob('*.json'))
            require(len(files) == 1, f'{label}: unexpected profile count')
            events = json.loads(files[0].read_text())
            rows = [r for r in result['measurements'] if r['Provider'] == provider and r['Phase'] == 'validation']
            expected_runs = len(rows) if role == 'encoder' else sum(r['DecodeSteps'] for r in rows)
            require(sum(e.get('name') == 'model_run' for e in events) == expected_runs, f'{label}: wrong run count')
            providers = collections.Counter(e.get('args', {}).get('provider') for e in events
                                           if e.get('name', '').endswith('_kernel_time'))
            expected_provider = {'cpu': 'CPUExecutionProvider', 'migraphx': 'MIGraphXExecutionProvider'}[provider]
            require(providers and set(providers) == {expected_provider}, f'{label}: fallback/missing attribution')
            require(not (root / f'{provider}-timing-{role}-profiles').exists(), 'timing profiles exist')
            if provider == 'migraphx':
                require('Model Compile: Begin' not in (root / f'{provider}-timing-{role}.log').read_text(),
                        'compilation marker in timing logs')
                if launch.get('cache_only'):
                    require('Model Compile: Begin' not in (root / f'{label}.log').read_text(),
                            'compilation marker in cache-only validation logs')
    require(len(result['startup']) == 8, 'missing worker startup reports')
    return launch, completed


def verify_cacheonly_guard(root, launch, completed):
    cap = 24 * 1024 ** 3
    stop_at = 22 * 1024 ** 3
    require(launch['memory_limit_unchanged_bytes'] == cap and launch['stop_at_bytes'] == stop_at,
            'unexpected cache-only memory policy')
    require(completed['guard_stop'] is None and completed['cache_unchanged'] is True,
            'cache-only resource/cache gate failed')
    baseline = launch['baseline_memory']
    require(completed['baseline_memory'] == baseline, 'memory baselines differ')
    samples = [json.loads(line) for line in (root / 'memory-samples.jsonl').read_text().splitlines()]
    require(bool(samples), 'missing memory samples')
    require(all(a['elapsed_ns'] < b['elapsed_ns'] for a, b in zip(samples, samples[1:])),
            'nonmonotonic memory samples')
    snapshots = [baseline, *samples, completed['final_memory']]
    for snapshot in snapshots:
        require(snapshot['limit_bytes'] == cap and 0 <= snapshot['current_bytes'] < stop_at,
                'memory limit or sampled guard violated')
        require(snapshot['events']['oom_kill'] == baseline['events']['oom_kill'], 'new OOM kill')
    require(completed['peak_sampled_bytes'] == max(s['current_bytes'] for s in [baseline, *samples]),
            'incorrect sampled memory peak')


def markdown_report(result, summaries):
    lines = ['# Persistent-worker benchmark', '', '**Completed and independently checked: PASS.**', '',
             'Serial two-process diagnostic on one machine; not production HTTP latency or throughput.', '',
             '## Warm request latency', '',
             '| Schedule | Audio | Samples/provider | CPU p50 / p95 | GPU p50 / p95 | Median speedup |',
             '|---|---:|---:|---:|---:|---:|']
    lookup = {(s['Provider'], s['Schedule'], s['Fixture']): s for s in summaries}
    for schedule in ['steady', 'mixed']:
        for seconds in [2, 6, 10]:
            fixture = f'audio-{seconds}s.wav'
            cpu, gpu = (lookup[(p, schedule, fixture)] for p in ['cpu', 'migraphx'])
            lines.append(f'| {schedule} | {seconds} s | {cpu["Count"]} | {cpu["TotalP50MS"]:.1f} / {cpu["TotalP95MS"]:.1f} ms | '
                         f'{gpu["TotalP50MS"]:.1f} / {gpu["TotalP95MS"]:.1f} ms | {cpu["TotalP50MS"] / gpu["TotalP50MS"]:.2f}x |')
    lines += ['', 'The clock includes in-memory WAV parsing, mel preprocessing, both Unix-socket RPCs,',
              'inference, output handling, and complete transcript rendering. Input disk reads, startup,',
              'correctness comparisons, and result persistence are excluded. Profiling/full numerical',
              'validation ran in separate workers. Every timed transcript and decision trajectory matched CPU.', '',
              '## Stage medians (steady schedule)', '',
              '| Provider | Audio | Preprocessing | Encoder native | Complete decoder native | IPC + other overhead |',
              '|---|---:|---:|---:|---:|---:|']
    for s in summaries:
        if s['Schedule'] == 'steady':
            lines.append(f'| {s["Provider"]} | {s["Fixture"]} | {s["PreprocessP50MS"]:.2f} ms | '
                         f'{s["EncoderNativeP50MS"]:.2f} ms | {s["DecoderNativeP50MS"]:.2f} ms | {s["IPCP50MS"]:.2f} ms |')
    lines += ['', 'Residual overhead is **not IPC alone**. Component medians need not add to total median.', '',
              '## Startup and first requests', '',
              '| Worker pair | Role | Launch-to-ready | Session creation |', '|---|---|---:|---:|']
    for s in result['startup']:
        lines.append(f'| {s["Label"]} | {s["Role"]} | {s["HostNS"] / 1e6:.1f} ms | '
                     f'{s["Ready"]["SessionCreationNS"] / 1e6:.1f} ms |')
    lines += ['', '| Provider | Phase | Audio | Request total | Encoder native | First decoder native call |',
              '|---|---|---|---:|---:|---:|']
    for r in result['measurements']:
        if r['Phase'] == 'first-shape' or (r['Provider'] == 'migraphx' and r['Phase'] == 'validation' and r['Sequence'] == 0):
            lines.append(f'| {r["Provider"]} | {r["Phase"]} | {r["Fixture"]} | {r["TotalNS"] / 1e6:.1f} ms | '
                         f'{r["EncoderNativeNS"] / 1e6:.1f} ms | {r["DecoderFirstNativeNS"] / 1e6:.1f} ms |')
    lines += ['', 'First validation requests may include compilation; first-shape timing requests may include',
              'disk-cache loading. Neither is included in warm percentiles. Timed cache hashes stayed unchanged.', '',
              '## Qualifications', '',
              '- One source recording, three nested excerpts (2/6/10 s), one machine, CPU measured first.',
              '- 30 steady and 10 mixed requests per length/provider; nearest-rank percentiles.',
              '- Mixed p95 is the maximum of ten observations, not a stable tail-latency estimate.',
              '- No randomized trials, confidence intervals, concurrency, HTTP or long-audio validation.',
              '- ORT profiles establish provider attribution, not a hardware trace.',
              '- Numerical tolerances were checked by the benchmark; maximum errors are recorded in `result.json`.',
              '- Launcher confirmed unchanged application diff fingerprint and production health OK.', '',
              'Evidence: `result.json`, `launcher-result.json`, validation profiles/traces, worker logs, and cache manifests.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=pathlib.Path, required=True)
    args = parser.parse_args()
    root = args.root
    result = json.loads((root / 'result.json').read_text())
    started = json.loads((root / 'started.json').read_text())
    summaries = verified_summaries(result, started)
    launch, completed = verify_files(root, result)
    text = markdown_report(result, summaries)
    if launch.get('cache_only'):
        text += ('\n## Cache-only retry\n\n'
                 'All shapes were precompiled in the failed v1 run. Fresh worker processes used copies of those caches; '
                 'no compilation markers appeared in validation or timing logs. Cache hashes stayed unchanged.\n\n'
                 f'Container cap remained 24 GiB; the launcher sampled memory with a 22 GiB stop threshold. '
                 f'Peak sampled usage was {completed["peak_sampled_bytes"] / 1024 ** 3:.2f} GiB, with no new OOM kills. '
                 'This sampled peak is not a continuous high-water measurement; short spikes may be missed. '
                 'The resource monitor ran during both CPU and GPU measurements.\n\n'
                 'These results describe cache-only execution, not cold compilation performance. '
                 'The original v1 compilation/shape-switch run remains failed.\n')
    with (root / 'performance-summary.md').open('x') as f:
        f.write(text)
    print(text)


if __name__ == '__main__':
    main()
