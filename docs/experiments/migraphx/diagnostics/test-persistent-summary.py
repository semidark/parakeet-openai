#!/usr/bin/env python3
"""Pure checker regression tests with synthetic data; never run inference."""
import copy
import importlib.util
import json
import pathlib
import tempfile
import unittest

PATH = pathlib.Path(__file__).with_name('summarize-persistent-v1.py')
spec = importlib.util.spec_from_file_location('summary_checker', PATH)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


def synthetic_result():
    started = {'timed_profiling': False, 'timed_repeats': 30, 'mixed_repeats': 10}
    result = {'status': 'PASS', 'percentile_method': 'nearest rank', 'measurements': [],
              'summary': [], 'startup': []}
    counts = {'validation': 2, 'first-shape': 1, 'warmup': 3, 'steady': 30, 'mixed': 10}
    for provider in ['cpu', 'migraphx']:
        for phase, count in counts.items():
            for seconds in [2, 6, 10]:
                for sequence in range(count):
                    result['measurements'].append({
                        'Provider': provider, 'Phase': phase, 'Fixture': f'audio-{seconds}s.wav',
                        'Sequence': sequence, 'Samples': seconds * 16000, 'Frames': seconds * 100 - 2,
                        'EncodedFrames': ((seconds * 100 - 3) // 8) + 1,
                        'Text': 'Synthetic test data', 'DecodeSteps': 1,
                        'Decisions': [{'Frame': 0, 'Token': 1, 'Duration': 2}],
                        'TotalNS': 10000000, 'PreprocessNS': 1000000, 'EncoderRPCNS': 4000000,
                        'DecoderRPCNS': 4000000, 'EncoderNativeNS': 3000000,
                        'DecoderNativeNS': 2000000, 'DecoderFirstNativeNS': 2000000})
        for schedule in ['steady', 'mixed']:
            for seconds in [2, 6, 10]:
                result['summary'].append({
                    'Provider': provider, 'Schedule': schedule, 'Fixture': f'audio-{seconds}s.wav',
                    'Count': counts[schedule], 'TotalP50MS': 10.0, 'TotalP95MS': 10.0,
                    'TotalMinMS': 10.0, 'TotalMaxMS': 10.0, 'EncoderNativeP50MS': 3.0,
                    'DecoderNativeP50MS': 2.0, 'PreprocessP50MS': 1.0, 'IPCP50MS': 4.0,
                    'RealTimeFactorP50': 10 / (seconds * 1000),
                    'RealTimeFactorP95': 10 / (seconds * 1000)})
    return result, started


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.result, self.started = synthetic_result()

    def check(self):
        return checker.verified_summaries(self.result, self.started)

    def test_valid_data(self):
        self.assertEqual(len(self.check()), 12)

    def test_independent_nearest_rank(self):
        self.assertEqual(checker.percentile(list(range(1, 31)), .95), 29)
        self.assertEqual(checker.percentile(list(range(1, 11)), .95), 10)
        with self.assertRaises(ValueError):
            checker.percentile([], .5)

    def test_reject_failure_and_profiling(self):
        self.result['status'] = 'FAIL'
        with self.assertRaises(ValueError):
            self.check()
        self.result['status'] = 'PASS'
        self.started['timed_profiling'] = True
        with self.assertRaises(ValueError):
            self.check()

    def test_missing_and_duplicate_samples(self):
        self.result['measurements'].pop()
        with self.assertRaises(ValueError):
            self.check()
        self.result, self.started = synthetic_result()
        rows = [r for r in self.result['measurements'] if r['Provider'] == 'migraphx'
                and r['Phase'] == 'steady' and r['Fixture'] == 'audio-6s.wav']
        rows[1]['Sequence'] = rows[0]['Sequence']
        with self.assertRaises(ValueError):
            self.check()

    def test_trajectory_drift(self):
        self.result['measurements'][-1]['Decisions'][0]['Duration'] = 3
        with self.assertRaises(ValueError):
            self.check()

    def test_bad_clocks_and_input_lengths(self):
        original = copy.deepcopy(self.result)
        for key, value in [('TotalNS', 1), ('EncoderNativeNS', 9000000),
                           ('DecoderFirstNativeNS', 9000000), ('Frames', 200)]:
            self.result = copy.deepcopy(original)
            self.result['measurements'][-1][key] = value
            with self.assertRaises(ValueError, msg=key):
                self.check()

    def test_bad_summary(self):
        self.result['summary'][0]['TotalP95MS'] = 999.0
        with self.assertRaises(ValueError):
            self.check()

    def test_duplicate_summary(self):
        self.result['summary'].append(self.result['summary'][0])
        with self.assertRaises(ValueError):
            self.check()

    def test_report_qualifications(self):
        text = checker.markdown_report(self.result, self.check())
        self.assertIn('IPC + other overhead', text)
        self.assertIn('not IPC alone', text)
        self.assertIn('maximum of ten observations', text)
        self.assertIn('not production HTTP latency', text)
        self.assertIn('Synthetic', self.result['measurements'][0]['Text'])


class CacheOnlyEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='summary-test-')
        self.addCleanup(self.directory.cleanup)
        self.root = pathlib.Path(self.directory.name)
        baseline = {'current_bytes': 1024 ** 3, 'limit_bytes': 24 * 1024 ** 3,
                    'events': {'oom_kill': 1}}
        self.launch = {'memory_limit_unchanged_bytes': 24 * 1024 ** 3,
                       'stop_at_bytes': 22 * 1024 ** 3, 'baseline_memory': baseline}
        self.completed = {'guard_stop': None, 'cache_unchanged': True,
                          'baseline_memory': copy.deepcopy(baseline),
                          'final_memory': copy.deepcopy(baseline),
                          'peak_sampled_bytes': 2 * 1024 ** 3}
        self.samples = [{'elapsed_ns': i + 1, 'current_bytes': 2 * 1024 ** 3,
                         'limit_bytes': 24 * 1024 ** 3, 'events': {'oom_kill': 1}}
                        for i in range(2)]

    def check(self):
        (self.root / 'memory-samples.jsonl').write_text(
            ''.join(json.dumps(sample) + '\n' for sample in self.samples))
        checker.verify_cacheonly_guard(self.root, self.launch, self.completed)

    def test_valid_existing_oom_baseline(self):
        self.check()

    def test_reject_failed_gates(self):
        original = copy.deepcopy(self.completed)
        for field, value in [('guard_stop', 'memory guard'), ('cache_unchanged', False),
                             ('peak_sampled_bytes', 1)]:
            self.completed = copy.deepcopy(original)
            self.completed[field] = value
            with self.assertRaises(ValueError, msg=field):
                self.check()

    def test_reject_new_oom_or_limit_change(self):
        original = copy.deepcopy(self.samples)
        for field, value in [('events', {'oom_kill': 2}), ('limit_bytes', 48 * 1024 ** 3),
                             ('current_bytes', 22 * 1024 ** 3)]:
            self.samples = copy.deepcopy(original)
            self.samples[0][field] = value
            with self.assertRaises(ValueError, msg=field):
                self.check()

    def test_reject_missing_or_nonmonotonic_samples(self):
        self.samples[1]['elapsed_ns'] = self.samples[0]['elapsed_ns']
        with self.assertRaises(ValueError):
            self.check()
        self.samples = []
        with self.assertRaises(ValueError):
            self.check()


if __name__ == '__main__':
    unittest.main()
