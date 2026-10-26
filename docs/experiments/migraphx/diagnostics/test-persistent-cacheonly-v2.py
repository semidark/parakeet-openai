#!/usr/bin/env python3
"""Pure resource-guard tests. No inference or environment mutation."""
import copy
import importlib.util
import pathlib
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('cacheonly', pathlib.Path(__file__).with_name('run-persistent-cacheonly-v2.py'))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.baseline = {'current_bytes': 1024 ** 3, 'limit_bytes': runner.CAP,
                         'events': {'oom_kill': 1}}

    def test_existing_oom_counter_does_not_fail_retry(self):
        self.assertIsNone(runner.guard_reason(self.baseline, self.baseline, False))

    def test_new_oom(self):
        current = copy.deepcopy(self.baseline)
        current['events']['oom_kill'] = 2
        self.assertIn('new container OOM', runner.guard_reason(current, self.baseline, False))

    def test_memory_guard_below_cap(self):
        current = copy.deepcopy(self.baseline)
        current['current_bytes'] = runner.STOP_AT
        self.assertLess(runner.STOP_AT, runner.CAP)
        self.assertIn('22 GiB', runner.guard_reason(current, self.baseline, False))

    def test_cap_change(self):
        current = copy.deepcopy(self.baseline)
        current['limit_bytes'] = 48 * 1024 ** 3
        self.assertIn('cap changed', runner.guard_reason(current, self.baseline, False))

    def test_compilation_forbidden(self):
        self.assertIn('unexpected compilation', runner.guard_reason(self.baseline, self.baseline, True))


    def test_compilation_scan_covers_both_workers(self):
        with tempfile.TemporaryDirectory(prefix='cacheonly-test-') as directory:
            root = pathlib.Path(directory)
            self.assertFalse(runner.compilation_seen(root))
            for role in ['encoder', 'decoder']:
                path = root / f'migraphx-validation-{role}.log'
                path.write_text('Model Compile: Begin\n')
                self.assertTrue(runner.compilation_seen(root), role)
                path.write_text('ordinary provider warning\n')
                self.assertFalse(runner.compilation_seen(root), role)


if __name__ == '__main__':
    unittest.main()
