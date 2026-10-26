#!/usr/bin/env python3
"""Strict attribution regression checks on temporary profiles only."""
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

CHECKER = pathlib.Path(__file__).with_name('check-decoder-profiles.py')


class CoverageTests(unittest.TestCase):
    def run_case(self, providers, runs=2, expected_runs=2, expected_success=False):
        with tempfile.TemporaryDirectory() as directory:
            events = [{'name': 'model_run'} for _ in range(runs)]
            for provider in providers:
                args = {} if provider is None else {'provider': provider}
                events.append({'name': 'node_kernel_time', 'args': args})
            path = pathlib.Path(directory) / 'profile.json'
            path.write_text(json.dumps(events))
            result = subprocess.run([sys.executable, str(CHECKER), directory,
                                     '--provider', 'migraphx', '--sessions', '1',
                                     '--runs-per-session', str(expected_runs)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode == 0, expected_success, result.stdout + result.stderr)

    def test_exclusively_gpu(self):
        self.run_case(['MIGraphXExecutionProvider'] * 2, expected_success=True)

    def test_cpu_fallback(self):
        self.run_case(['CPUExecutionProvider'] * 2)

    def test_mixed_attribution(self):
        self.run_case(['MIGraphXExecutionProvider', 'CPUExecutionProvider'])

    def test_missing_attribution(self):
        self.run_case(['MIGraphXExecutionProvider', None])

    def test_no_kernels(self):
        self.run_case([])

    def test_wrong_run_count(self):
        self.run_case(['MIGraphXExecutionProvider'], runs=1, expected_runs=2)


if __name__ == '__main__':
    unittest.main()
