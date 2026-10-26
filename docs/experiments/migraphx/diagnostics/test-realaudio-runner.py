#!/usr/bin/env python3
"""Pure regression tests; never launch inference or modify run evidence."""
import copy
import importlib.util
import pathlib
import tempfile
import unittest
from unittest import mock

PATH = pathlib.Path(__file__).with_name('run-realaudio-v1.py')
spec = importlib.util.spec_from_file_location('realaudio_runner', PATH)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class RunnerTests(unittest.TestCase):
    def test_full_transcription_comparison(self):
        baseline = {'InputHash': 'input', 'ModelHash': 'decoder', 'VocabHash': 'vocab',
                    'Lengths': [75], 'Text': 'Real speech',
                    'Decisions': [{'Frame': 0, 'Token': 1, 'Duration': 2}]}
        runner.same_transcription(copy.deepcopy(baseline), baseline, 'test')
        for key in baseline:
            changed = copy.deepcopy(baseline)
            changed[key] = None
            with self.assertRaises(RuntimeError, msg=key):
                runner.same_transcription(changed, baseline, 'test')

    def test_empty_transcript_rejected(self):
        blank = {'InputHash': 'input', 'ModelHash': 'decoder', 'VocabHash': 'vocab',
                 'Lengths': [75], 'Text': '', 'Decisions': [{'Frame': 0}]}
        with self.assertRaises(RuntimeError):
            runner.same_transcription(blank, blank, 'test')

    def test_exclusive_notes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / 'note.json'
            runner.exclusive_json(path, {'original': True})
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                runner.exclusive_json(path, {'changed': True})
            self.assertEqual(path.read_bytes(), before)

    def test_cache_manifest_detects_changes(self):
        old_out = runner.OUT
        try:
            with tempfile.TemporaryDirectory() as directory:
                runner.OUT = pathlib.Path(directory)
                cache = runner.OUT / 'cache'
                cache.mkdir()
                with self.assertRaises(RuntimeError):
                    runner.cache_manifest('cache')
                artifact = cache / 'shape.mxr'
                artifact.write_bytes(b'original')
                first = runner.cache_manifest('cache')
                self.assertEqual(first, runner.cache_manifest('cache'))
                artifact.write_bytes(b'changed')
                self.assertNotEqual(first, runner.cache_manifest('cache'))
        finally:
            runner.OUT = old_out

    def test_known_overrides_are_cleared(self):
        self.assertIn('MIGRAPHX_DISABLE_PASSES', runner.CLEAR)
        self.assertIn('ORT_MIGRAPHX_MODEL_CACHE_PATH', runner.CLEAR)
        self.assertIn('ORT_MIGRAPHX_FP16_ENABLE', runner.CLEAR)

    def test_process_environment_isolation(self):
        old_out, old_results = runner.OUT, runner.stage_results
        try:
            with tempfile.TemporaryDirectory() as directory:
                runner.OUT = pathlib.Path(directory)
                runner.stage_results = []
                for label, kind in [('encoder', 'encode'), ('decoder', 'decode')]:
                    runner.exclusive_json(runner.OUT / f'{label}.json', {'Runs': 2, 'Text': 'speech'})
                    with mock.patch.object(runner.subprocess, 'run', return_value=mock.Mock(returncode=0)) as run, mock.patch.object(runner, 'check_profiles'):
                        runner.stage(label, kind, 'migraphx', 'in.gob', 'out.gob', 'ref.gob', label + '-cache')
                    command = run.call_args.args[0]
                    self.assertIn('env', command)
                    for name in runner.CLEAR:
                        index = command.index(name)
                        self.assertEqual(command[index - 1], '-u')
                    if kind == 'encode':
                        self.assertIn('MIGRAPHX_DISABLE_PASSES=simplify_reshapes', command)
                    else:
                        self.assertFalse(any(a.startswith('MIGRAPHX_DISABLE_PASSES=') for a in command))
                    self.assertIn('/work/realaudio-v1/' + label + '-cache', command)
                    self.assertIn('timeout', command)
        finally:
            runner.OUT, runner.stage_results = old_out, old_results


if __name__ == '__main__':
    unittest.main()
