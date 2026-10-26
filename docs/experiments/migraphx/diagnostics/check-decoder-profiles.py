#!/usr/bin/env python3
"""Report provider attribution; fail full-GPU validation on fallback/missing events."""
import argparse
import collections
import json
import pathlib
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=pathlib.Path)
    parser.add_argument("--provider", choices=["cpu", "migraphx"], required=True)
    parser.add_argument("--sessions", type=int, required=True)
    parser.add_argument("--runs-per-session", type=int, required=True)
    args = parser.parse_args()
    files = sorted(args.directory.glob("*.json"))
    if args.sessions < 1 or args.runs_per_session < 1 or len(files) != args.sessions:
        raise ValueError(f"expected {args.sessions} profiles; found {len(files)}")
    expected = {"cpu": "CPUExecutionProvider", "migraphx": "MIGraphXExecutionProvider"}[args.provider]
    total = collections.Counter()
    problems = []
    for path in files:
        events = json.loads(path.read_text())
        kernels = [e for e in events if e.get("name", "").endswith("_kernel_time")]
        counts = collections.Counter(e.get("args", {}).get("provider", "MISSING") for e in kernels)
        total.update(counts)
        runs = [e for e in events if e.get("name") == "model_run"]
        print(f"PROFILE {path.name}: model_runs={len(runs)} kernel_providers={dict(counts)}")
        if len(runs) != args.runs_per_session:
            problems.append(f"{path.name}: unexpected model_run count {len(runs)}")
        if not kernels or set(counts) != {expected}:
            problems.append(f"{path.name}: provider coverage is not exclusively {expected}")
    print(f"TOTAL KERNEL PROVIDERS: {dict(total)}")
    if problems:
        raise ValueError("; ".join(problems))
    print(f"DECODER {args.provider.upper()} PROFILE COVERAGE: PASS (ORT attribution, not a hardware trace)")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"DECODER PROFILE COVERAGE: FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
