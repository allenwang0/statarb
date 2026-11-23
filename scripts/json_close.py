"""
Compare two JSON files: exact on structure, keys, strings, booleans and
integers; floats within an absolute tolerance. Used in CI to check that a fresh
case-study run matches the committed results/case_study.json, since the last
digits of statsmodels/BLAS output can differ between machines.

    python scripts/json_close.py committed.json fresh.json [--atol 1e-9]
"""

import argparse
import json
import sys


def diff(a, b, path="$", atol=1e-9):
    if isinstance(a, float) or isinstance(b, float):
        ok = isinstance(a, int | float) and isinstance(b, int | float) and not isinstance(a, bool)
        if ok and abs(a - b) <= atol:
            return []
        return [f"{path}: {a!r} != {b!r}"]
    if type(a) is not type(b):
        return [f"{path}: type {type(a).__name__} != {type(b).__name__}"]
    if isinstance(a, dict):
        if a.keys() != b.keys():
            return [f"{path}: keys differ: {sorted(a.keys() ^ b.keys())}"]
        return [m for k in a for m in diff(a[k], b[k], f"{path}.{k}", atol)]
    if isinstance(a, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} != {len(b)}"]
        return [
            m
            for i, (x, y) in enumerate(zip(a, b, strict=True))
            for m in diff(x, y, f"{path}[{i}]", atol)
        ]
    return [] if a == b else [f"{path}: {a!r} != {b!r}"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("expected")
    parser.add_argument("actual")
    parser.add_argument("--atol", type=float, default=1e-9)
    args = parser.parse_args()
    with open(args.expected) as f, open(args.actual) as g:
        problems = diff(json.load(f), json.load(g), atol=args.atol)
    for line in problems[:20]:
        print(line)
    if problems:
        sys.exit(f"{len(problems)} difference(s) beyond atol={args.atol}")
    print("JSON matches within tolerance")


if __name__ == "__main__":
    main()
