#!/usr/bin/python3
"""Whether trace B's network evaluations match trace A's.

Per evaluation: kernel list, grids, blocks, LDS and scalar arguments must be
equal, and buffer arguments equal after renaming each buffer by first
appearance (so a replacement may allocate differently but must alias the same
way). Also compares the host-to-device uploads and, for traces that hashed
c32_post_merge_head_half (HIPTRACE_DEEP_KERNELS), the network output of every
evaluation both traces hold. Prints MATCH, or the differences and DIFFERENT
with exit status 1. A trace without a network evaluation is a difference.
"""
import argparse
import sys

import analyze


def load(path):
    evals, first, uploads, outputs = [], None, [], {}
    for seq, call, fields, positional, deep, line in analyze.parse(path):
        if call in analyze.LAUNCH_CALLS:
            l = analyze.Launch(seq, fields, deep)
            if not l.network:
                continue
            first = first or l.label
            if l.label == first:
                evals.append([])
            evals[-1].append(l)
            if l.kernel == "c32_post_merge_head_half" and "a7" in deep:
                outputs[len(evals) - 1] = deep["a7"]
        elif call == "hipMemcpy" and fields.get("kind") == "H2D":
            uploads.append((fields["bytes"], fields["src_fnv"]))
    return evals, uploads, outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False, allow_abbrev=False,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("a", metavar="A", help="the reference trace (required; no default)")
    parser.add_argument("b", metavar="B", help="the trace to check (required; no default)")
    parser.add_argument("-h", "--help", action="help", help="show this help and exit (default: off)")
    args = parser.parse_args()
    a, b = load(args.a), load(args.b)
    empty = [path for path, trace in ((args.a, a), (args.b, b)) if not trace[0]]
    for path in empty:
        print(f"{path}: no network evaluation")
    if empty:
        print("DIFFERENT")
        return 1
    bad = 0
    if len(a[0]) != len(b[0]):
        print(f"evaluations: {len(a[0])} vs {len(b[0])}")
        bad = 1
    for i, (x, y) in enumerate(zip(a[0], b[0])):
        cx, cy = analyze.canonical(x), analyze.canonical(y)
        if cx == cy:
            continue
        bad = 1
        first = next((k for k, (u, v) in enumerate(zip(cx, cy)) if u != v), min(len(cx), len(cy)))
        print(f"eval{i}: {len(x)} vs {len(y)} launches; first difference at launch {first}:")
        print(f"  A {cx[first] if first < len(cx) else '-'}")
        print(f"  B {cy[first] if first < len(cy) else '-'}")
    if a[1] != b[1]:
        if sorted(a[1]) == sorted(b[1]):
            print(f"host-to-device uploads: the same {len(a[1])} payloads in a different order")
        else:
            bad = 1
            only_a = len(set(a[1]) - set(b[1]))
            only_b = len(set(b[1]) - set(a[1]))
            print(f"host-to-device uploads differ: {len(a[1])} vs {len(b[1])} copies; {only_a} payloads only in A, "
                  f"{only_b} only in B")
    common = sorted(set(a[2]) & set(b[2]))
    for i in common:
        if a[2][i] != b[2][i]:
            bad = 1
            print(f"eval{i}: network output {a[2][i]} vs {b[2][i]}")
    if common:
        print(f"network outputs compared for evaluations {common}")
    print("MATCH" if not bad else "DIFFERENT")
    return bad


if __name__ == "__main__":
    sys.exit(main())
