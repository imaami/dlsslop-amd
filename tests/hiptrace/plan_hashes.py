#!/usr/bin/python3
"""Print the values hip-plan-test pins, as a trace of dlsslopd serving frames shows them.

Per trace, one line: the FNV-1a 64 and launch count of the canonical launch
list of each of the first three network evaluations, and the distinct kernels
of the second. A canonical list has a line per launch, LABEL, GRID, BLOCK, LDS
and ARGUMENTS separated by tabs, with each buffer renamed B<k>+<offset> in the
order of first appearance, as compare.py compares them. hip-plan-test's first,
later and history hashes are eval0 and eval1 of a served trace and eval2 of a
served trace with --motion.

Then, from the synchronous host-to-device copies up to the end of the first
evaluation: the pool (the device allocations that are never an upload's
destination, in creation order, less the network's input and output), the
gather maps (BYTES:FNV of each upload that vit_gather reads as argument 1) and
the weights: how many are read and their bytes, the FNV-1a 64 of the lines
"BYTES LAUNCH ARGUMENT\\n" in upload order (the first launch of the first
evaluation to read the weight, and the argument), and how many no launch reads.
"""
import argparse
import sys

import analyze


def fnv(data, h=14695981039346656037):
    for b in data:
        h = ((h ^ b) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def rows(evaluation):
    return "".join("\t".join(c[:4]) + "\t" + ",".join(c[4]) + "\n" for c in analyze.canonical(evaluation))


def allocation(argument):
    match = analyze.POINTER.match(argument)
    return match.group(1) if match else None


def values(path):
    records = analyze.parse(path)
    evals, first = [], None
    for seq, call, fields, positional, deep, line in records:
        if call not in analyze.LAUNCH_CALLS:
            continue
        l = analyze.Launch(seq, fields, deep)
        if not l.network:
            continue
        first = first or l.label
        if l.label == first:
            evals.append([])
        evals[-1].append(l)
    if not evals:
        return evals, f"{path}: no network evaluation"
    steady = evals[1] if len(evals) > 1 else evals[0]
    head = [path.split("/")[-1]]
    head += [f"eval{i}={fnv(rows(e).encode()):016x}/{len(e)}" for i, e in enumerate(evals[:3])]
    head.append(f"kernels={len({l.label for l in steady})}")

    # Each allocation's first reader in the first evaluation: (position, argument, kernel).
    readers = {}
    for position, l in enumerate(evals[0]):
        for index, argument in enumerate(l.args):
            if allocation(argument):
                readers.setdefault(allocation(argument), (position, index, l.kernel))
    end = evals[0][-1].seq
    engine = {allocation(evals[0][0].args[0])}
    engine |= {allocation(l.args[7]) for l in evals[0] if l.kernel == "c32_post_merge_head_half"}
    created, uploaded, copies = [], set(), []
    for seq, call, fields, positional, deep, line in records:
        if call == "hipMalloc" and positional and "rc" not in fields and seq <= end:
            created.append((positional[-1], int(fields["bytes"])))
        elif call in ("hipMemcpy", "hipMemcpyAsync") and "src_fnv" in fields:
            uploaded.add(fields["dst"].split("+")[0])
            if call == "hipMemcpy" and seq <= end:
                copies.append((fields["dst"].split("+")[0], int(fields["bytes"]), fields["src_fnv"]))
    pool = [size for ident, size in created if ident not in uploaded and ident not in engine]
    gathers = [f"{size}:{payload}" for ident, size, payload in copies
               if readers.get(ident, (0, 0, ""))[1:] == (1, "vit_gather")]
    weights = [(ident, size) for ident, size, payload in copies if readers.get(ident, (0, 0, ""))[2] != "vit_gather"]
    read = [(size, readers[ident]) for ident, size in weights if ident in readers]
    digest = fnv("".join(f"{size} {position} {index}\n" for size, (position, index, _) in read).encode())
    return evals, "\n".join([
        " ".join(head),
        f"  pool {' '.join(map(str, pool))} ({len(pool)})",
        "  gathers" + "".join(f" {g}" for g in gathers),
        f"  weights {len(read)} {sum(size for size, _ in read)} uploads={digest:016x} "
        f"unread={len(weights) - len(read)}",
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False, allow_abbrev=False,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("traces", nargs="+", metavar="TRACE", help="trace to read (required; no default)")
    parser.add_argument("-p", "--print", type=int, metavar="EVAL", dest="evaluation",
                        help="instead, print the canonical launch list of evaluation EVAL (0: the first) of each "
                             "TRACE, as hip-plan-test --print prints the plan's (default: unset)")
    parser.add_argument("-h", "--help", action="help", help="show this help and exit (default: off)")
    args = parser.parse_args()
    status = 0
    for path in args.traces:
        evals, text = values(path)
        if not evals:
            print(text, file=sys.stderr)
            status = 1
        elif args.evaluation is None:
            print(text)
        elif 0 <= args.evaluation < len(evals):
            sys.stdout.write(rows(evals[args.evaluation]))
        else:
            print(f"{path}: no evaluation {args.evaluation}; it has {len(evals)}", file=sys.stderr)
            status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
