#!/usr/bin/python3
"""Summarise traces of the tracing HIP runtime: evaluations, frames, kernels, allocations, uploads.

Writes, next to each NAME.trace:
  NAME.summary.txt   the summary
  NAME.evaldiff.txt  every launch that differs between consecutive network evaluations
  NAME.allocs.tsv    every allocation: id, bytes, phase, uploads, freed, first use
  NAME.uploads.tsv   every host-to-device copy: order, call, destination, bytes, FNV, phase, source, first use
  NAME.eval.tsv      the steady-state network evaluation's launches (the second evaluation)
  NAME.frame.tsv     the second served frame's calls (mark 0 to the last timing call), numbers dropped
"""
import argparse
import collections
import difflib
import re

NATIVE = "linux_native.hsaco"
POINTER = re.compile(r"^([dhrx]\d+)\+(\d+)$")
# The trace lines of kernel launches: hipModuleLaunchKernel and hipExtModuleLaunchKernel.
LAUNCH_CALLS = ("launch", "ext_launch", "ext_launch_anyorder")


def fnv(text):
    h = 14695981039346656037
    for b in text.encode():
        h = ((h ^ b) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


class Launch:
    def __init__(self, seq, fields, deep):
        self.seq = seq
        self.n = int(fields["n"])
        self.label = fields["k"]
        self.file, _, self.kernel = self.label.partition(":")
        self.grid, self.block, self.lds, self.stream = fields["grid"], fields["block"], fields["lds"], fields["s"]
        self.args = fields["args"].split(",") if fields["args"] else []
        self.deep = deep
        self.network = self.file != NATIVE

    def shape(self):
        return (self.label, self.grid, self.block, self.lds)

    def signature(self):
        return self.shape() + (tuple(self.args),)

    def row(self):
        return f"{self.label}\tgrid={self.grid}\tblock={self.block}\tlds={self.lds}\t{','.join(self.args)}"


def parse(path):
    records = []
    for line in open(path):
        line = line.rstrip("\n")
        head = line.split(" ", 3)
        if len(head) < 3:
            continue
        seq, call = int(head[0]), head[2]
        rest = head[3] if len(head) > 3 else ""
        deep = {}
        if call in LAUNCH_CALLS and " deep" in rest:
            rest, _, tail = rest.partition(" deep")
            deep = dict(t.split("=", 1) for t in tail.split())
        fields, positional = {}, []
        for token in re.findall(r'(\S+="[^"]*"|\S+)', rest):
            if "=" in token:
                k, v = token.split("=", 1)
                fields[k] = v
            else:
                positional.append(token)
        records.append((seq, call, fields, positional, deep, line))
    return records


def canonical(evaluation):
    """The launch list with buffers renamed in order of first appearance: equal
    canonical forms mean the same kernels, shapes, scalars and aliasing."""
    names = {}
    out = []
    for l in evaluation:
        args = []
        for a in l.args:
            m = POINTER.match(a)
            if m:
                args.append(f"B{names.setdefault(m.group(1), len(names))}+{m.group(2)}")
            else:
                args.append(a)
        out.append(l.shape() + (tuple(args),))
    return out


def analyze(path):
    name = path[:-6] if path.endswith(".trace") else path
    records = parse(path)
    # Pass 1: launches, evaluations, frames.
    launches, evaluations, frames = [], [], []
    first_kernel = None
    frame = None
    counts = collections.Counter()
    for seq, call, fields, positional, deep, line in records:
        counts[call] += 1
        if call == "hipEventRecord" and positional and positional[0] == "e1":
            frame = {"index": len(frames), "start": seq, "end": None, "calls": []}
            frames.append(frame)
        if frame is not None and frame["end"] is None:
            frame["calls"].append(line.split(" ", 1)[1])
            if call == "hipEventElapsedTime" and positional[:2] == ["e3", "e4"]:
                frame["end"] = seq
        if call in LAUNCH_CALLS:
            l = Launch(seq, fields, deep)
            launches.append(l)
            if l.network:
                if first_kernel is None:
                    first_kernel = l.label
                if l.label == first_kernel:
                    evaluations.append([])
                evaluations[-1].append(l)
    ranges = [(e[0].seq, e[-1].seq) for e in evaluations]
    for f in frames:
        f["end"] = f["end"] if f["end"] is not None else records[-1][0]
        f["evaluations"] = [i for i, (a, b) in enumerate(ranges) if f["start"] <= a <= f["end"]]
        f["launches"] = [l for l in launches if f["start"] <= l.seq <= f["end"]]

    def phase(seq):
        for i, (a, b) in enumerate(ranges):
            if a <= seq <= b:
                return f"eval{i}"
        for f in frames:
            if f["start"] <= seq <= f["end"]:
                return f"frame{f['index']}"
        if not ranges or seq < ranges[0][0]:
            return "setup"
        last = frames[-1]["end"] if frames else ranges[-1][1]
        if seq > last:
            return "teardown"
        if not frames or seq < frames[0]["start"]:
            return "prepare"
        return "between"

    # Pass 2: allocations, uploads, frees, first uses.
    allocs = collections.OrderedDict()
    uploads, frees, first_use = [], {}, {}
    for seq, call, fields, positional, deep, line in records:
        if call in ("hipMalloc", "hipHostMalloc", "hipHostRegister") or call == "hipExternalMemoryGetMappedBuffer":
            if positional and "rc" not in fields:
                ident = positional[-1]
                allocs[ident] = {"bytes": int(fields.get("bytes", 0)), "phase": phase(seq), "uploads": 0}
        elif call in ("hipFree", "hipHostFree", "hipHostUnregister") and positional:
            frees.setdefault(positional[0].split("+")[0], phase(seq))
        elif call in ("hipMemcpy", "hipMemcpyAsync") and "src_fnv" in fields:
            dst = fields["dst"].split("+")[0]
            if dst in allocs:
                allocs[dst]["uploads"] += 1
            uploads.append([len(uploads), call, fields["dst"], int(fields["bytes"]), fields["src_fnv"], phase(seq),
                            fields.get("src", "")])
    for l in launches:
        for i, a in enumerate(l.args):
            m = POINTER.match(a)
            if m:
                first_use.setdefault(m.group(1), f"{l.label}#a{i}@n{l.n}")

    out = []
    w = out.append
    w(f"trace {path}")
    w(f"header: {records[0][5] if records else ''}")
    w("")
    w("== calls ==")
    for call, n in sorted(counts.items(), key=lambda x: -x[1]):
        w(f"  {call:36s} {n}")
    w("")
    w(f"== network evaluations: {len(evaluations)} (each starts with {first_kernel}) ==")
    w("  launches per evaluation: " + " ".join(str(len(e)) for e in evaluations))
    w(f"  all launches {len(launches)}; network {sum(len(e) for e in evaluations)}; "
      f"linux_native {sum(1 for l in launches if not l.network)}")
    diff_lines = []
    if evaluations:
        w("")
        w("== consecutive evaluations compared (full list in NAME.evaldiff.txt) ==")
        for j in range(1, len(evaluations)):
            a, b = evaluations[j - 1], evaluations[j]
            if [x.shape() for x in a] != [x.shape() for x in b]:
                w(f"  eval{j} vs eval{j - 1}: KERNEL LIST DIFFERS ({len(b)} vs {len(a)} launches)")
                sa, sb = [x.label for x in a], [x.label for x in b]
                for op in difflib.SequenceMatcher(a=sa, b=sb, autojunk=False).get_opcodes():
                    if op[0] != "equal":
                        w(f"    {op[0]} eval{j - 1}[{op[1]}:{op[2]}] -> eval{j}[{op[3]}:{op[4]}]: "
                          f"{sorted(set(sa[op[1]:op[2]]))[:4]} -> {sorted(set(sb[op[3]:op[4]]))[:4]}")
                continue
            differing = []
            for pos, (x, y) in enumerate(zip(a, b)):
                if x.signature() != y.signature():
                    changed = [f"a{i}:{u}->{v}" for i, (u, v) in enumerate(zip(x.args, y.args)) if u != v]
                    differing.append((pos, x.kernel, changed))
            scalars = [d for d in differing if any(not POINTER.match(c.split(":", 1)[1].split("->")[0]) or
                                                   not POINTER.match(c.split("->")[1]) for c in d[2])]
            same_flow = canonical(a) == canonical(b)
            if not differing:
                verdict = "identical launch list (kernels, shapes, arguments, buffers)"
            else:
                verdict = (f"{len(differing)} launches differ; {len(scalars)} of them in non-buffer arguments; "
                           f"{'same dataflow up to buffer renaming' if same_flow else 'DATAFLOW DIFFERS after renaming'}")
            w(f"  eval{j} vs eval{j - 1}: {verdict}")
            for pos, kernel, changed in scalars[:12]:
                w(f"    [{pos}] {kernel}: {' '.join(changed)}")
            diff_lines.append(f"== eval{j} vs eval{j - 1}: {verdict}")
            for pos, kernel, changed in differing:
                diff_lines.append(f"  [{pos}] {kernel}: {' '.join(changed)}")
        steady = evaluations[1] if len(evaluations) > 1 else evaluations[0]
        w("")
        w(f"== distinct network kernels in the steady-state evaluation ({'eval1' if len(evaluations) > 1 else 'eval0'}) ==")
        per = collections.OrderedDict()
        for l in steady:
            per.setdefault(l.label, []).append(l)
        w(f"  {len(per)} distinct kernels, {len(steady)} launches")
        for label, ls in per.items():
            shapes = sorted(set(f"{x.grid}/{x.block}" for x in ls))
            w(f"  {len(ls):4d}  {label}  lds={ls[0].lds}  shapes={' '.join(shapes)}")
        files = collections.Counter(l.file for l in steady)
        w("  per module: " + ", ".join(f"{f}={n}" for f, n in files.most_common()))
        with open(name + ".eval.tsv", "w") as f:
            for i, l in enumerate(steady):
                f.write(f"{i}\t{l.row()}\n")
    w("")
    w(f"== served or self-test frames (hipEventRecord e1 .. hipEventElapsedTime e3 e4): {len(frames)} ==")
    for f in frames:
        natives = collections.Counter(l.kernel for l in f["launches"] if not l.network)
        net = sum(1 for l in f["launches"] if l.network)
        calls = collections.Counter(c.split(" ")[1] for c in f["calls"])
        w(f"  frame{f['index']}: evaluations {f['evaluations']}; launches {len(f['launches'])} (network {net}); "
          f"native {dict(natives)}")
        w("    other calls: " + ", ".join(f"{k}={v}" for k, v in calls.most_common() if k != "launch"))

    def stripped(f):
        return [re.sub(r" n=\d+", "", c) for c in f["calls"]]

    for j in range(1, len(frames)):
        a, b = stripped(frames[j - 1]), stripped(frames[j])
        ops = [op for op in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes() if op[0] != "equal"]
        w(f"  frame{j} vs frame{j - 1}: {len(b)} vs {len(a)} calls; {len(ops)} differing runs")
        for tag, i1, i2, j1, j2 in ops[:8]:
            for x in a[i1:i2][:4]:
                w(f"    - [{i1}] {x[:220]}")
            for y in b[j1:j2][:4]:
                w(f"    + [{j1}] {y[:220]}")
    if len(frames) > 1:
        with open(name + ".frame.tsv", "w") as f:
            f.write("\n".join(stripped(frames[1])) + "\n")
    w("")
    w("== allocations by phase (uploaded = the target of a host-to-device copy) ==")
    groups = collections.OrderedDict()
    for ident, a in allocs.items():
        key = (a["phase"], ident[0], "uploaded" if a["uploads"] else "scratch")
        g = groups.setdefault(key, [0, 0, collections.Counter()])
        g[0] += 1
        g[1] += a["bytes"]
        g[2][a["bytes"]] += 1
    for (ph, kind, what), (n, b, sizes) in groups.items():
        detail = "" if what == "uploaded" else " sizes=" + ",".join(
            f"{s}x{c}" if c > 1 else str(s) for s, c in sorted(sizes.items(), key=lambda x: -x[0]))
        w(f"  {ph:10s} {kind} {what:8s} count={n:4d} bytes={b:11d} ({b / 1048576:7.2f} MiB){detail}")
    device = [a for i, a in allocs.items() if i[0] == "d"]
    total = sum(a["bytes"] for a in device)
    w(f"  device: {len(device)} allocations, {total} bytes ({total / 1048576:.2f} MiB); "
      f"freed by phase: {dict(collections.Counter(frees.get(i, 'never') for i in allocs if i[0] == 'd'))}")
    w("")
    sync = [u for u in uploads if u[1] == "hipMemcpy"]
    ub = sum(u[3] for u in sync)
    w(f"== synchronous host-to-device copies: {len(sync)}, {ub} bytes ({ub / 1048576:.2f} MiB); "
      f"ordered digest {fnv(''.join(f'{u[3]}:{u[4]}' + chr(10) for u in sync)):016x} ==")
    w("  (digest = FNV-1a 64 of the lines 'BYTES:SRC_FNV\\n' in upload order)")
    by = collections.OrderedDict()
    for u in sync:
        c = by.setdefault(u[5], [0, 0])
        c[0] += 1
        c[1] += u[3]
    w("  by phase: " + ", ".join(f"{k}={n} ({b} bytes)" for k, (n, b) in by.items()))
    net = [u for u in sync if u[5] in ("setup", "eval0")]
    gathers = [u for u in net if "vit_gather#a1" in first_use.get(u[2].split("+")[0], "")]
    weights = [u for u in net if u not in gathers]
    w(f"  of which ViT gather maps (vit_gather argument 1): {len(gathers)} copies, "
      f"{' '.join(str(u[3]) + ':' + u[4] for u in gathers)}")
    w(f"  model weights alone: {len(weights)} copies, {sum(u[3] for u in weights)} bytes "
      f"({sum(u[3] for u in weights) / 1048576:.2f} MiB), ordered digest "
      f"{fnv(''.join(f'{u[3]}:{u[4]}' + chr(10) for u in weights)):016x}")
    w(f"  network weights (setup + eval0, uploaded lazily by the first evaluation): {len(net)} copies, "
      f"{sum(u[3] for u in net)} bytes ({sum(u[3] for u in net) / 1048576:.2f} MiB), ordered digest "
      f"{fnv(''.join(f'{u[3]}:{u[4]}' + chr(10) for u in net)):016x}")
    asyncs = [u for u in uploads if u[1] == "hipMemcpyAsync"]
    w(f"  asynchronous host-to-device copies: {len(asyncs)}")
    for u in asyncs[:12]:
        w(f"    {u[5]}: {u[2]} <- {u[6]} {u[3]} bytes fnv {u[4]}")
    d2h = [(r[0], r[2]) for r in records if r[1] == "hipMemcpy" and r[2].get("kind") in ("D2H", "DEFAULT") and "dst_fnv" in r[2]]
    if d2h:
        w("")
        w(f"== synchronous device-to-host copies: {len(d2h)} (phase src bytes dst_fnv) ==")
        for seq, f in d2h[:40]:
            w(f"  {phase(seq)} {f['src']} {f['bytes']} {f['dst_fnv']}")
    deeps = [(l, phase(l.seq)) for l in launches if l.deep and l.kernel == "c32_post_merge_head_half"]
    if deeps:
        w("")
        w("== network output: c32_post_merge_head_half argument 7 after the launch (bytes:fnv) ==")
        for l, ph in deeps:
            w(f"  {ph} n={l.n} out={l.args[7]} {l.deep.get('a7', '?')}")
    open(name + ".summary.txt", "w").write("\n".join(out) + "\n")
    open(name + ".evaldiff.txt", "w").write("\n".join(diff_lines) + "\n")
    with open(name + ".allocs.tsv", "w") as f:
        f.write("id\tbytes\tphase\tuploads\tfreed\tfirst_use\n")
        for ident, a in allocs.items():
            f.write(f"{ident}\t{a['bytes']}\t{a['phase']}\t{a['uploads']}\t{frees.get(ident, 'never')}\t"
                    f"{first_use.get(ident, '')}\n")
    with open(name + ".uploads.tsv", "w") as f:
        f.write("order\tcall\tdst\tbytes\tsrc_fnv\tphase\tsrc\tfirst_use\n")
        for u in uploads:
            f.write("\t".join(str(x) for x in u) + "\t" + first_use.get(u[2].split("+")[0], "") + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False, allow_abbrev=False,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("traces", nargs="+", metavar="TRACE", help="trace to summarise (required; no default)")
    parser.add_argument("-h", "--help", action="help", help="show this help and exit (default: off)")
    for path in parser.parse_args().traces:
        analyze(path)
        print("analyzed", path)


if __name__ == "__main__":
    main()
