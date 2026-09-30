#!/usr/bin/python3
"""Summarise traces of the tracing Vulkan layer.

Writes, next to each NAME.trace:
  NAME.summary.txt   device, features, memory types, builds, frames, output hashes
  NAME.frames.tsv    per frame: submit, dispatches, pipelines, barriers, copies, push
                     constants, descriptor writes and objects created since the last
                     submission, input and output FNVs
  NAME.frame.txt     the canonical command stream (vkt.canonical) of the first and last
                     frame of each build, with resource descriptions
  NAME.framediff.txt what changes between consecutive frames of one build, by
                     command (handles kept, so resource identity across frames shows)
  NAME.allocs.tsv    every allocation: size, type flags, what is bound to it, when it
                     was made and freed
  NAME.pipelines.tsv every compute pipeline: shader file, FNV, entry, subgroup size,
                     specialization, layout, and its dispatches in the last frame
  NAME.uploads.tsv   every copy out of mapped host memory, with its FNV and target
"""
import argparse
from collections import Counter, defaultdict

import vkt


def builds_of(trace):
    """Event indices at which a network build starts: each CreatePipelineCache."""
    marks = [e.index for e in trace.events if e.call == "CreatePipelineCache"]
    if not marks:
        marks = [next((e.index for e in trace.events if e.call == "CreateComputePipeline"), 0)]
    return marks


def build_of(marks, index):
    k = -1
    for i, m in enumerate(marks):
        if index >= m:
            k = i
    return k


def frame_stats(trace, s):
    calls = Counter(c.call for c in s.cmds)
    pipes = set()
    shaders = set()
    push_bytes = 0
    for c in s.cmds:
        if c.call.startswith("CmdDispatch") and c.state and c.state["pipe"]:
            pipes.add(trace.pipeline_token(c.state["pipe"]))
            shaders.add(trace.pipeline_def(c.state["pipe"])[1])
        if c.call == "CmdPushConstants":
            push_bytes += int(c.f["size"])
    writes = sum(1 for e in s.before if e.call == "DescriptorWrite")
    created = sum(1 for e in s.before if e.call.startswith(("Create", "Allocate")))
    return {
        "cmds": len(s.cmds),
        "dispatches": s.dispatches,
        "pipelines": len(pipes),
        "shaders": len(shaders),
        "barriers": calls["CmdPipelineBarrier"] + calls["CmdPipelineBarrier2"],
        "copies": sum(v for k, v in calls.items() if k.startswith(("CmdCopy", "CmdBlit"))),
        "fills": calls["CmdFillBuffer"] + calls["CmdUpdateBuffer"] + calls["CmdClearColorImage"],
        "binds": calls["CmdBindPipeline"] + calls["CmdBindDescriptorSets"],
        "push": calls["CmdPushConstants"],
        "push_bytes": push_bytes,
        "timestamps": calls["CmdWriteTimestamp"] + calls["CmdWriteTimestamp2"],
        "writes_before": writes,
        "created_before": created,
    }


def main(path):
    t = vkt.Trace(path)
    base = path[:-6] if path.endswith(".trace") else path
    marks = builds_of(t)
    fr = vkt.frames(t)
    out = []
    w = out.append
    w(f"trace {path}: {len(t.events)} events, {len(t.submits)} submissions, {len(fr)} frames, {len(marks)} builds")
    for d in t.devices:
        w(f"device {d.line.split(' ', 3)[3]}")
    for e in t.events:
        if e.call in ("DeviceFeatures", "DeviceChain"):
            w(f"  {e.call} {e.f.get('struct', '')} {e.f.get('on', e.f.get('other', ''))}")
    for e in t.events:
        if e.call == "MemoryType":
            w(f"  memory type {e.f['index']}: flags {e.f['flags']} heap {e.f['heap']} ({int(e.f['heapsize']) >> 20} MiB)")
    calls = Counter(e.call for e in t.events)
    w("calls: " + ", ".join(f"{k} {v}" for k, v in calls.most_common()))

    # Builds.
    w("")
    w("builds (each starts at a CreatePipelineCache):")
    bounds = marks + [len(t.events)]
    for k in range(len(marks)):
        lo, hi = bounds[k], bounds[k + 1]
        first_frame = next((i for i, s in enumerate(fr) if s.event.index > lo), None)
        if first_frame is not None:
            hi = min(hi, fr[first_frame].event.index)
        ev = t.events[lo:hi]
        c = Counter(e.call for e in ev)
        mem = [e for e in ev if e.call == "AllocateMemory" and e.f.get("rc") == "0"]
        ups = [u for s in t.submits if lo <= s.event.index < hi for u in s.uploads]
        subs = [s for s in t.submits if lo <= s.event.index < hi]
        shaders = {e.f.get("file") for e in ev if e.call == "CreateShaderModule"}
        # A build's own frame image comes just before its cache.
        w(f"  build {k}: events {lo}..{hi}, before frame {first_frame}: {c['CreateShaderModule']} shader modules "
          f"({len(shaders)} files), {c['CreateComputePipeline']} pipelines, {c['CreateDescriptorSetLayout']} set layouts, "
          f"{c['CreateDescriptorPool']} pools, {c['DescriptorWrite']} descriptor writes, {len(mem)} allocations "
          f"({sum(int(e.f['size']) for e in mem)} bytes), {c['CreateBuffer']} buffers, {c['CreateImage']} images, "
          f"{len(subs)} submissions, {len(ups)} uploads ({sum(int(u.f['bytes']) for u in ups)} bytes)")
        fills = [cmd for s in subs for cmd in s.cmds if cmd.call in ("CmdFillBuffer", "CmdClearColorImage", "CmdUpdateBuffer")]
        if fills:
            w(f"    fills in its submissions: {len(fills)}: " +
              "; ".join(f"{x.call} {x.f.get('buf', x.f.get('img'))} {x.f.get('offset', '')}+{x.f.get('size', '')}"
                        for x in fills[:8]))
        other = [s for s in subs if s.cmds and not s.uploads]
        if other:
            w(f"    submissions without uploads: " + "; ".join(
                f"sub {s.number}: " + ",".join(f"{a}x{b}" for a, b in Counter(x.call for x in s.cmds).items())
                for s in other[:8]))

    # Frames.
    w("")
    w("frames (submissions that dispatch):")
    rows = []
    header = ["frame", "submit", "build", "cmds", "dispatches", "pipelines", "shaders", "barriers", "copies", "fills",
              "binds", "push", "push_bytes", "timestamps", "writes_before", "created_before", "input_fnv", "output_fnv"]
    for i, s in enumerate(fr):
        st = frame_stats(t, s)
        ins = ",".join(f"{u.f['bytes']}:{u.f['fnv']}" for u in s.uploads) or "-"
        outs = ",".join(f"{r.f['bytes']}:{r.f['fnv']}" for r in s.readbacks) or "-"
        b = build_of(marks, s.event.index)
        rows.append([i, s.number, b] + [st[h] for h in header[3:16]] + [ins, outs])
        w(f"  frame {i} sub {s.number} build {b}: {st['cmds']} cmds, {st['dispatches']} dispatches, "
          f"{st['pipelines']} pipelines ({st['shaders']} shader files), {st['barriers']} barriers, {st['copies']} copies, "
          f"{st['fills']} fills, {st['push']} push constants ({st['push_bytes']} bytes), {st['timestamps']} timestamps; "
          f"since the last submission {st['writes_before']} descriptor writes, {st['created_before']} objects created; "
          f"in {ins} out {outs}")
        for h in s.hashes:
            w(f"    hash {h.f.get('sel')} {h.f.get('res')} {h.f.get('bytes', '')} {h.f.get('fnv', h.f.get('skip'))}")
    with open(base + ".frames.tsv", "w") as fh:
        fh.write("\t".join(header) + "\n")
        for r in rows:
            fh.write("\t".join(str(x) for x in r) + "\n")

    # Canonical streams and frame-to-frame changes.
    by_build = defaultdict(list)
    for i, s in enumerate(fr):
        by_build[build_of(marks, s.event.index)].append((i, s))
    with open(base + ".frame.txt", "w") as fh:
        for b, items in sorted(by_build.items()):
            for i, s in ([items[0], items[-1]] if len(items) > 1 else items):
                lines, sigs = vkt.canonical(t, s)
                fh.write(f"=== frame {i} (submission {s.number}, build {b}): {len(lines)} commands\n")
                for k, l in enumerate(lines):
                    fh.write(f"{k}\t{l}\n")
                fh.write("resources:\n")
                for k, v in sigs.items():
                    fh.write(f"  {k}\t{v}\n")
    w("")
    w("frame-to-frame (consecutive frames of one build, handles as traced):")
    with open(base + ".framediff.txt", "w") as fh:
        for b, items in sorted(by_build.items()):
            for (i, s), (j, u) in zip(items, items[1:]):
                canon = vkt.Canon(t)  # one naming for both frames: the same handle keeps its name
                la = [canon.command(c) for c in s.cmds]
                lb = [canon.command(c) for c in u.cmds]
                # Keep traced handle names: resources as themselves.
                ra = [f"{c.call} " + " ".join(f"{k}={v}" for k, v in c.f.items() if k not in ('cb', 'i')) for c in s.cmds]
                rb = [f"{c.call} " + " ".join(f"{k}={v}" for k, v in c.f.items() if k not in ('cb', 'i')) for c in u.cmds]
                diffs = [k for k in range(max(len(ra), len(rb)))
                         if k >= len(ra) or k >= len(rb) or ra[k] != rb[k] or la[k] != lb[k]]
                fh.write(f"=== frame {i} -> {j} (build {b}): {len(ra)} -> {len(rb)} commands, {len(diffs)} differ\n")
                for k in diffs[:200]:
                    fh.write(f"  {k}\n    - {ra[k] if k < len(ra) else '-'}\n    + {rb[k] if k < len(rb) else '-'}\n")
                between = Counter(e.call for e in u.before)
                ins_a = [x.f.get("fnv") for x in s.uploads]
                ins_b = [x.f.get("fnv") for x in u.uploads]
                w(f"  frame {i} -> {j}: {len(diffs)} of {len(rb)} commands differ "
                  f"({', '.join(sorted({(ra[k] if k < len(ra) else rb[k]).split()[0] for k in diffs})) or 'none'}); "
                  f"input {'same' if ins_a == ins_b else 'differs'}; between them: "
                  + (", ".join(f"{k} {v}" for k, v in between.most_common()) or "nothing"))

    # Allocations.
    bound_to = defaultdict(list)
    for e in t.events:
        if e.call.startswith("Bind") and "mem" in e.f:
            obj = e.f.get("buf") or e.f.get("img")
            bound_to[e.f["mem"]].append((obj, int(e.f["offset"])))
    freed = {e.f["mem"]: e.index for e in t.events if e.call == "FreeMemory"}
    frame_at = [s.event.index for s in fr]

    def phase(index):
        k = sum(1 for x in frame_at if x < index)
        return f"build{build_of(marks, index)}/before-frame{k}"

    total = Counter()
    with open(base + ".allocs.tsv", "w") as fh:
        fh.write("mem\tsize\ttype\tflags\textra\tbound\tmade\tfreed\n")
        for e in t.events:
            if e.call != "AllocateMemory" or e.f.get("rc") != "0":
                continue
            m = e.f["mem"]
            objs = []
            for obj, off in bound_to.get(m, []):
                o = t.objs.get(obj)
                objs.append(f"{obj}@{off}:{t.describe(obj).replace(' ', '/')}" if o else f"{obj}@{off}")
            extra = ",".join(f"{k}={v}" for k, v in e.f.items() if k in ("dedicated", "import", "export", "allocflags"))
            fh.write(f"{m}\t{e.f['size']}\t{e.f['type']}\t{e.f['flags']}\t{extra or '-'}\t{' '.join(objs) or '-'}\t"
                     f"{phase(e.index)}\t{phase(freed[m]) if m in freed else 'never'}\n")
            total[phase(e.index)] += int(e.f["size"])
    w("")
    w("allocations by phase (bytes): " + ", ".join(f"{k} {v}" for k, v in total.items()))
    live_at_last = 0
    if fr:
        last = fr[-1].event.index
        for e in t.events:
            if e.call == "AllocateMemory" and e.f.get("rc") == "0" and e.index < last and freed.get(e.f["mem"], 1 << 62) > last:
                live_at_last += int(e.f["size"])
    w(f"live device memory at the last frame: {live_at_last} bytes ({live_at_last / 2**20:.1f} MiB)")
    sizes = Counter()
    for e in t.events:
        if e.call == "CreateBuffer" and e.f.get("rc") == "0":
            sizes[("buffer", e.f["size"], e.f["usage"])] += 1
        if e.call == "CreateImage" and e.f.get("rc") == "0":
            sizes[("image", f"{e.f['format']} {e.f['extent']}", e.f["usage"])] += 1
    w("resources created (kind, size or format+extent, usage: count):")
    for (kind, what, usage), n in sorted(sizes.items(), key=lambda x: (x[0][0], -int(x[0][1]) if x[0][1].isdigit() else 0)):
        w(f"  {kind} {what} usage={usage}: {n}")

    # Pipelines.
    last_counts = Counter()
    if fr:
        for c in fr[-1].cmds:
            if c.call.startswith("CmdDispatch") and c.state and c.state["pipe"]:
                last_counts[c.state["pipe"]] += 1
    with open(base + ".pipelines.tsv", "w") as fh:
        fh.write("pipe\tshader\tfnv\tentry\tsubgroup\tspec\tspecdata\tlayout\tdispatches_last_frame\ttoken\n")
        for e in t.events:
            if e.call != "CreateComputePipeline":
                continue
            p = e.f["pipe"]
            text, shader = t.pipeline_def(p)
            fnv = text.split()[0].split("=")[1]
            fh.write(f"{p}\t{shader}\t{fnv}\t{e.f.get('entry')}\t{e.f.get('subgroup', '-')}\t{e.f.get('spec')}\t"
                     f"{e.f.get('specdata', '')}\t{t.layout_sig(e.f.get('layout', ''))}\t{last_counts.get(p, 0)}\t"
                     f"{t.pipeline_token(p)}\n")
    if fr:
        per_shader = Counter()
        for p, n in last_counts.items():
            per_shader[t.pipeline_def(p)[1]] += n
        w("")
        w(f"last frame's dispatches by shader file ({sum(per_shader.values())} dispatches, {len(last_counts)} pipelines):")
        w("  " + ", ".join(f"{k} {v}" for k, v in per_shader.most_common()))

    # Uploads.
    with open(base + ".uploads.tsv", "w") as fh:
        fh.write("submit\tframe\tat\tbytes\tfnv\ttarget\n")
        for s in t.submits:
            for u in s.uploads:
                cb, _, rest = u.f["at"].partition("#")
                idx = int(rest.split(".")[0])
                target = "-"
                for c in s.cmds:
                    if c.f.get("cb") == cb and c.f.get("i") == str(idx):
                        target = c.f.get("dst", "-")
                fh.write(f"{s.number}\t{int(s.frame)}\t{u.f['at']}\t{u.f['bytes']}\t{u.f['fnv']}\t{target}\n")
    setup_uploads = [u for s in t.submits if not s.frame for u in s.uploads]
    digest = 14695981039346656037
    for u in setup_uploads:
        for ch in f"{u.f['bytes']}:{u.f['fnv']}\n".encode():
            digest = ((digest ^ ch) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    w("")
    w(f"setup uploads: {len(setup_uploads)} copies, {sum(int(u.f['bytes']) for u in setup_uploads)} bytes, "
      f"ordered digest {digest:016x} (FNV-1a 64 of 'BYTES:FNV\\n' lines)")
    w("output hashes (readbacks of each frame, bytes:fnv):")
    for i, s in enumerate(fr):
        w(f"  frame {i}: " + (" ".join(f"{r.f['at'].split('#')[1]}:{r.f['bytes']}:{r.f['fnv']}" for r in s.readbacks) or "-"))
    with open(base + ".summary.txt", "w") as fh:
        fh.write("\n".join(out) + "\n")
    print(f"{path}: {len(fr)} frames, {len(marks)} builds -> {base}.summary.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, add_help=False, allow_abbrev=False,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("traces", nargs="+", metavar="TRACE", help="trace to summarise (required; no default)")
    parser.add_argument("-h", "--help", action="help", help="show this help and exit (default: off)")
    for path in parser.parse_args().traces:
        main(path)
