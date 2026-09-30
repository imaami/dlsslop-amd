#!/usr/bin/python3
"""Whether trace B's device work matches trace A's.

Per frame (a submission that dispatches and is not a build's one-shot), the
canonical command streams of vkt.py must be equal: every dispatch with its
pipeline's definition (shader, entry, specialization, layout), the contents of
the sets it binds and its push constant bytes; every barrier, copy, fill,
update, clear, query and timestamp; resources renamed by first use in the frame
and described at first use (size, usage, format, extent). Then the frame's host
data: the input it uploaded and the answer it read back (FNVs).

Content hashes (VKTRACE_HASH) are compared per selector kind (i2b, copydst,
dispatch, all, buf, img) as multisets of (bytes, FNV), so resources may be
created, named and hashed in another order. They are compared in each frame,
and in the state after setup: for every region hashed in a submission before
the first frame that is not a frame itself, its last hash there, when its
resource is still alive at the first frame. A kind that both traces' headers
select must be hashed in both or in neither.

Before the frames, the setup: the pipelines the frames use, and the upload
payloads (bytes and FNV) in order, or as a multiset.

Prints MATCH, or each difference, the first one per frame with context, then
DIFFERENT with exit status 1.
"""
import argparse
import re
import sys
from collections import Counter

import vkt

QUERY_CALLS = ("CmdResetQueryPool", "CmdWriteTimestamp", "CmdBeginQuery", "CmdEndQuery")


def network_dispatch(trace, cmd):
    """A dispatch of a pipeline whose shader is one of the network's SPIR-V files."""
    return (cmd.call.startswith("CmdDispatch") and cmd.state and cmd.state["pipe"]
            and trace.pipeline_def(cmd.state["pipe"])[1] != "?")


def frame_view(trace, submit, args):
    if args.span == "network":
        # From the frame's first network dispatch to its last: what the runtime recorded,
        # without the host's transport or a layer's composition around it.
        cmds = submit.cmds
        idx = [i for i, c in enumerate(cmds) if network_dispatch(trace, c)]
        view = vkt.Submit(submit.number, submit.queue, submit.cbs, submit.event)
        view.cmds = cmds[idx[0]:idx[-1] + 1] if idx else []
        view.dispatches = submit.dispatches
        submit = view
    lines, sigs = vkt.canonical(trace, submit, by=args.by, literal=args.literal)
    if args.ignore_queries:
        # The runtime's timing ring and a host's own timestamps: no effect on the frame's data.
        lines = [l for l in lines if not l.startswith(QUERY_CALLS)]
    if args.span == "dispatch" and lines:
        idx = [i for i, l in enumerate(lines) if l.startswith("CmdDispatch")]
        if idx:
            lines = lines[idx[0]:idx[-1] + 1]
    return lines, sigs


def answer_hash(trace, submit):
    """The FNV of the network's answer: the first image-to-buffer copy after the frame's last
    network dispatch, read back from host memory or hashed by VKTRACE_HASH=i2b."""
    idx = [i for i, c in enumerate(submit.cmds) if network_dispatch(trace, c)]
    if not idx:
        return None
    copy = next((c for c in submit.cmds[idx[-1] + 1:] if c.call == "CmdCopyImageToBuffer"), None)
    if copy is None:
        return None
    label = f"{copy.f['cb']}#{copy.f['i']}.0"
    for r in submit.readbacks:
        if r.f.get("at") == label:
            return f"{r.f.get('bytes')}:{r.f.get('fnv')}"
    for h in submit.hashes:
        if f"i2b:{label}" in h.f.get("sel", "").split("|"):
            return f"{h.f.get('bytes')}:{h.f.get('fnv', 'skip=' + h.f.get('skip', '?'))}"
    return "unhashed"


def explain(a, b):
    """Which parts of two differing canonical lines differ; for push constants, which words."""
    if not (a.startswith("CmdDispatch") and b.startswith("CmdDispatch")):
        return []

    def parts(line):
        head, _, rest = line.partition(" sets=")
        sets, _, push = rest.rpartition(" push=")
        return {"head": head, "sets": sets, "push": push}
    pa, pb = parts(a), parts(b)
    out = [f"differs: {', '.join(k for k in pa if pa[k] != pb[k])}"]
    if pa["push"] != pb["push"]:
        xa, xb = pa["push"].strip("[]").split(","), pb["push"].strip("[]").split(",")
        for r, (u, v) in enumerate(zip(xa, xb)):
            words = [i for i in range(0, max(len(u), len(v)), 8) if u[i:i + 8] != v[i:i + 8]]
            if words:
                out.append(f"push range {r}: words " + ", ".join(
                    f"{i // 8}: {u[i:i + 8]} vs {v[i:i + 8]}" for i in words[:12]))
    if pa["sets"] != pb["sets"]:
        ia, ib = pa["sets"].split(";"), pb["sets"].split(";")
        for u, v in zip(ia, ib):
            if u != v:
                out.append(f"binding: {u} vs {v}")
                break
    return out


def frames_of(trace, args):
    fs = vkt.frames(trace)
    if args.span == "network":
        fs = [s for s in fs if any(network_dispatch(trace, c) for c in s.cmds)]
    return fs


def host(events):
    return [(e.f.get("bytes"), e.f.get("fnv")) for e in events]


def selected_kinds(selectors):
    """The content hash kinds that a VKTRACE_HASH value selects."""
    kinds = set()
    for item in selectors.split(","):
        if item in ("i2b", "copydst", "all"):
            kinds.add(item)
        elif item == "storage" or re.fullmatch(r"dispatch=\d+", item):
            kinds.add("dispatch")
        elif re.fullmatch(r"(buf|img)\d+", item):
            kinds.add(item[:3])
    return kinds


def kinds(event):
    """The kinds of a Hash line: a region that several selectors named was hashed once and
    counts under each of their kinds."""
    return {re.sub(r"\d+$", "", label.partition(":")[0]) for label in event.f.get("sel", "").split("|")}


def value(event):
    return event.f.get("bytes", "-"), event.f.get("fnv") or "skip=" + event.f.get("skip", "?")


def content(submit):
    """A submission's content hashes: {kind: Counter of (bytes, FNV or skip=REASON)}."""
    out = {}
    for e in submit.hashes:
        for kind in kinds(e):
            out.setdefault(kind, Counter())[value(e)] += 1
    return out


def compare_content(what, a, b, both, tally):
    """Prints how the content hashes of A and B differ, per kind; returns 1 if they do. A kind
    that only one of them holds differs when it is in BOTH, the kinds both traces select.
    TALLY counts, per kind, what was compared and what only one trace hashed."""
    bad = 0
    for kind in sorted(set(a) | set(b)):
        if kind not in a or kind not in b:
            side = "A" if kind in a else "B"
            if kind in both:
                bad = 1
                print(f"{what}: content hashes ({kind}) only in {side}, though both traces select {kind}")
            tally[f"{kind} only in {side}"] += 1
            continue
        tally[f"{kind} compared"] += 1
        if a[kind] == b[kind]:
            continue
        bad = 1
        only_a, only_b = a[kind] - b[kind], b[kind] - a[kind]
        print(f"{what}: content hashes ({kind}) differ: {sum(a[kind].values())} vs {sum(b[kind].values())}, "
              f"{sum(only_a.values())} only in A, {sum(only_b.values())} only in B")
        for side, only in (("A", only_a), ("B", only_b)):
            for (n, h), k in list(only.items())[:5]:
                print(f"  only {side}: {k}x bytes={n} {h}")
    return bad


def setup_state(trace, frames):
    """The content hashes of the state after setup, as content() gives them, or None without
    frames: for each region hashed in a submission before the first frame that is not a frame
    itself, its last hash there, unless its resource is destroyed before the first frame. A
    build's staging buffers therefore do not count, however the build is split."""
    if not frames:
        return None
    first = frames[0].event.index
    last = {}
    for s in trace.submits:
        if s.event.index >= first:
            break
        if s.frame:
            continue
        for e in s.hashes:
            res = e.f.get("res", "")
            resource = vkt.NAME.match(res)
            if resource and trace.destroyed.get(resource.group(0), first) < first:
                continue
            for kind in kinds(e):
                last[(kind, res)] = value(e)
    out = {}
    for (kind, _), v in last.items():
        out.setdefault(kind, Counter())[v] += 1
    return out


def pairs_of(spec):
    """The frame index pairs of --frames: A:B items, each side an index or a range I-J."""
    pairs = []
    for item in spec.split(","):
        x, _, y = item.partition(":")
        xa = list(range(int(x.split("-")[0]), int(x.split("-")[-1]) + 1))
        ya = list(range(int(y.split("-")[0]), int(y.split("-")[-1]) + 1))
        pairs += list(zip(xa, ya))
    return pairs


def main():
    p = argparse.ArgumentParser(description=__doc__, add_help=False, allow_abbrev=False,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("a", metavar="A", help="the reference trace (required; no default)")
    p.add_argument("b", metavar="B", help="the trace to check (required; no default)")
    p.add_argument("-y", "--by", choices=("buffer", "memory"), default="buffer",
                   help="rename resources as buffers and images, or as allocation + offset (default: buffer)")
    p.add_argument("-l", "--literal", action="store_true",
                   help="compare bind and push commands as recorded too (default: off)")
    p.add_argument("-s", "--span", choices=("all", "dispatch", "network"), default="all",
                   help="compare whole frames; each frame from its first dispatch to its last; or only the frames "
                        "that run the network, from its first dispatch to its last, and their answers: a network "
                        "dispatch's shader is a file in the traced VKTRACE_SPIRV (default: all)")
    p.add_argument("-f", "--frames", metavar="PAIRS",
                   help="comma-separated A:B frame index pairs or ranges to compare, such as 1:0 or 2-5:0-3 "
                        "(default: every frame of A with the frame of B at the same index)")
    p.add_argument("-S", "--skip-setup", action="store_true",
                   help="compare the frames only, not the setup (default: off)")
    p.add_argument("-q", "--ignore-queries", action="store_true",
                   help="leave query pool resets, timestamps and queries out of the frames (default: off)")
    p.add_argument("-n", "--no-host", action="store_true",
                   help="ignore the frames' uploads, readbacks and content hashes (default: off)")
    p.add_argument("-c", "--context", type=int, default=2, metavar="LINES",
                   help="lines of context before a difference (default: 2)")
    p.add_argument("-h", "--help", action="help", help="show this help and exit (default: off)")
    args = p.parse_args()
    if args.frames and not re.fullmatch(r"\d+(-\d+)?:\d+(-\d+)?(,\d+(-\d+)?:\d+(-\d+)?)*", args.frames):
        p.error("--frames takes A:B pairs such as 1:0 or 2-5:0-3, separated by commas")

    A, B = vkt.Trace(args.a), vkt.Trace(args.b)
    fa, fb = frames_of(A, args), frames_of(B, args)
    both = selected_kinds(A.header.get("hash", "")) & selected_kinds(B.header.get("hash", ""))
    bad = 0
    tally = Counter()

    if not args.skip_setup:
        # Pipelines the frames use, as a set of definitions.
        def used(trace, fs):
            out = Counter()
            for s in fs:
                for c in s.cmds:
                    if c.call.startswith("CmdDispatch") and c.state and c.state["pipe"]:
                        out[trace.pipeline_def(c.state["pipe"])] += 1
            return out
        ua, ub = used(A, fa), used(B, fb)
        if set(ua) != set(ub):
            bad = 1
            only_a = sorted(set(ua) - set(ub), key=lambda d: d[1])
            only_b = sorted(set(ub) - set(ua), key=lambda d: d[1])
            print(f"pipelines used by frames differ: {len(ua)} vs {len(ub)} definitions; "
                  f"{len(only_a)} only in A, {len(only_b)} only in B")
            for d in only_a[:5]:
                print(f"  only A: {d[1]} {d[0][:240]}")
            for d in only_b[:5]:
                print(f"  only B: {d[1]} {d[0][:240]}")
        # Uploads outside frames.
        sa = [u for s in A.submits if not s.frame for u in host(s.uploads)]
        sb = [u for s in B.submits if not s.frame for u in host(s.uploads)]
        if sa != sb:
            if Counter(sa) == Counter(sb):
                print(f"setup uploads: the same {len(sa)} payloads in a different order")
            else:
                bad = 1
                ca, cb = Counter(sa), Counter(sb)
                print(f"setup uploads differ: {len(sa)} vs {len(sb)} copies, "
                      f"{sum((ca - cb).values())} only in A, {sum((cb - ca).values())} only in B")
                for (n, h), k in list((ca - cb).items())[:5]:
                    print(f"  only A: {k}x bytes={n} fnv={h}")
                for (n, h), k in list((cb - ca).items())[:5]:
                    print(f"  only B: {k}x bytes={n} fnv={h}")
        else:
            print(f"setup uploads: {len(sa)} identical")
        xa, xb = setup_state(A, fa), setup_state(B, fb)
        if xa is not None and xb is not None:
            bad |= compare_content("state after setup", xa, xb, both, tally)

    if args.frames:
        pairs = pairs_of(args.frames)
    else:
        if len(fa) != len(fb):
            bad = 1
            print(f"frames: {len(fa)} vs {len(fb)}")
        pairs = [(i, i) for i in range(min(len(fa), len(fb)))]

    compared = 0
    for i, j in pairs:
        if i >= len(fa) or j >= len(fb):
            bad = 1
            print(f"frame {i}:{j}: missing ({len(fa)} vs {len(fb)} frames)")
            continue
        sa_, sb_ = fa[i], fb[j]
        la, siga = frame_view(A, sa_, args)
        lb, sigb = frame_view(B, sb_, args)
        compared += 1
        if la != lb:
            bad = 1
            first = next((k for k, (u, v) in enumerate(zip(la, lb)) if u != v), min(len(la), len(lb)))
            print(f"frame {i}:{j} (submit {sa_.number}:{sb_.number}): {len(la)} vs {len(lb)} commands; "
                  f"first difference at command {first}:")
            for k in range(max(0, first - args.context), first):
                print(f"  = {la[k][:400]}")
            print(f"  A {la[first][:1200] if first < len(la) else '-'}")
            print(f"  B {lb[first][:1200] if first < len(lb) else '-'}")
            if first < len(la) and first < len(lb):
                for note in explain(la[first], lb[first]):
                    print(f"  {note}")
            continue
        diff_sigs = {k: (siga.get(k), sigb.get(k)) for k in set(siga) | set(sigb) if siga.get(k) != sigb.get(k)}
        if diff_sigs:
            bad = 1
            print(f"frame {i}:{j}: same commands, resources differ:")
            for k, (x, y) in sorted(diff_sigs.items())[:10]:
                print(f"  {k}: {x} vs {y}")
        if args.span == "network":
            xa, xb = answer_hash(A, sa_), answer_hash(B, sb_)
            if xa != xb:
                bad = 1
                print(f"frame {i}:{j}: answer {xa} vs {xb}")
        elif not args.no_host:
            for what, xa, xb in (("uploads", host(sa_.uploads), host(sb_.uploads)),
                                 ("readbacks", host(sa_.readbacks), host(sb_.readbacks))):
                if xa != xb:
                    bad = 1
                    print(f"frame {i}:{j}: {what} {xa} vs {xb}")
            bad |= compare_content(f"frame {i}:{j}", content(sa_), content(sb_), both, tally)
    for what, n in sorted(tally.items()):
        print(f"content hashes: {what}: {n}")
    print(f"compared {compared} frames")
    print("MATCH" if not bad else "DIFFERENT")
    return bad


if __name__ == "__main__":
    sys.exit(main())
