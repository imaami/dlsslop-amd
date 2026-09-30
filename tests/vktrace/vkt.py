"""Read traces of the tracing Vulkan layer and replay them into frames and canonical command streams.

A trace line is "SEQ tTHREAD CALL key=value ...", with every handle renamed by kind
and creation order (buf12, img3, pipe40). Replaying the log tracks what every name
was created as, what memory it is bound to, what each descriptor set holds, and the
commands each command buffer recorded, so each queue submission can be expanded
into the commands it ran.

Canonical form (canonical()): a submission's commands with
- dispatches written with their effective state (the bound pipeline's definition,
  every bound set's contents at bind time, the push constant bytes the pipeline
  layout covers), and the bind/push commands themselves dropped;
- every resource renamed by first use in that submission (B0, I1, V2, M3, ...),
  its description (size and usage, format and extent, ...) given at first use;
- pipelines, layouts and samplers named by what they are, not by creation order.
Two implementations that record the same work therefore produce the same lines,
whatever order they created things in and however their handles are numbered.
"""
import hashlib
import re
from collections import defaultdict

NAME = re.compile(r"\b(inst|pd|dev|q|mem|buf|img|view|bview|smp|shm|pc|dsl|pl|pipe|dp|ds|cp|cb|fence|sem|qp|ev|sc)(\d+)\b")
RESOURCE_KINDS = {"buf": "B", "img": "I", "mem": "M", "qp": "Q", "bview": "T", "sem": "S", "fence": "F", "sc": "C"}


def short(text, n=10):
    return hashlib.sha1(text.encode()).hexdigest()[:n]


class Event:
    __slots__ = ("seq", "tid", "call", "f", "line", "index")

    def __init__(self, seq, tid, call, f, line, index):
        self.seq, self.tid, self.call, self.f, self.line, self.index = seq, tid, call, f, line, index


def parse(path):
    """(events, header) of a trace: its events in order, and the fields of its first line,
    "# vktrace 1 pid=... hash=... hash_submits=... spirv=..."; other '#' lines are skipped."""
    out, header = [], {}
    with open(path, errors="replace") as fh:
        for line in fh:
            if line.startswith("# vktrace ") and not header:
                header = dict(tok.partition("=")[::2] for tok in line.split()[3:] if "=" in tok)
                continue
            if not line or line[0] == "#" or line == "\n":
                continue
            parts = line.rstrip("\n").split(" ")
            if len(parts) < 3 or not parts[0].isdigit():
                continue
            f = {}
            for tok in parts[3:]:
                k, sep, v = tok.partition("=")
                if sep:
                    f[k] = v
            out.append(Event(int(parts[0]), parts[1], parts[2], f, line.rstrip("\n"), len(out)))
    return out, header


def split_list(v):
    """The items of "[a,b,c]"."""
    v = v.strip()
    if v.startswith("[") and v.endswith("]"):
        v = v[1:-1]
    return [x for x in v.split(",") if x] if v else []


class Cmd:
    __slots__ = ("call", "f", "event", "state")

    def __init__(self, call, f, event, state=None):
        self.call, self.f, self.event, self.state = call, f, event, state


class Submit:
    """One batch of one queue submission."""

    def __init__(self, number, queue, cbs, event):
        self.number = number          # the device's submission ordinal (sub=)
        self.queue = queue
        self.cbs = cbs
        self.event = event
        self.cmds = []                # Cmd, in execution order
        self.uploads = []             # Event
        self.readbacks = []           # Event
        self.hashes = []              # Event
        self.before = []              # events between the previous submission and this one
        self.dispatches = 0

        self.one_shot = False         # its command pool lived for this submission alone

    @property
    def frame(self):
        """A submission that dispatches and is not a build's one-shot (a command pool made
        after the previous submission and destroyed before the next)."""
        return self.dispatches > 0 and not self.one_shot


class Trace:
    def __init__(self, path):
        self.path = path
        self.events, self.header = parse(path)
        self.objs = {}                # name -> creating Event
        self.destroyed = {}           # buf/img name -> index of the Event that destroyed it
        self.bound = {}               # buf/img name -> (mem, offset)
        self.sets = {}                # ds -> {(binding, elem): desc string}
        self.recording = {}           # cb -> [Cmd]
        self.submits = []
        self.devices = []
        self.cb_pool = {}             # cb -> cp
        self.pool_life = {}           # cp -> [created, destroyed]
        self._replay()
        for i, sub in enumerate(self.submits):
            before = self.submits[i - 1].event.index if i else -1
            after = self.submits[i + 1].event.index if i + 1 < len(self.submits) else 1 << 62
            lives = [self.pool_life.get(self.cb_pool.get(cb)) for cb in sub.cbs]
            sub.one_shot = bool(lives) and all(l and l[0] > before and l[1] is not None and l[1] < after
                                               for l in lives)

    # -- replay ---------------------------------------------------------
    def _replay(self):
        rec_state = {}
        pending = []
        by_sub = {}
        for e in self.events:
            c, f = e.call, e.f
            for key in ("buf", "img", "view", "smp", "shm", "pc", "dsl", "pl", "pipe", "dp", "mem", "qp", "fence",
                        "sem", "cp", "sc", "dev"):
                v = f.get(key)
                if v and c.startswith(("Create", "Allocate")) and NAME.fullmatch(v) and v not in self.objs:
                    self.objs[v] = e
            if c == "CreateDevice" and "dev" in f:
                self.devices.append(e)
            elif c in ("DestroyBuffer", "DestroyImage"):
                self.destroyed[f.get("buf") or f.get("img")] = e.index
            if c == "AllocateDescriptorSets":
                for item in split_list(f.get("sets", "")):
                    s = item.partition(":")[0]
                    self.sets[s] = {}
                    self.objs[s] = e
            elif c == "AllocateCommandBuffers":
                for cb in split_list(f.get("cbs", "")):
                    self.objs[cb] = e
                    self.cb_pool[cb] = f.get("cp")
            elif c == "CreateCommandPool" and "cp" in f:
                self.pool_life[f["cp"]] = [e.index, None]
            elif c == "DestroyCommandPool" and f.get("cp") in self.pool_life:
                self.pool_life[f["cp"]][1] = e.index
            elif c in ("BindBufferMemory", "BindBufferMemory2"):
                self.bound[f["buf"]] = (f["mem"], int(f["offset"]))
            elif c in ("BindImageMemory", "BindImageMemory2"):
                self.bound[f["img"]] = (f["mem"], int(f["offset"]))
            elif c == "GetSwapchainImagesKHR":
                for img in split_list(f.get("imgs", "")):
                    self.objs.setdefault(img, e)
            elif c == "DescriptorWrite":
                what = " ".join(f"{k}={v}" for k, v in f.items() if k not in ("set", "binding", "elem"))
                self.sets.setdefault(f["set"], {})[(int(f["binding"]), int(f["elem"]))] = what
            elif c == "DescriptorCopy":
                # One line per descriptor copied.
                src_set, sb, se = f["src"].split(":")
                dst_set, db, de = f["dst"].split(":")
                val = self.sets.get(src_set, {}).get((int(sb), int(se)))
                if val is not None:
                    self.sets.setdefault(dst_set, {})[(int(db), int(de))] = val
            elif c in ("BeginCommandBuffer", "ResetCommandBuffer"):
                self.recording[f["cb"]] = []
                rec_state[f["cb"]] = {"pipe": None, "sets": {}, "push": bytearray(256), "pushed": bytearray(256)}
            elif c.startswith("Cmd") and "cb" in f:
                cb = f["cb"]
                st = rec_state.setdefault(cb, {"pipe": None, "sets": {}, "push": bytearray(256),
                                               "pushed": bytearray(256)})
                cmd = Cmd(c, f, e)
                if c == "CmdBindPipeline" and f.get("bind") == "COMPUTE":
                    st["pipe"] = f["pipe"]
                elif c == "CmdBindDescriptorSets" and f.get("bind") == "COMPUTE":
                    first = int(f["first"])
                    for i, s in enumerate(split_list(f["sets"])):
                        st["sets"][first + i] = (s, dict(self.sets.get(s, {})))
                elif c == "CmdPushConstants":
                    off, size = int(f["offset"]), int(f["size"])
                    data = bytes.fromhex(f["data"])
                    st["push"][off:off + size] = data
                    st["pushed"][off:off + size] = b"\1" * size
                elif c.startswith("CmdDispatch"):
                    cmd.state = {"pipe": st["pipe"], "sets": dict(st["sets"]), "push": bytes(st["push"]),
                                 "pushed": bytes(st["pushed"])}
                self.recording.setdefault(cb, []).append(cmd)
            elif c in ("QueueSubmit", "QueueSubmit2") and "sub" in f:
                s = Submit(int(f["sub"]), f["q"], split_list(f["cbs"]), e)
                for cb in s.cbs:
                    s.cmds.extend(self.recording.get(cb, []))
                s.dispatches = sum(1 for x in s.cmds if x.call.startswith("CmdDispatch"))
                s.before = pending
                pending = []
                self.submits.append(s)
                by_sub[(f["q"], s.number)] = s
                by_sub[s.number] = s
            elif c in ("Upload", "Readback", "Hash"):
                s = by_sub.get(int(f["sub"]))
                if s is not None:
                    {"Upload": s.uploads, "Readback": s.readbacks, "Hash": s.hashes}[c].append(e)
            else:
                pending.append(e)

    # -- descriptions -----------------------------------------------------
    def layout_sig(self, pl):
        e = self.objs.get(pl)
        if not e:
            return f"?{pl}"
        sets = []
        for dsl in split_list(e.f.get("sets", "")):
            d = self.objs.get(dsl)
            sets.append(d.f.get("bindings", "?") if d else "?")
        return f"sets=[{';'.join(sets)}] push={e.f.get('push', '[]')}"

    def pipeline_def(self, pipe):
        """What a pipeline is: its shader, entry, specialization and layout."""
        e = self.objs.get(pipe)
        if not e:
            return f"?{pipe}", "?"
        f = e.f
        if f.get("module", "") == "inline":
            shader_file, shader_fnv = f.get("file", "?"), f.get("fnv", "?")
        else:
            m = self.objs.get(f.get("module", ""))
            shader_file = m.f.get("file", "?") if m else "?"
            shader_fnv = m.f.get("fnv", "?") if m else "?"
        text = (f"shader={shader_fnv} entry={f.get('entry')} stage={f.get('stage')} flags={f.get('flags')} "
                f"stageflags={f.get('stageflags')} subgroup={f.get('subgroup', '-')} spec={f.get('spec')} "
                f"specdata={f.get('specdata', '')} layout={{{self.layout_sig(f.get('layout', ''))}}}")
        return text, shader_file

    def pipeline_token(self, pipe):
        text, shader_file = self.pipeline_def(pipe)
        label = shader_file if shader_file != "?" else "fnv" + text.split()[0].split("=")[1]
        return f"{label}#{short(text)}"

    def push_ranges(self, pipe):
        e = self.objs.get(pipe)
        pl = self.objs.get(e.f.get("layout", "")) if e else None
        out = []
        if pl:
            for r in split_list(pl.f.get("push", "")):
                _, _, span = r.partition(":")
                off, _, size = span.partition("+")
                out.append((int(off), int(size)))
        return out

    def describe(self, name):
        """A resource's description, for its first use in a canonical stream."""
        e = self.objs.get(name)
        if not e:
            return "unknown"
        f = e.f
        kind = NAME.fullmatch(name).group(1)
        if kind == "buf":
            return f"size={f.get('size')} usage={f.get('usage')} flags={f.get('flags')}" + (
                f" external={f['external']}" if "external" in f else "")
        if kind == "img":
            if e.call == "GetSwapchainImagesKHR":
                return "swapchain"
            return (f"{f.get('type')} {f.get('format')} {f.get('extent')} mips={f.get('mips')} layers={f.get('layers')} "
                    f"tiling={f.get('tiling')} usage={f.get('usage')} flags={f.get('flags')}")
        if kind == "mem":
            return f"size={f.get('size')} flags={f.get('flags')}"
        if kind == "qp":
            return f"type={f.get('type')} count={f.get('count')}"
        return e.call


class Canon:
    """Canonical names within one stream: resources by first use, the rest by content."""

    def __init__(self, trace, by="buffer"):
        self.t = trace
        self.by = by
        self.ids = {}
        self.counts = defaultdict(int)
        self.sigs = {}

    def resource(self, name):
        if name in self.ids:
            return self.ids[name]
        kind = NAME.fullmatch(name).group(1)
        if self.by == "memory" and kind in ("buf", "img") and name in self.t.bound:
            mem, off = self.t.bound[name]
            token = f"{self.resource(mem)}+{off}/{kind[0]}"
            self.ids[name] = token
            self.sigs[token] = self.t.describe(name)
            return token
        letter = RESOURCE_KINDS.get(kind, kind)
        token = f"{letter}{self.counts[letter]}"
        self.counts[letter] += 1
        self.ids[name] = token
        self.sigs[token] = self.t.describe(name)
        return token

    def view(self, name):
        e = self.t.objs.get(name)
        if not e:
            return f"?{name}"
        f = e.f
        return (f"view({self.resource(f['img'])},{f.get('type')},{f.get('format')},{f.get('range')},"
                f"{f.get('swizzle')})")

    def sampler(self, name):
        e = self.t.objs.get(name)
        if not e:
            return "null" if name == "null" else f"?{name}"
        text = " ".join(f"{k}={v}" for k, v in e.f.items() if k not in ("smp", "rc"))
        return f"smp#{short(text, 8)}"

    def name(self, match):
        kind, full = match.group(1), match.group(0)
        if kind in RESOURCE_KINDS:
            return self.resource(full)
        if kind == "view":
            return self.view(full)
        if kind == "smp":
            return self.sampler(full)
        if kind == "pipe":
            return self.t.pipeline_token(full)
        if kind == "pl":
            return "layout#" + short(self.t.layout_sig(full), 8)
        if kind == "ds":
            return "set"
        return kind  # cb, q and the like carry no meaning across runs

    def subst(self, text):
        return NAME.sub(self.name, text)

    def set_contents(self, contents):
        items = []
        for (b, el), what in sorted(contents.items()):
            items.append(f"b{b}.{el}:{self.subst(what)}")
        return "{" + ";".join(items) + "}"

    def dispatch(self, cmd):
        st = cmd.state
        pipe = st["pipe"]
        grid = " ".join(f"{k}={cmd.f[k]}" for k in ("base", "x", "y", "z", "buf", "offset") if k in cmd.f)
        sets = ",".join(f"{i}:{self.set_contents(c)}" for i, (s, c) in sorted(st["sets"].items()))
        push = []
        for off, size in self.t.push_ranges(pipe) if pipe else []:
            data = st["push"][off:off + size]
            unset = sum(1 for b in st["pushed"][off:off + size] if not b)
            push.append(data.hex() + (f"(unset{unset})" if unset else ""))
        pipe_token = self.t.pipeline_token(pipe) if pipe else "none"
        return f"{cmd.call} pipe={pipe_token} {grid} sets=[{sets}] push=[{','.join(push)}]"

    def command(self, cmd, literal=False):
        if cmd.call.startswith("CmdDispatch") and cmd.state is not None:
            return self.dispatch(cmd)
        fields = " ".join(f"{k}={v}" for k, v in cmd.f.items() if k not in ("cb", "i", "n"))
        return f"{cmd.call} {self.subst(fields)}"


BIND_CALLS = ("CmdBindPipeline", "CmdBindDescriptorSets", "CmdPushConstants")


def canonical(trace, submit, by="buffer", literal=False, canon=None):
    """(lines, signatures) of one submission."""
    c = canon or Canon(trace, by)
    lines = []
    for cmd in submit.cmds:
        if not literal and cmd.call in BIND_CALLS:
            continue
        lines.append(c.command(cmd, literal))
    return lines, c.sigs


def frames(trace):
    return [s for s in trace.submits if s.frame]

