#!/usr/bin/env python3
"""Check the tracing HIP runtime and its tools without a GPU.

The runtime forwards to a fake runtime and loads a code object made here; its
trace must be exactly the expected one, and analyze.py, compare.py and
plan_hashes.py must read it. trace.sh and the shared client, shmclient, run a
stand-in dlsslopd, and the client must report the settings trace.sh gives it.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import re
import stat
import struct
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(description=__doc__, add_help=False, allow_abbrev=False,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('-h', '--help', action='help', help='show this help and exit (default: off)')
parser.add_argument('runtime', type=Path, help='the tracing runtime, libhiptrace.so (required; no default)')
parser.add_argument('fake', type=Path, help='the fake runtime it forwards to (required; no default)')
parser.add_argument('client', type=Path,
                    help='shmclient, in the same directory as the runtime (required; no default)')
args = parser.parse_args()
runtime, fake, client = args.runtime.resolve(), args.fake.resolve(), args.client.resolve()
assert client.parent == runtime.parent, 'trace.sh finds the runtime and the client in one build tree'


def fnv(data):
    h = 14695981039346656037
    for b in data:
        h = ((h ^ b) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def msgpack(value):
    """VALUE in MessagePack, as much of it as AMDGPU metadata uses."""
    if isinstance(value, dict):
        assert len(value) < 16
        return bytes([0x80 | len(value)]) + b''.join(msgpack(k) + msgpack(v) for k, v in value.items())
    if isinstance(value, list):
        assert len(value) < 16
        return bytes([0x90 | len(value)]) + b''.join(msgpack(item) for item in value)
    if isinstance(value, str):
        data = value.encode()
        return (bytes([0xa0 | len(data)]) if len(data) < 32 else bytes([0xd9, len(data)])) + data
    assert 0 <= value < 65536
    return bytes([value]) if value < 128 else b'\xcd' + struct.pack('>H', value)


def kernel(name, arguments):
    """A kernel's metadata; ARGUMENTS are (name, value kind, size) in order."""
    entries, offset = [], 0
    for argument, kind, size in arguments:
        entries.append({'.name': argument, '.offset': offset, '.size': size, '.value_kind': kind})
        offset += size
    return {'.name': name, '.symbol': name + '.kd', '.kernarg_segment_size': (offset + 7) & ~7,
            '.group_segment_fixed_size': 512, '.private_segment_fixed_size': 0, '.vgpr_count': 24,
            '.sgpr_count': 16, '.vgpr_spill_count': 0, '.sgpr_spill_count': 0, '.wavefront_size': 32,
            '.max_flat_workgroup_size': 256, '.args': entries}


def code_object(kernels):
    """An ELF64 whose one section is the NT_AMDGPU_METADATA note describing KERNELS."""
    desc = msgpack({'amdhsa.version': [1, 2], 'amdhsa.kernels': kernels})
    note = struct.pack('<III', 7, len(desc), 32) + b'AMDGPU\0\0' + desc + bytes(-len(desc) % 4)
    shoff = 64 + len(note)
    # ET_DYN, EM_AMDGPU; no program headers; section headers after the note.
    header = b'\x7fELF\x02\x01\x01' + bytes(9) + struct.pack('<HHIQQQIHHHHHH', 3, 224, 1, 0, 0, shoff, 0, 64, 0, 0,
                                                             64, 2, 0)
    sections = bytes(64) + struct.pack('<IIQQQQIIQQ', 0, 7, 0, 0, 64, len(note), 0, 0, 4, 0)
    return header + note + sections


def drive(trace, image, modules):
    """In this process: load the runtime and make the calls the expected trace lists."""
    os.environ.update(HIPTRACE_REAL=str(fake), HIPTRACE_FILE=str(trace), HIPTRACE_MODULES=str(modules),
                      HIPTRACE_DEEP='0-1', HIPTRACE_DEEP_KERNELS='tail')
    hip = ctypes.CDLL(str(runtime))
    P, U, size_t = ctypes.c_void_p, ctypes.c_uint, ctypes.c_size_t
    signatures = {
        'hipStreamCreate': [ctypes.POINTER(P)], 'hipEventCreate': [ctypes.POINTER(P)], 'hipEventRecord': [P, P],
        'hipEventElapsedTime': [ctypes.POINTER(ctypes.c_float), P, P], 'hipEventDestroy': [P],
        'hipGraphLaunch': [P, P],
        'hipModuleLoadData': [ctypes.POINTER(P), P], 'hipModuleGetFunction': [ctypes.POINTER(P), P, ctypes.c_char_p],
        'hipMalloc': [ctypes.POINTER(P), size_t], 'hipFree': [P], 'hipHostMalloc': [ctypes.POINTER(P), size_t, U],
        'hipHostFree': [P], 'hipHostRegister': [P, size_t, U], 'hipHostUnregister': [P],
        'hipMemsetAsync': [P, ctypes.c_int, size_t, P], 'hipMemcpy': [P, P, size_t, ctypes.c_int],
        'hipMemcpyAsync': [P, P, size_t, ctypes.c_int, P],
        'hipModuleLaunchKernel': [P] + [U] * 7 + [P, ctypes.POINTER(P), ctypes.POINTER(P)],
        'hipExtModuleLaunchKernel': [P] + [U] * 6 + [size_t, P, ctypes.POINTER(P), ctypes.POINTER(P), P, P, U],
    }
    for name, types in signatures.items():
        getattr(hip, name).argtypes = types

    def call(name, *arguments, expected=0):
        rc = getattr(hip, name)(*arguments)
        assert rc == expected, (name, rc)

    stream, module, probe, tail, gate = P(), P(), P(), P(), P()
    buffers, events = [P() for _ in range(4)], [P() for _ in range(4)]
    call('hipStreamCreate', ctypes.byref(stream))
    image_buffer = ctypes.create_string_buffer(image, len(image))
    call('hipModuleLoadData', ctypes.byref(module), ctypes.cast(image_buffer, P))
    call('hipModuleGetFunction', ctypes.byref(probe), module, b'probe')
    call('hipModuleGetFunction', ctypes.byref(tail), module, b'tail')
    for buffer, size in zip(buffers, (1000, 2000, 3000, len(FRAME))):
        call('hipMalloc', ctypes.byref(buffer), size)
    call('hipHostMalloc', ctypes.byref(gate), 4, 0)
    call('hipMemsetAsync', buffers[0], 7, 1000, stream)
    payload = ctypes.create_string_buffer(UPLOAD, len(UPLOAD))
    call('hipMemcpy', buffers[1], ctypes.cast(payload, P), len(UPLOAD), 1)
    for event in events:
        call('hipEventCreate', ctypes.byref(event))

    # A frame as dlsslopd serves one: the input comes from registered memory,
    # between the events that analyze.py takes as the frame's bounds.
    staging = ctypes.create_string_buffer(FRAME, len(FRAME))
    call('hipHostRegister', ctypes.cast(staging, P), len(FRAME), 1)
    call('hipEventRecord', events[0], stream)
    call('hipMemcpyAsync', buffers[3], ctypes.cast(staging, P), len(FRAME), 1, stream)
    call('hipEventRecord', events[1], stream)

    def params(*values):
        return (P * len(values))(*(ctypes.addressof(v) for v in values))

    def launch(function, *values):
        call('hipModuleLaunchKernel', function, 4, 1, 1, 64, 1, 1, 0, stream, params(*values), None)

    probe_arguments = (P(buffers[3].value), P(buffers[1].value), P(buffers[2].value + 64), ctypes.c_uint32(1280),
                       ctypes.c_uint32(7), ctypes.c_uint64(0x0102030405060708), P(gate.value))
    tail_arguments = (P(buffers[0].value + 16), ctypes.c_uint16(9))
    launch(probe, *probe_arguments)
    launch(tail, *tail_arguments)
    launch(probe, *probe_arguments)
    # Its global size is in threads: 256 of them in groups of 64 are 4 groups.
    call('hipExtModuleLaunchKernel', tail, 256, 1, 1, 64, 1, 1, 0, stream, params(*tail_arguments), None, None, None,
         0)
    call('hipEventRecord', events[2], stream)
    call('hipMemcpyAsync', ctypes.cast(staging, P), buffers[3], len(FRAME), 2, stream)
    call('hipEventRecord', events[3], stream)
    elapsed = ctypes.c_float()
    call('hipEventElapsedTime', ctypes.byref(elapsed), events[0], events[1])
    call('hipEventElapsedTime', ctypes.byref(elapsed), events[2], events[3])

    call('hipGraphLaunch', None, stream, expected=500)
    host = ctypes.create_string_buffer(1000)
    call('hipMemcpy', ctypes.cast(host, P), buffers[0], 1000, 2)
    for event in events:
        call('hipEventDestroy', event)
    call('hipHostUnregister', ctypes.cast(staging, P))
    call('hipHostFree', gate)
    for buffer in buffers:
        call('hipFree', buffer)


def run(command, expected=0, env=None):
    result = subprocess.run([str(c) for c in command], env=env, text=True, capture_output=True, timeout=60)
    assert result.returncode == expected, (command, result.returncode, result.stdout, result.stderr)
    return result.stdout


def tool(name, *arguments, expected=0):
    return run([sys.executable, HERE / name, *arguments], expected=expected)


UPLOAD = bytes((i * 7) & 255 for i in range(2000))
FRAME = bytes((i * 13 + 5) & 255 for i in range(512))
KERNELS = [
    kernel('probe', [('input', 'global_buffer', 8), ('weights', 'global_buffer', 8), ('scratch', 'global_buffer', 8),
                     ('width', 'by_value', 4), ('count', 'by_value', 4), ('seed', 'by_value', 8),
                     ('gate', 'global_buffer', 8), ('', 'hidden_block_count_x', 4)]),
    kernel('tail', [('output', 'global_buffer', 8), ('mode', 'by_value', 2)]),
]

with tempfile.TemporaryDirectory(prefix='hiptrace-test-') as directory:
    root = Path(directory)
    modules = root / 'modules'
    modules.mkdir()
    image = code_object(KERNELS)
    (modules / 'probe.hsaco').write_bytes(image)
    trace = root / 'probe.trace'
    pid = os.fork()
    if not pid:
        status = 1
        try:
            drive(trace, image, modules)
            status = 0
        except BaseException as error:
            print(f'hiptrace test: driving the runtime: {error!r}', file=sys.stderr)
        # C's exit runs the runtime's destructor, which logs the launch count;
        # Python's own shutdown would unwind the parent's context here.
        ctypes.CDLL(None).exit(status)
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0, 'the runtime could not be driven'

    lines = trace.read_text().splitlines()
    header = re.fullmatch(rf'0 t0 trace real={re.escape(str(fake))} module_files=1 modules_dir={re.escape(str(modules))}'
                          r' deep=0-1 deep_kernels=1 missing=((?: \w+)+)', lines[0])
    assert header, lines[0]
    missing = header[1].split()
    zeros, sevens, tail_sevens = fnv(bytes(2936)), fnv(b'\x07' * 1000), fnv(b'\x07' * 984)
    probe_args = 'd4+0,d2+0,d3+64,1280,7,0x0102030405060708,h1+0'
    tail_deep = f'args=d1+16,0x0009 rc=0 deep a0=984:{tail_sevens:016x}'
    expected = [
        'hipStreamCreate s1 rc=0',
        f'hipModuleLoadData m1 file=probe.hsaco bytes={len(image)} fnv={fnv(image):016x} kernels=2',
        'hipModuleGetFunction f1 m1 k=probe.hsaco:probe args=7 kernarg=56 lds=512 scratch=0 vgpr=24 sgpr=16 '
        'vgpr_spill=0 sgpr_spill=0 wave=32 max_group=256 layout=p8@0,p8@8,p8@16,v4@24,v4@28,v8@32,p8@40,h4@48 rc=0',
        'hipModuleGetFunction f2 m1 k=probe.hsaco:tail args=2 kernarg=16 lds=512 scratch=0 vgpr=24 sgpr=16 '
        'vgpr_spill=0 sgpr_spill=0 wave=32 max_group=256 layout=p8@0,v2@8 rc=0',
        'hipMalloc d1 bytes=1000 flags=0',
        'hipMalloc d2 bytes=2000 flags=0',
        'hipMalloc d3 bytes=3000 flags=0',
        f'hipMalloc d4 bytes={len(FRAME)} flags=0',
        'hipHostMalloc h1 bytes=4 flags=0',
        'hipMemsetAsync dst=d1+0 value=7 bytes=1000 s=s1 rc=0',
        f'hipMemcpy kind=H2D dst=d2+0 src=host bytes=2000 src_fnv={fnv(UPLOAD):016x} rc=0',
        *(f'hipEventCreate e{n} rc=0' for n in range(1, 5)),
        f'hipHostRegister r1 bytes={len(FRAME)} flags=1',
        'hipEventRecord e1 s=s1 rc=0',
        f'hipMemcpyAsync kind=H2D dst=d4+0 src=r1+0 bytes={len(FRAME)} s=s1 src_fnv={fnv(FRAME):016x} rc=0',
        'hipEventRecord e2 s=s1 rc=0',
        f'launch n=0 f=f1 k=probe.hsaco:probe grid=4x1x1 block=64x1x1 lds=0 s=s1 args={probe_args} rc=0 deep '
        f'a0=uploaded a1=uploaded a2=2936:{zeros:016x} a6=4:{fnv(bytes(4)):016x}',
        f'launch n=1 f=f2 k=probe.hsaco:tail grid=4x1x1 block=64x1x1 lds=0 s=s1 {tail_deep}',
        f'launch n=2 f=f1 k=probe.hsaco:probe grid=4x1x1 block=64x1x1 lds=0 s=s1 args={probe_args} rc=0',
        f'ext_launch n=3 f=f2 k=probe.hsaco:tail grid=4x1x1 block=64x1x1 lds=0 s=s1 {tail_deep}',
        'hipEventRecord e3 s=s1 rc=0',
        f'hipMemcpyAsync kind=D2H dst=r1+0 src=d4+0 bytes={len(FRAME)} s=s1 rc=0',
        'hipEventRecord e4 s=s1 rc=0',
        'hipEventElapsedTime e1 e2 rc=0',
        'hipEventElapsedTime e3 e4 rc=0',
        'hipGraphLaunch rc=500',
        f'hipMemcpy kind=D2H dst=host src=d1+0 bytes=1000 dst_fnv={sevens:016x} rc=0',
        *(f'hipEventDestroy e{n}' for n in range(1, 5)),
        f'hipHostUnregister r1+0 bytes={len(FRAME)}',
        'hipHostFree h1+0 bytes=4',
        'hipFree d1+0 bytes=1000',
        'hipFree d2+0 bytes=2000',
        'hipFree d3+0 bytes=3000',
        f'hipFree d4+0 bytes={len(FRAME)}',
        'exit launches=4',
    ]
    called = {line.split(' ', 1)[0] for line in expected} - {'launch', 'ext_launch', 'exit', 'hipGraphLaunch'}
    called |= {'hipModuleLaunchKernel', 'hipExtModuleLaunchKernel', 'hipStreamSynchronize'}
    assert 'hipGraphLaunch' in missing and not called & set(missing), missing
    got = [re.sub(r'^\d+ t0 ', '', line) for line in lines[1:]]
    assert [line.split(' ', 1)[0] for line in lines] == [str(n) for n in range(len(lines))], lines
    for n, (want, have) in enumerate(zip(expected, got), 1):
        assert want == have, f'line {n}:\n  expected {want}\n  got      {have}'
    assert len(got) == len(expected), got[len(expected):] or expected[len(got):]

    # The tools: two evaluations, each starting at the first network kernel,
    # in one frame.
    tool('analyze.py', trace)
    summary = (root / 'probe.summary.txt').read_text()
    for line in ('== network evaluations: 2 (each starts with probe.hsaco:probe) ==',
                 '  launches per evaluation: 2 2',
                 '== served or self-test frames (hipEventRecord e1 .. hipEventElapsedTime e3 e4): 1 ==',
                 '  frame0: evaluations [0, 1]; launches 4 (network 4); native {}'):
        assert line + '\n' in summary, (line, summary)
    assert tool('compare.py', trace, trace).splitlines() == ['MATCH']
    empty = root / 'empty.trace'
    empty.write_text(lines[0] + '\n')
    assert tool('compare.py', trace, empty, expected=1).splitlines() == \
        [f'{empty}: no network evaluation', 'DIFFERENT']
    assert tool('compare.py', empty, empty, expected=1).splitlines() == \
        [f'{empty}: no network evaluation'] * 2 + ['DIFFERENT']
    changed = root / 'changed.trace'
    changed.write_text(trace.read_text().replace('args=d1+16,0x0009', 'args=d3+16,0x0009'))
    report = tool('compare.py', trace, changed, expected=1)
    assert 'eval0: 2 vs 2 launches; first difference at launch 1:' in report and report.endswith('DIFFERENT\n'), report
    canonical = ('probe.hsaco:probe\t4x1x1\t64x1x1\t0\tB0+0,B1+0,B2+64,1280,7,0x0102030405060708,B3+0\n'
                 'probe.hsaco:tail\t4x1x1\t64x1x1\t0\tB4+16,0x0009\n')
    assert tool('plan_hashes.py', '--print', '0', trace) == canonical
    assert tool('plan_hashes.py', '--print', '1', trace) == canonical
    evaluation, weights = f'{fnv(canonical.encode()):016x}/2', fnv(b'2000 0 1\n')
    # The pool leaves out the network's input, d4, and the uploaded d2.
    assert tool('plan_hashes.py', trace).splitlines() == [
        f'probe.trace eval0={evaluation} eval1={evaluation} kernels=2',
        '  pool 1000 3000 (2)',
        '  gathers',
        f'  weights 1 2000 uploads={weights:016x} unread=0',
    ]
    tool('plan_hashes.py', empty, expected=1)

    # HIPTRACE_DEEP: ordinals and ranges, and a malformed value refused.
    for value, logged in (('3,5-7', '3-4,5-7'), ('10:20', 'invalid')):
        header_only = root / 'deep.trace'
        result = subprocess.run([sys.executable, '-c', f'import ctypes; ctypes.CDLL({str(runtime)!r})'],
                                env=os.environ | {'HIPTRACE_REAL': str(fake), 'HIPTRACE_FILE': str(header_only),
                                                  'HIPTRACE_DEEP': value},
                                text=True, capture_output=True, timeout=60)
        assert result.returncode == 0, result
        assert f' deep={logged} ' in header_only.read_text().splitlines()[0], header_only.read_text()
        assert ('HIPTRACE_DEEP=10:20 is not' in result.stderr) == (logged == 'invalid'), result.stderr
    for name in ('analyze.py', 'compare.py', 'plan_hashes.py'):
        assert '(default: off)' in tool(name, '--help')

    # trace.sh: its defaults, and the commands it runs with a stand-in dlsslopd
    # that records them and exits with 3.
    script = HERE / 'trace.sh'
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(('HIPTRACE_', 'DLSSLOP_', 'XDG_'))}
    env['HOME'] = str(root / 'home')
    helptext = run([script, '--help'], env=env)
    options = re.split(r'\n(?=  -)', helptext.split('\nOptions:\n', 1)[1].split('\n\n', 1)[0])
    assert [re.match(r'  -\w, --([\w-]+)', entry)[1] for entry in options] == \
        ['build', 'daemon', 'modules', 'assets', 'output', 'channel', 'name', 'tier', 'self-test', 'performance',
         'passes', 'motion', 'frames', 'deep', 'deep-kernels', 'help'], options
    assert all('(default: ' in entry for entry in options), options
    assert f'(default: {root}/home/.local/share/dlsslop-amd/model:' in helptext, helptext
    assert f'(default: /tmp/dlsslop-hiptrace-{os.getuid()}:' in helptext, helptext
    helptext = run([script, '--help'], env=env | {'XDG_DATA_HOME': '/data', 'XDG_RUNTIME_DIR': '/run/me'})
    assert '(default: /data/dlsslop-amd/model:' in helptext and '(default: /run/me/dlsslop-hiptrace:' in helptext
    for arguments in (['--bogus'], ['--tier', '480'], ['--self-test', '--motion'], ['--passes', '0'], ['--tier'],
                      ['--deep', '10:20'], ['--deep', '1-2,'], ['serve']):
        run([script, *arguments], expected=2, env=env)
    record = root / 'record.json'
    daemon = root / 'dlsslopd'
    daemon.write_text(f'#!{sys.executable}\n'
                      'import json, os, sys\n'
                      'names = ("DLSSLOP_HIP_LIBRARY", "HIPTRACE_FILE", "HIPTRACE_MODULES", "HIPTRACE_DEEP",'
                      ' "HIPTRACE_DEEP_KERNELS")\n'
                      f'open({str(record)!r}, "w").write(json.dumps({{"argv": sys.argv[1:],'
                      ' "env": {name: os.environ.get(name) for name in names}}))\n'
                      'print("stand-in dlsslopd")\n'
                      'sys.exit(3)\n')
    daemon.chmod(0o755)
    (root / 'assets').mkdir()
    output, channel = root / 'out', root / 'channel'
    common = ['--build', runtime.parent, '--daemon', daemon, '--modules', modules, '--assets', root / 'assets',
              '--output', output, '--channel', channel]
    run([script, *common, '--self-test', '-t', '900', '-p', '-D', '0-5'], expected=3, env=env)
    name = 'selftest-900-perf'
    assert stat.S_IMODE(channel.stat().st_mode) == 0o700
    assert 'stand-in dlsslopd' in (output / f'{name}.log').read_text()
    assert json.loads(record.read_text()) == {
        'argv': ['--config', '/dev/null', '--backend', 'hip', '--tier', '900', '--passes', '1', '--modules',
                 str(modules), '--assets', str(root / 'assets'), '--shm', f'{channel}/{name}.bin', '--performance',
                 '--self-test'],
        'env': {'DLSSLOP_HIP_LIBRARY': str(runtime), 'HIPTRACE_FILE': f'{output}/{name}.trace',
                'HIPTRACE_MODULES': str(modules), 'HIPTRACE_DEEP': '0-5', 'HIPTRACE_DEEP_KERNELS': ''}}
    # Served: the client stores the settings, then starts the stand-in, which
    # exits before it is ready.
    run([script, *common, '--motion', '--passes', '2', '--frames=ABBA', '-k', 'tail'], expected=1, env=env)
    name = 'serve-720-motion-passes2'
    log = (output / f'{name}.log').read_text()
    assert 'settings: tier=720 passes=2 mvec=1 before frame 0' in log and 'daemon exited before it was ready' in log
    assert 'stand-in dlsslopd' in (output / f'{name}.daemon.log').read_text()
    got = json.loads(record.read_text())
    assert got['argv'] == ['--config', '/dev/null', '--backend', 'hip', '--tier', '720', '--passes', '2', '--modules',
                           str(modules), '--assets', str(root / 'assets'), '--shm', f'{channel}/{name}.bin'], got
    assert got['env']['HIPTRACE_FILE'] == f'{output}/{name}.trace' and got['env']['HIPTRACE_DEEP_KERNELS'] == 'tail'

print('PASS: tracing runtime trace, trace tools and run script')
