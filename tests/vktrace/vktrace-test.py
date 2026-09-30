#!/usr/bin/env python3
"""Check the tracing Vulkan layer and its tools.

Without a device: trace.sh's options and defaults, and the commands it runs
with a stand-in dlsslopd and a stand-in shmclient. On Mesa's lavapipe, with
the Khronos validation layer below the tracing layer, vktrace-probe runs a
setup submission, a one-shot dispatch and two frames. Its trace must hold every
command as recorded, descriptor writes and copies that span bindings under the
bindings they reach, and the FNV-1a 64 of what the probe itself wrote and read
in its Upload, Readback and Hash lines; the validation layer must report
nothing, also about the layer's own hashing commands. compare.py must match the
trace with itself, with a trace hashed another way, with one whose content
hashes come in another order, with one whose setup is split and leaves a
staging buffer alive and, told to ignore queries, with one whose frames use no
query pool, and must find each changed field of a frame, those queries, a
changed state after setup and a setup left unhashed; analyze.py must summarise
it.
Exits 77 after the device-free checks when lavapipe or the validation layer is
missing.
"""
import argparse
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(description=__doc__, add_help=False, allow_abbrev=False,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('-h', '--help', action='help', help='show this help and exit (default: off)')
parser.add_argument('layer', type=Path,
                    help='the tracing layer, libvktrace.so, with its manifest and probe.spv in vktrace/ beside it '
                         '(required; no default)')
parser.add_argument('probe', type=Path, help='vktrace-probe (required; no default)')
parser.add_argument('client', type=Path,
                    help='shmclient, in the same directory as the layer (required; no default)')
args = parser.parse_args()
layer, probe, client = args.layer.resolve(), args.probe.resolve(), args.client.resolve()
build, manifests = layer.parent, layer.parent / 'vktrace'
assert client.parent == build, 'trace.sh finds the layer and the client in one build tree'
assert (manifests / 'VkLayer_LOCAL_vktrace.json').is_file() and (manifests / 'probe.spv').is_file(), manifests


def run(command, expected=0, env=None):
    result = subprocess.run([str(c) for c in command], env=env, text=True, capture_output=True, timeout=120)
    assert result.returncode == expected, (command, result.returncode, result.stdout, result.stderr)
    return result.stdout


def tool(name, *arguments, expected=0):
    return run([sys.executable, HERE / name, *arguments], expected=expected)


def lavapipe():
    """Mesa's lavapipe ICD manifest in the loader's search path, or None."""
    data = os.environ.get('XDG_DATA_DIRS') or '/usr/local/share:/usr/share'
    for directory in ['/etc/vulkan/icd.d'] + [f'{d}/vulkan/icd.d' for d in data.split(':') if d]:
        found = sorted(Path(directory).glob('lvp_icd*.json'))
        if found:
            return found[0]
    return None


with tempfile.TemporaryDirectory(prefix='vktrace-test-') as directory:
    root = Path(directory)
    clean = {key: value for key, value in os.environ.items()
             if not key.startswith(('VKTRACE_', 'VK_', 'XDG_', 'DLSSLOP_', 'DLSSNR_'))}

    # trace.sh: its defaults, and the commands it runs with a stand-in
    # dlsslopd that records them and exits with 3.
    script = HERE / 'trace.sh'
    env = clean | {'HOME': str(root / 'home')}
    helptext = run([script, '--help'], env=env)
    options = re.split(r'\n(?=  -)', helptext.split('\nOptions:\n', 1)[1].split('\n\n', 1)[0])
    assert [re.match(r'  -\w, --([\w-]+)', entry)[1] for entry in options] == \
        ['build', 'daemon', 'spirv', 'model', 'output', 'channel', 'name', 'tier', 'self-test', 'passes', 'motion',
         'fp16', 'frames', 'hash', 'hash-submits', 'help'], options
    assert all('(default: ' in entry for entry in options), options
    assert f'(default: {root}/home/.local/share/dlsslop-amd/dlssnr.bin:' in helptext, helptext
    assert f'(default: /tmp/dlsslop-vktrace-{os.getuid()}:' in helptext, helptext
    helptext = run([script, '--help'], env=env | {'XDG_DATA_HOME': '/data', 'XDG_RUNTIME_DIR': '/run/me'})
    assert '(default: /data/dlsslop-amd/dlssnr.bin:' in helptext and '(default: /run/me/dlsslop-vktrace:' in helptext
    for arguments in (['--bogus'], ['--tier', '480'], ['--self-test', '--motion'], ['--self-test', '--fp16'],
                      ['--self-test', '--', '--sharpness', '0.5'], ['--passes', '0'], ['--tier'], ['serve'],
                      ['--name', 'a/b']):
        run([script, *arguments], expected=2, env=env)
    record = root / 'record.json'
    daemon = root / 'dlsslopd'
    daemon.write_text(f'#!{sys.executable}\n'
                      'import json, os, sys\n'
                      'names = ("VK_ADD_LAYER_PATH", "VK_INSTANCE_LAYERS", "VKTRACE_FILE", "VKTRACE_SPIRV",'
                      ' "VKTRACE_HASH", "VKTRACE_HASH_SUBMITS")\n'
                      f'open({str(record)!r}, "w").write(json.dumps({{"argv": sys.argv[1:],'
                      ' "env": {name: os.environ.get(name) for name in names}}))\n'
                      'print("stand-in dlsslopd")\n'
                      'sys.exit(3)\n')
    daemon.chmod(0o755)
    model, output, channel = root / 'dlssnr.bin', root / 'out', root / 'channel'
    common = ['--build', build, '--daemon', daemon, '--model', model, '--output', output, '--channel', channel]
    run([script, *common, '--self-test'], expected=1, env=env)
    model.write_bytes(b'model')
    run([script, *common, '--self-test', '-t', '1080', '-P', '2', '-D', 'all', '--hash-submits=7-12'], expected=3,
        env=env)
    name = 'selftest-1080-passes2'
    assert stat.S_IMODE(channel.stat().st_mode) == 0o700
    assert 'stand-in dlsslopd' in (output / f'{name}.log').read_text()
    assert json.loads(record.read_text()) == {
        'argv': ['--config', '/dev/null', '--backend', 'vulkan', '--tier', '1080', '--passes', '2', '--vulkan-model',
                 str(model), '--shm', f'{channel}/{name}.bin', '--self-test', '--output', f'{output}/{name}.ppm'],
        'env': {'VK_ADD_LAYER_PATH': str(manifests), 'VK_INSTANCE_LAYERS': 'VK_LAYER_LOCAL_vktrace',
                'VKTRACE_FILE': f'{output}/{name}.trace', 'VKTRACE_SPIRV': f'{build}/vulkan-nr/network',
                'VKTRACE_HASH': 'all', 'VKTRACE_HASH_SUBMITS': '7-12'}}
    # Served: the client stores the settings, then starts the stand-in, which
    # exits before it is ready. The layer goes above those the environment names.
    run([script, *common, '--motion', '-H', '--frames=ABBA', '--spirv', 'shaders', '--', '--sharpness', '1:0.5'],
        expected=1, env=env | {'VK_INSTANCE_LAYERS': 'VK_LAYER_KHRONOS_validation'})
    name = 'serve-720-fp16-motion'
    log = (output / f'{name}.log').read_text()
    assert 'settings: tier=720 passes=1 mvec=1 before frame 0' in log and 'daemon exited before it was ready' in log
    assert 'stand-in dlsslopd' in (output / f'{name}.daemon.log').read_text()
    got = json.loads(record.read_text())
    assert got['argv'] == ['--config', '/dev/null', '--backend', 'vulkan', '--tier', '720', '--passes', '1',
                           '--vulkan-model', str(model), '--shm', f'{channel}/{name}.bin'], got
    assert got['env'] | {'VKTRACE_FILE': None} == {
        'VK_ADD_LAYER_PATH': str(manifests), 'VK_INSTANCE_LAYERS': 'VK_LAYER_LOCAL_vktrace:VK_LAYER_KHRONOS_validation',
        'VKTRACE_FILE': None, 'VKTRACE_SPIRV': 'shaders', 'VKTRACE_HASH': '',
        'VKTRACE_HASH_SUBMITS': ''}, got
    # The client's whole command line, from a stand-in client in a build tree
    # that holds the layer, which records it and exits with 5.
    standin = root / 'standin'
    standin.mkdir()
    (standin / 'libvktrace.so').symlink_to(layer)
    (standin / 'vktrace').symlink_to(manifests)
    client_record = root / 'client.json'
    (standin / 'shmclient').write_text(f'#!{sys.executable}\n'
                                       'import json, sys\n'
                                       f'open({str(client_record)!r}, "w").write(json.dumps(sys.argv[1:]))\n'
                                       'sys.exit(5)\n')
    (standin / 'shmclient').chmod(0o755)
    for arguments, name, width, frames, extra in (
            (['--tier', '900', '--passes', '2', '--motion', '--fp16', '--frames', 'ABBA', '--', '--sharpness',
              '1:0.5', '--intensity', '0.5'], 'serve-900-fp16-motion-passes2', '1600', 'ABBA',
             ['--mvec', '1', '--fp16', '--sharpness', '1:0.5', '--intensity', '0.5']),
            ([], 'serve-720', '1280', 'AAB', [])):
        run([script, '--build', standin, *common[2:], *arguments], expected=5, env=env)
        tier = str(int(width) * 9 // 16)
        passes = '2' if 'passes2' in name else '1'
        assert json.loads(client_record.read_text()) == [
            '--shm', f'{channel}/{name}.bin', '--width', width, '--height', tier, '--tier', tier, '--passes', passes,
            '--frames', frames, '--log', f'{output}/{name}.daemon.log', *extra, '--', str(daemon), '--config',
            '/dev/null', '--backend', 'vulkan', '--tier', tier, '--passes', passes, '--vulkan-model', str(model),
            '--shm', f'{channel}/{name}.bin'], client_record.read_text()
    for name in ('analyze.py', 'compare.py', 'gen_names.py'):
        assert '(default: off)' in tool(name, '--help')

    icd = lavapipe()
    if not icd:
        print('SKIP: no lavapipe ICD; the device-free checks passed')
        sys.exit(77)

    def trace(name, hash, submits='', options=()):
        """Runs the probe on lavapipe under the tracing layer, and none of the
        user's own layers; returns its output and trace."""
        path = root / f'{name}.trace'
        result = subprocess.run([str(probe), *options, str(manifests / 'probe.spv')], text=True, capture_output=True,
                                timeout=120,
                                env=clean | {'XDG_DATA_HOME': str(root), 'XDG_CONFIG_HOME': str(root),
                                             'VK_DRIVER_FILES': str(icd), 'VK_ADD_LAYER_PATH': str(manifests),
                                             'VK_INSTANCE_LAYERS': 'VK_LAYER_LOCAL_vktrace', 'VKTRACE_FILE': str(path),
                                             'VKTRACE_SPIRV': str(manifests), 'VKTRACE_HASH': hash,
                                             'VKTRACE_HASH_SUBMITS': submits,
                                             'VK_KHRONOS_VALIDATION_VALIDATE_SYNC': 'true'})
        if result.returncode == 77:
            print(result.stderr, end='')
            print('SKIP: the device-free checks passed')
            sys.exit(77)
        assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)
        output = result.stdout + result.stderr
        assert 'Validation Error' not in output and 'VUID' not in output, output
        return output, path

    def calls(path):
        """The trace's lines without their sequence and thread numbers."""
        return [re.sub(r'^\d+ t\d+ ', '', line) for line in path.read_text().splitlines()]

    def recorded(lines, begin, submit):
        """The lines from the first BEGIN to the first SUBMIT after it."""
        start = lines.index(begin)
        return lines[start:lines.index(submit, start) + 1]

    # The probe's own FNVs: the upload and the fill of the setup, what the
    # one-shot dispatch writes, and each frame's result, storage image and b.
    output, storage = trace('storage', 'storage,i2b')
    setup = re.search(r'^setup upload=(\w+) fill=(\w+)$', output, re.M)
    oneshot = re.search(r'^oneshot result=(\w+) b=(\w+)$', output, re.M)
    frames = re.findall(r'^frame (\d) upload=(\w+) result=(\w+) image=(\w+) b=(\w+)$', output, re.M)
    assert setup and oneshot and len(frames) == 2 and frames[0][2:4] != frames[1][2:4], output
    upload, fill = setup.groups()
    text = storage.read_text()
    assert text.startswith(f'# vktrace 1 pid='), text
    assert f' hash=storage,i2b hash_submits=- spirv={manifests}\n' in text.splitlines(True)[0], text
    for pattern in (r' CreateShaderModule shm=shm1 bytes=\d+ fnv=[0-9a-f]{16} file=probe\.spv rc=0\n',
                    r' DeviceFeatures dev=dev1 struct=VkPhysicalDeviceFeatures on=\[shaderInt64\]\n',
                    r' DeviceFeatures dev=dev1 struct=VkPhysicalDeviceVulkan12Features on=\[shaderInt8\]\n',
                    r' DeviceFeatures dev=dev1 struct=VkPhysicalDeviceVulkan13Features on=\[synchronization2\]\n',
                    rf' Upload sub=1 at=cb1#0\.0 mem=mem\d+\+0 bytes=16384 fnv={upload}\n',
                    r' CreateComputePipeline pipe=pipe2 cache=null layout=pl1 flags=0x0 stage=0x20 module=shm1 '
                    r'entry=main stageflags=0x0 spec=\[0:0\+4\] specdata=03000000 chain=\[\] rc=0\n'):
        assert re.search(pattern, text), pattern
    lines = calls(storage)
    # A write and a copy that each span bindings 0 and 1, then the image.
    descriptors = ['UpdateDescriptorSets writes=2 copies=0',
                   'DescriptorWrite set=ds1 binding=0 elem=0 type=STORAGE_BUFFER buf=buf2 off=0 range=WHOLE',
                   'DescriptorWrite set=ds1 binding=1 elem=0 type=STORAGE_BUFFER buf=buf3 off=16384 range=16384',
                   'DescriptorWrite set=ds1 binding=2 elem=0 type=STORAGE_IMAGE view=view1 layout=GENERAL smp=null',
                   'UpdateDescriptorSets writes=0 copies=2',
                   'DescriptorCopy src=ds1:0:0 dst=ds2:0:0',
                   'DescriptorCopy src=ds1:1:0 dst=ds2:1:0',
                   'DescriptorCopy src=ds1:2:0 dst=ds2:2:0']
    assert recorded(lines, descriptors[0], descriptors[-1]) == descriptors, lines
    # Every command as recorded: the one-shot dispatch, then the two frames.
    expected = ['CreateCommandPool cp=cp2 family=0 flags=0x0 rc=0',
                'AllocateCommandBuffers cp=cp2 level=PRIMARY cbs=[cb2] rc=0',
                'BeginCommandBuffer cb=cb2 flags=0x1',
                'CmdPipelineBarrier cb=cb2 i=0 src=0x1000 dst=0x800 dep=0x0 mem=[0x1000>0x60] buf=[] '
                'img=[img1:UNDEFINED>GENERAL:0x0>0x40:ign>ign:0x1/0+1/0+1]',
                'CmdBindPipeline cb=cb2 i=1 bind=COMPUTE pipe=pipe1',
                'CmdBindDescriptorSets cb=cb2 i=2 bind=COMPUTE layout=pl1 first=0 sets=[ds1] dyn=[]',
                'CmdPushConstants cb=cb2 i=3 layout=pl1 stages=0x20 offset=0 size=8 data=0500000040000000',
                'CmdDispatch cb=cb2 i=4 n=0 pipe=pipe1 x=64 y=1 z=1',
                'EndCommandBuffer cb=cb2 cmds=5 dispatches=1 rc=0',
                'ResetFences fences=[fence1] rc=0',
                'QueueSubmit sub=2 q=q1 batch=0 cbs=[cb2] wait=[] signal=[] fence=fence1']
    assert recorded(lines, expected[0], expected[-1]) == expected, lines
    assert lines.index('DestroyCommandPool cp=cp2') < lines.index('BeginCommandBuffer cb=cb1 flags=0x1', lines.index(
        expected[-1])), lines
    for k, old in enumerate(('GENERAL', 'SHADER_READ_ONLY_OPTIMAL')):
        expected = [
            'ResetCommandBuffer cb=cb1 flags=0x0',
            'BeginCommandBuffer cb=cb1 flags=0x1',
            'CmdResetQueryPool cb=cb1 i=0 qp=qp1 first=0 count=2',
            'CmdWriteTimestamp cb=cb1 i=1 stage=0x1 qp=qp1 query=0',
            'CmdCopyBuffer cb=cb1 i=2 src=buf1 dst=buf2 regions=[0>0+16384]',
            f'CmdPipelineBarrier cb=cb1 i=3 src=0x1800 dst=0x800 dep=0x0 mem=[0x1000>0x20] buf=[] '
            f'img=[img1:{old}>GENERAL:0x40>0x40:ign>ign:0x1/0+1/0+1]',
            f'CmdBindPipeline cb=cb1 i=4 bind=COMPUTE pipe=pipe{k + 1}',
            f'CmdBindDescriptorSets cb=cb1 i=5 bind=COMPUTE layout=pl1 first=0 sets=[ds{k + 1}] dyn=[]',
            f'CmdPushConstants cb=cb1 i=6 layout=pl1 stages=0x20 offset=0 size=8 data=0{7 + k}00000040000000',
            f'CmdDispatch cb=cb1 i=7 n=0 pipe=pipe{k + 1} x=64 y=1 z=1',
            'CmdPipelineBarrier2 cb=cb1 i=8 dep=0x0 mem=[0x800:0x400000000>0x100000000:0x800] buf=[] '
            'img=[img1:GENERAL>TRANSFER_SRC_OPTIMAL:0x800:0x400000000>0x100000000:0x800:ign>ign:0x1/0+1/0+1]',
            'CmdCopyBuffer cb=cb1 i=9 src=buf3 dst=buf4 regions=[16384>0+16384]',
            'CmdCopyImageToBuffer cb=cb1 i=10 src=img1:TRANSFER_SRC_OPTIMAL dst=buf4 '
            'regions=[16384:0:0:0x1/0/0+1:0,0,0:64x64x1]',
            'CmdPipelineBarrier cb=cb1 i=11 src=0x1000 dst=0x4000 dep=0x0 mem=[0x1000>0x2000] buf=[] img=[]',
            *(['CmdPipelineBarrier cb=cb1 i=12 src=0x1000 dst=0x800 dep=0x0 mem=[] buf=[] img=[img1:'
               'TRANSFER_SRC_OPTIMAL>SHADER_READ_ONLY_OPTIMAL:0x0>0x20:ign>ign:0x1/0+1/0+1]'] if k == 0 else []),
            f'CmdWriteTimestamp cb=cb1 i={13 - k} stage=0x2000 qp=qp1 query=1',
            f'EndCommandBuffer cb=cb1 cmds={14 - k} dispatches=1 rc=0',
            'ResetFences fences=[fence1] rc=0',
            f'QueueSubmit sub={3 + k} q=q1 batch=0 cbs=[cb1] wait=[] signal=[] fence=fence1']
        start = lines.index(f'QueueSubmit sub={2 + k} q=q1 batch=0 cbs=[cb{2 - k}] wait=[] signal=[] fence=fence1')
        assert recorded(lines[start:], expected[0], expected[-1]) == expected, lines[start:]
    # The storage the one-shot dispatch and each frame bound, from the write
    # and from the copy; the layer hashed none of the setup.
    assert f' Hash sub=2 sel=dispatch0:set0.b1 res=buf3+16384+16384 bytes=16384 fnv={oneshot[1]}\n' in text
    for index, _, result, image, _ in frames:
        sub = int(index) + 3
        for pattern in (rf' Upload sub={sub} at=cb1#2\.0 mem=mem\d+\+0 bytes=16384 fnv={upload}\n',
                        rf' Readback sub={sub} at=cb1#9\.0 mem=mem\d+\+0 bytes=16384 fnv={result} via=fence\n',
                        rf' Readback sub={sub} at=cb1#10\.0 mem=mem\d+\+16384 bytes=16384 fnv={image} via=fence\n',
                        rf' Hash sub={sub} sel=dispatch0:set0\.b0 res=buf2\+0\+16384 bytes=16384 fnv={upload}\n',
                        rf' Hash sub={sub} sel=dispatch0:set0\.b1 res=buf3\+16384\+16384 bytes=16384 fnv={result}\n',
                        rf' Hash sub={sub} sel=dispatch0:set0\.b2 res=img1 bytes=16384 fnv={image}\n',
                        rf' Hash sub={sub} sel=i2b:cb1#10\.0 res=buf4\+16384\+16384 skip=no-transfer-src\n'):
            assert re.search(pattern, text), pattern
    assert ' Hash sub=1 ' not in text, text
    oneshot_image = re.search(r' Hash sub=2 sel=dispatch0:set0\.b2 res=img1 bytes=16384 fnv=(\w+)\n', text)[1]

    # The frames in canonical form: the copied set holds what the written one
    # does; the one-shot dispatch is not a frame.
    sys.path.insert(0, str(HERE))
    import vkt
    replayed = vkt.Trace(storage)
    assert [s.number for s in vkt.frames(replayed)] == [3, 4] and replayed.submits[1].one_shot, replayed.submits
    for k, frame in enumerate(vkt.frames(replayed)):
        dispatch = [line for line in vkt.canonical(replayed, frame)[0] if line.startswith('CmdDispatch')]
        assert len(dispatch) == 1 and re.fullmatch(
            r'CmdDispatch pipe=probe\.spv#[0-9a-f]{10} x=64 y=1 z=1 sets=\[0:\{'
            r'b0\.0:type=STORAGE_BUFFER buf=B1 off=0 range=WHOLE;'
            r'b1\.0:type=STORAGE_BUFFER buf=B2 off=16384 range=16384;'
            r'b2\.0:type=STORAGE_IMAGE view=view\(I0,2D,R8G8B8A8_UNORM,0x1/0\+1/0\+1,iiii\) layout=GENERAL '
            r'smp=null\}\] '
            rf'push=\[0{7 + k}00000040000000\]', dispatch[0]), dispatch

    # Every buffer and image after every submission: the setup's state too.
    # The layer copies the image from GENERAL after the one-shot dispatch and
    # from TRANSFER_SRC_OPTIMAL after frame 1, and after frame 0 moves it out
    # of SHADER_READ_ONLY_OPTIMAL and back, where frame 1's first barrier
    # expects it; the validation layer checks both.
    output, hashed = trace('all', 'all')
    assert re.findall(r'^(?:oneshot|frame) .*$', output, re.M) == \
        [f'oneshot result={oneshot[1]} b={oneshot[2]}'] + \
        [f'frame {i} upload={u} result={r} image={m} b={b}' for i, u, r, m, b in frames], output
    text = hashed.read_text()
    for pattern in (rf' Hash sub=1 sel=all res=buf1\+0\+16384 bytes=16384 fnv={upload}\n',
                    rf' Hash sub=1 sel=all res=buf2\+0\+16384 bytes=16384 fnv={upload}\n',
                    rf' Hash sub=1 sel=all res=buf3\+0\+32768 bytes=32768 fnv={fill}\n',
                    r' Hash sub=1 sel=all res=buf4\+0\+32768 skip=no-transfer-src\n',
                    r' Hash sub=1 sel=all res=img1 skip=layout\n',
                    rf' Hash sub=2 sel=all res=buf3\+0\+32768 bytes=32768 fnv={oneshot[2]}\n',
                    rf' Hash sub=2 sel=all res=img1 bytes=16384 fnv={oneshot_image}\n',
                    *(rf' Hash sub={int(i) + 3} sel=all res=buf3\+0\+32768 bytes=32768 fnv={b}\n'
                      for i, _, _, _, b in frames),
                    *(rf' Hash sub={int(i) + 3} sel=all res=img1 bytes=16384 fnv={m}\n' for i, _, _, m, _ in frames)):
        assert re.search(pattern, text), pattern

    # Items that mean nothing are reported and ignored.
    output, ignored = trace('ignored', 'bogus,dispatch=x,img')
    text = ignored.read_text()
    for item in ('bogus', 'dispatch=x', 'img'):
        assert f"vktrace: ignoring VKTRACE_HASH item '{item}'" in output, output
        assert f'\n# ignored VKTRACE_HASH item {item}\n' in text, text
    assert ' Hash ' not in text, text

    # The tools.
    report = tool('compare.py', storage, storage)
    assert report.endswith('compared 2 frames\nMATCH\n'), report
    assert 'setup uploads: 1 identical\n' in report and 'content hashes: dispatch compared: 3\n' in report, report
    report = tool('compare.py', '--frames', '0:1', storage, storage, expected=1)
    assert 'frame 0:1 (submit 3:4): 11 vs 10 commands; first difference at command 3:' in report, report
    assert report.endswith('DIFFERENT\n'), report
    report = tool('compare.py', hashed, hashed)
    assert 'content hashes: all compared: 3\n' in report and report.endswith('MATCH\n'), report
    # Hashing takes the layer's own commands and waits: the probe's command
    # streams, uploads and readbacks stay the same.
    report = tool('compare.py', storage, hashed)
    assert 'content hashes: all only in B: 3\n' in report and report.endswith('MATCH\n'), report
    # Without the frames' query pool resets and timestamps, the frames match
    # only when queries are ignored, and then without the pool itself.
    quiet = root / 'quiet.trace'
    quiet.write_text(''.join(line for line in storage.read_text().splitlines(True)
                             if not re.search(r' Cmd(ResetQueryPool|WriteTimestamp) ', line)))
    report = tool('compare.py', storage, quiet, expected=1)
    assert 'frame 0:0 (submit 3:3): 11 vs 8 commands; first difference at command 0:' in report, report
    report = tool('compare.py', '--ignore-queries', storage, quiet)
    assert report.endswith('compared 2 frames\nMATCH\n'), report

    # One field changed in a copy of the storage trace: each is a difference.
    def edit(name, source, after, anchor, old, new):
        """A copy of SOURCE with OLD replaced by NEW in the first line after the one
        that contains AFTER that contains ANCHOR."""
        text = source.read_text().splitlines(True)
        start = next(n for n, line in enumerate(text) if after in line)
        n = next(n for n in range(start, len(text)) if anchor in text[n])
        assert old in text[n], (anchor, text[n])
        text[n] = text[n].replace(old, new)
        path = root / f'{name}.trace'
        path.write_text(''.join(text))
        return path

    frame1 = ' QueueSubmit sub=3 '
    for name, after, anchor, old, new, notes in (
            ('push', frame1, ' CmdPushConstants ', 'data=08', 'data=09',
             ['frame 1:1 (submit 4:4): 10 vs 10 commands; first difference at command 4:', '  differs: push\n',
              '  push range 0: words 0: 08000000 vs 09000000\n']),
            ('grid', frame1, ' CmdDispatch ', ' x=64 ', ' x=32 ',
             ['frame 1:1 (submit 4:4): 10 vs 10 commands; first difference at command 4:', '  differs: head\n']),
            ('range', '# vktrace', ' DescriptorWrite set=ds1 binding=1 ', 'range=16384', 'range=8192',
             ['frame 0:0 (submit 3:3): 11 vs 11 commands; first difference at command 4:', '  differs: sets\n']),
            ('offset', '# vktrace', ' DescriptorWrite set=ds1 binding=1 ', 'off=16384', 'off=0',
             ['frame 0:0 (submit 3:3): 11 vs 11 commands; first difference at command 4:', '  differs: sets\n']),
            ('specdata', '# vktrace', ' CreateComputePipeline pipe=pipe2 ', 'specdata=03000000', 'specdata=04000000',
             ['pipelines used by frames differ: 2 vs 2 definitions; 1 only in A, 1 only in B\n',
              'frame 1:1 (submit 4:4): 10 vs 10 commands; first difference at command 4:', '  differs: head\n']),
            ('specmap', '# vktrace', ' CreateComputePipeline pipe=pipe2 ', 'spec=[0:0+4]', 'spec=[1:0+4]',
             ['pipelines used by frames differ: 2 vs 2 definitions; 1 only in A, 1 only in B\n',
              'frame 1:1 (submit 4:4): 10 vs 10 commands; first difference at command 4:', '  differs: head\n']),
            ('barrier', frame1, ' CmdPipelineBarrier ', 'mem=[0x1000>0x20]', 'mem=[0x1000>0x1000]',
             ['frame 1:1 (submit 4:4): 10 vs 10 commands; first difference at command 3:']),
            ('barrier2', frame1, ' CmdPipelineBarrier2 ', '0x100000000:0x800]', '0x100000000:0x400000000]',
             ['frame 1:1 (submit 4:4): 10 vs 10 commands; first difference at command 5:']),
            ('upload', frame1, ' Upload sub=4 at=cb1#2.0 ', f'fnv={upload}', 'fnv=0000000000000000',
             ['frame 1:1: uploads ']),
            ('readback', frame1, ' Readback sub=4 at=cb1#9.0 ', f'fnv={frames[1][2]}', 'fnv=0000000000000000',
             ['frame 1:1: readbacks ']),
            ('hash', frame1, ' Hash sub=4 sel=dispatch0:set0.b1 ', f'fnv={frames[1][2]}', 'fnv=0000000000000000',
             ['frame 1:1: content hashes (dispatch) differ: 3 vs 3, 1 only in A, 1 only in B\n'])):
        report = tool('compare.py', storage, edit(name, storage, after, anchor, old, new), expected=1)
        assert all(note in report for note in notes) and report.endswith('DIFFERENT\n'), (name, report)

    lines = hashed.read_text().splitlines()
    # The same content, hashed in another order and under other names.
    moved, held = [], []
    for line in lines + ['']:
        if ' Hash ' in line:
            held.append(re.sub(r'res=buf(\d+)', r'res=buf9\1', line))
            continue
        moved += reversed(held)
        held = []
        moved.append(line)
    (root / 'moved.trace').write_text('\n'.join(moved))
    assert tool('compare.py', hashed, root / 'moved.trace').endswith('MATCH\n')
    # The state after setup is each live region's last hash before the first
    # frame: the one-shot dispatch's b replaces the fill, and a staging buffer
    # destroyed before the first frame does not count, whichever submission
    # hashed it last.
    report = tool('compare.py', hashed, edit('stale', hashed, ' Hash sub=1 sel=all res=buf3', '', f'fnv={fill}',
                                              'fnv=0000000000000000'))
    assert report.endswith('MATCH\n'), report
    report = tool('compare.py', hashed, edit('changed', hashed, ' Hash sub=2 sel=all res=buf3', '', f'fnv={oneshot[2]}',
                                              'fnv=0000000000000000'), expected=1)
    assert 'state after setup: content hashes (all) differ: 5 vs 5, 1 only in A, 1 only in B\n' \
           f'  only A: 1x bytes=32768 {oneshot[2]}\n  only B: 1x bytes=32768 0000000000000000\n' in report, report
    output, split = trace('split', 'all', options=['--split-setup'])
    text = split.read_text()
    assert f' Hash sub=3 sel=all res=buf5+0+16384 bytes=16384 fnv={upload}\n' in text, text
    # Its staging buffer lives until the one-shot submission, sub 3, has completed.
    order = [next(n for n, line in enumerate(calls(split)) if line.startswith(call))
             for call in ('QueueSubmit sub=3 ', 'DestroyBuffer buf=buf5', 'QueueSubmit sub=4 ')]
    assert order == sorted(order), order
    report = tool('compare.py', hashed, split)
    assert 'setup uploads: 1 identical\n' in report and 'content hashes: all compared: 3\n' in report, report
    assert report.endswith('MATCH\n'), report
    # A trace that selects all but hashed only the frames has no state after
    # setup to match.
    output, unhashed = trace('frames-only', 'all', submits='3-4')
    assert ' Hash sub=2 ' not in unhashed.read_text() and ' Hash sub=3 ' in unhashed.read_text()
    report = tool('compare.py', hashed, unhashed, expected=1)
    assert 'state after setup: content hashes (all) only in A, though both traces select all\n' in report, report
    assert report.endswith('DIFFERENT\n'), report

    tool('analyze.py', storage)
    summary = (root / 'storage.summary.txt').read_text()
    assert ', 4 submissions, 2 frames, ' in summary.splitlines()[0], summary
    for suffix in ('frames.tsv', 'frame.txt', 'framediff.txt', 'allocs.tsv', 'pipelines.tsv', 'uploads.tsv'):
        assert (root / f'storage.{suffix}').stat().st_size, suffix

print('PASS: tracing layer on lavapipe, trace tools and run script')
