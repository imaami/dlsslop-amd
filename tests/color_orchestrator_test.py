#!/usr/bin/env python3
"""Exercise the diagnostic CLI against a synthetic, independently driven layer.

The fake control is a separate process and publishes real PNG captures/manifests.
These checks validate orchestration and observations, not Radeon execution.
"""

import fcntl
import json
import os
from pathlib import Path
import select
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "scripts" / "dlsslop-test"
INITIAL = {
    "enabled": "1", "hold": "0", "passes": "2", "mvec": "1",
    "intensity": "0.65", "local-tone": "0.8", "local-structure": "0.7",
    "color-preserve": "0.75", "sharpness": "0.2", "detail": "0.5", "color": "0.3", "bypass": "0",
    "reversible": "1", "debug-view": "0", "debug-scale": "0.75",
    "compare": "0", "apply-model": "1",
    "working-scale": "1", "unrelated-setting": "17",
}


FAKE_CONTROL = r'''
import json
import os
from pathlib import Path
import struct
import sys
import time
import zlib

state_path = Path(os.environ['FAKE_CONTROL_STATE'])
captures = Path(os.environ['FAKE_CAPTURE_ROOT'])
state = json.loads(state_path.read_text())
settings = state['settings']
args = sys.argv[1:]
requests = []
changed = False
capture = False
while args:
    arg = args.pop(0)
    if arg in ('--settings', '--status'):
        requests.append(arg)
    elif arg == '--shm':
        args.pop(0)
    elif arg == '--capture':
        assert args.pop(0) == '1'
        capture = True
        changed = True
    else:
        assert arg.startswith('--'), arg
        key = arg[2:]
        assert key in settings, key
        settings[key] = args.pop(0)
        changed = True
if changed:
    state['seq'] += 1
status_seq = state['seq']
state['calls'] = state.get('calls', []) + [sys.argv[1:]]

def png(path, pixels):
    def chunk(kind, value):
        return struct.pack('>I', len(value)) + kind + value + struct.pack('>I', zlib.crc32(kind + value) & 0xffffffff)
    rows = b''.join(b'\0' + bytes(sum(row, [])) for row in pixels)
    width, height = len(pixels[0]), len(pixels)
    path.write_bytes(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)) + chunk(b'IDAT', zlib.compress(rows)) + chunk(b'IEND', b''))

if capture:
    state['captures'] += 1
    number = state['captures']
    batch = captures / ('batch-%d' % number)
    batch.mkdir()
    before = [[[51, 51, 51], [102, 102, 102]], [[153, 153, 153], [204, 204, 204]]]
    amount = 0
    if float(settings['apply-model']):
        if settings['debug-view'] == '2':
            amount = 26 * int(settings['passes'])
        elif settings['debug-view'] == '0':
            amount = round(16 * int(settings['passes']) * float(settings['color']))
    after = [[[min(255, r + amount), g, b] for r, g, b in row] for row in before]
    png(batch / 'before_00.png', before)
    png(batch / 'after_00.png', after)
    seq = str(state['seq'])
    meta = {'frames': '1', 'width': '2', 'height': '2', 'vk_format': '37',
            'encoding': 'png', 'row_pitch': '8', 'bytes_per_pixel': '4',
            'capture_metadata_version': '1', 'capture_control_seq': seq,
            'frame_control_seq': seq, 'request_seq': str(number),
            'response_seq': str(number), 'ok_seq': str(number),
            'before_hash': 'same-held-input', 'batch_dir': batch.name}
    for key in ('passes', 'debug-view', 'apply-model', 'bypass', 'hold', 'detail', 'color'):
        meta[key.replace('-', '_')] = settings[key]
    if number == 2:
        if state['mode'] == 'stale':
            meta['capture_control_seq'] = str(state['seq'] - 1)
        elif state['mode'] == 'changed-input':
            meta['before_hash'] = 'another-frame'
        elif state['mode'] == 'failed-response':
            meta['ok_seq'] = '0'
        elif state['mode'] == 'generation-bump':
            # Another control generation raced the requested capture token. The
            # settings are unchanged; a second capture must use this generation.
            state['seq'] += 1
            meta['capture_control_seq'] = str(state['seq'])
            meta['frame_control_seq'] = str(state['seq'])
        elif state['mode'] == 'wrong-metadata':
            meta['apply_model'] = '1'
    content = ''.join('%s %s\n' % item for item in meta.items())
    (batch / 'manifest.txt').write_text(content)
    def publish():
        temporary = captures / ('manifest-%d.tmp' % number)
        temporary.write_text(content)
        temporary.replace(captures / 'manifest.txt')
    if state['mode'] != 'delayed':
        publish()
    elif os.fork() == 0:
        # Like the layer, publish frames after the control request returned.
        null = os.open(os.devnull, os.O_RDWR)
        for descriptor in (0, 1, 2):
            os.dup2(null, descriptor)
        time.sleep(0.2)
        publish()
        os._exit(0)

if '--settings' in requests and state['captures'] == 6 and state['mode'] == 'concurrent' and not state.get('concurrent_change'):
    settings['detail'] = '0.73'
    state['concurrent_change'] = True
state_path.write_text(json.dumps(state))
for request in requests:
    if request == '--settings':
        print('\n'.join('%s=%s' % item for item in settings.items()))
    else:
        print('initialised=1\nmodel_up=1\nlayer_composition_up=1\nhelper_reason=NATIVE')
        print('control_seq=%s' % status_seq)
'''


FAKE_WORKER = r'''
import ctypes
import json
import os
from pathlib import Path
import select
import struct
import sys

directory, state_path, shm = map(Path, sys.argv[1:4])
mode = sys.argv[4]
libc = ctypes.CDLL(None, use_errno=True)
fd = libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
assert fd >= 0, ctypes.get_errno()
assert libc.inotify_add_watch(fd, os.fsencode(directory), 0x8 | 0x80 | 0x100) >= 0
owner = {'trace_metadata_schema': 2, 'pid': os.getpid(), 'shm': str(shm)}
if mode == 'old-owner':
    del owner['trace_metadata_schema']
elif mode == 'wrong-channel':
    owner['shm'] += '.other'
elif mode == 'invalid-owner-pid':
    owner['pid'] = 0
(directory / 'owner.json').write_text(json.dumps(owner))
print('ready', flush=True)

def pfm(path, red_offset):
    # A 2x2 RGB image with four neutral shades; PFM rows run bottom to top.
    samples = []
    for shade in (0.6, 0.8, 0.2, 0.4):
        samples.extend((shade + red_offset, shade, shade))
    path.write_bytes(b'PF\n2 2\n-1.0\n' + struct.pack('<12f', *samples))

while True:
    request = directory / 'request'
    if not request.is_file():
        select.select([fd], [], [])
        try:
            os.read(fd, 65536)
        except BlockingIOError:
            pass
        continue
    token = request.read_text().strip()
    request.unlink()
    state = json.loads(state_path.read_text())
    settings = state['settings']
    passes = int(settings['passes'])
    metadata = {
        'trace_metadata_schema': 2, 'passes': passes,
        'held_input': int(settings['hold']), 'held_input_end': int(settings['hold']),
        'control_seq': state['seq'], 'control_seq_end': state['seq'],
        'tuning_seq': state['seq'], 'tuning_seq_end': state['seq'],
        'intensity': float(settings['intensity']),
        'local_tone': float(settings['local-tone']),
        'local_structure': float(settings['local-structure']),
        'color_preserve': float(settings['color-preserve']), 'sharpness': float(settings['sharpness']), 'motion': int(settings['mvec']),
        'source_proxy_hash': 'same-proxy-input',
    }
    if mode == 'missing-held-input':
        del metadata['held_input']
    elif mode == 'missing-held-end':
        del metadata['held_input_end']
    elif mode == 'old-summary':
        del metadata['trace_metadata_schema']
    elif mode == 'not-held':
        metadata['held_input'] = 0
    elif mode == 'released-input':
        metadata['held_input_end'] = 0
    elif mode == 'changed-proxy' and passes > 1:
        metadata['source_proxy_hash'] = 'changed-proxy-input'
    elif mode == 'changed-generation':
        metadata['control_seq_end'] += 1
    elif mode == 'changed-tuning':
        metadata['tuning_seq_end'] += 1
    status, error = ('failed', 'synthetic worker failure') if mode == 'failed-trace' else ('complete', '')
    trace = directory / token
    trace.mkdir()
    stages = []
    for number in range(1, passes + 1):
        for stage, offset in (('input', (number - 1) * 0.05), ('raw', number * 0.05)):
            name = 'pass-%02d-%s.pfm' % (number, stage)
            pfm(trace / name, offset)
            stages.append({'file': name})
    (trace / 'summary.json').write_text(json.dumps({
        'schema': 1, 'status': status, 'metadata': metadata, 'error': error, 'stages': stages,
    }))
    temporary = directory / (token + '.tmp')
    temporary.write_text('complete\n')
    temporary.replace(directory / (token + '.done'))
'''


class ColorOrchestratorTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.captures = self.directory / "captures"
        self.captures.mkdir()
        self.output = self.directory / "report"
        self.state = self.directory / "fake-state.json"
        self.control = self.directory / "fake-control"
        self.control.write_text("#!" + sys.executable + "\n" + FAKE_CONTROL)
        self.control.chmod(0o700)

    def start_worker(self, mode):
        trace_dir = self.directory / "traces"
        trace_dir.mkdir()
        worker = self.directory / "fake-worker.py"
        worker.write_text(FAKE_WORKER)
        process = subprocess.Popen([sys.executable, str(worker), str(trace_dir),
                                    str(self.state), str(self.directory / "shm.bin"), mode],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True)

        def stop():
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
            process.stdout.close()
            process.stderr.close()

        self.addCleanup(stop)
        self.assertTrue(select.select([process.stdout], [], [], 3)[0],
                        "fake worker did not publish ownership")
        self.assertEqual(process.stdout.readline().strip(), "ready")
        return trace_dir

    def invoke(self, mode, trace_mode=None, max_passes=1):
        self.state.write_text(json.dumps({"settings": dict(INITIAL), "seq": 100,
                                         "captures": 0, "mode": mode}))
        env = dict(os.environ, FAKE_CONTROL_STATE=str(self.state),
                   FAKE_CAPTURE_ROOT=str(self.captures))
        extra = []
        if trace_mode is not None:
            extra = ["--trace-dir", str(self.start_worker(trace_mode))]
        result = subprocess.run([sys.executable, str(PROGRAM),
                                 "--control", str(self.control),
                                 "--shm", str(self.directory / "shm.bin"),
                                 "--capture-dir", str(self.captures),
                                 "--output-dir", str(self.output),
                                 "--max-passes", str(max_passes), "--timeout", "1"] + extra,
                                env=env, text=True, capture_output=True, timeout=30)
        return result

    def current(self):
        return json.loads(self.state.read_text())["settings"]

    def outcome(self):
        return json.loads((self.output / "run.json").read_text())

    def test_success_records_red_shift_without_neural_correctness_verdict(self):
        result = self.invoke("success")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.current(), INITIAL)
        self.assertTrue(self.outcome()["completed"])
        self.assertTrue(self.outcome()["restored"])
        report = json.loads((self.output / "report.json").read_text())
        self.assertTrue(report["clean_frame_identity"]["passed"])
        model = report["comparisons"]["proxy_to_model-1"]
        composed = report["comparisons"]["original_to_composed-1"]
        self.assertGreater(model["red_excess_delta"], 0.09)
        self.assertGreater(composed["red_excess_delta"], 0.05)
        self.assertNotIn("passed", model)
        self.assertNotIn("passed", composed)
        self.assertNotIn("neural_correct", report)
        cases = json.loads((self.output / "captures.json").read_text())
        self.assertEqual(len(cases), 6)
        self.assertEqual(cases[0]["name"], "current")
        self.assertEqual(cases[0]["settings"]["intensity"], INITIAL["intensity"])
        for case in cases:
            self.assertTrue((self.output / case["directory"] / "manifest.txt").is_file())
        self.assertTrue((self.output / "report.md").is_file())
        # Batches move into the report instead of piling up in the capture directory.
        self.assertEqual([path.name for path in self.captures.iterdir()], ["manifest.txt"])

    def assert_ignores_earlier_manifest(self, sequence):
        # The public manifest survives the channel, whose generations restart.
        (self.captures / "earlier-batch").mkdir()
        (self.captures / "manifest.txt").write_text(
            f"capture_metadata_version 1\ncapture_control_seq {sequence}\n"
            f"frame_control_seq {sequence}\nbatch_dir earlier-batch\n")
        result = self.invoke("delayed")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(self.outcome()["completed"])
        self.assertEqual(self.current(), INITIAL)
        self.assertEqual(json.loads(self.state.read_text())["captures"], 6)
        cases = json.loads((self.output / "captures.json").read_text())
        self.assertNotIn("earlier-batch", {case["manifest"]["batch_dir"] for case in cases})

    def test_earlier_manifest_above_first_token_is_ignored(self):
        self.assert_ignores_earlier_manifest(103)

    def test_earlier_manifest_far_ahead_is_ignored(self):
        self.assert_ignores_earlier_manifest(5000)

    def test_running_diagnostic_refuses_before_reading_settings(self):
        with (self.directory / "color-test.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = self.invoke("success")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Another dlsslop-test is using this channel", result.stderr)
        self.assertFalse(self.output.exists())
        state = json.loads(self.state.read_text())
        self.assertEqual((state["seq"], state["captures"]), (100, 0))
        self.assertEqual(self.current(), INITIAL)
        self.assertFalse([call for call in state["calls"] if "--status" not in call])

    def test_help_states_each_default_once(self):
        text = subprocess.run([sys.executable, str(PROGRAM), "--help"], env=dict(os.environ, COLUMNS="500"),
                              text=True, capture_output=True, check=True).stdout
        self.assertNotIn("None", text)
        self.assertEqual(text.count("(default:"), 8, text)
        self.assertIn("--trace-dir (default: unset; no raw tensors)", text)
        self.assertIn("else /tmp/dlssnr-captures)", text)
        self.assertIn("invocation (default: 60)", text)

    def test_capture_directory_follows_the_layer(self):
        probe = ("import json, runpy, sys; "
                 "print(runpy.run_path(sys.argv[1])['capture_directory'](json.loads(sys.argv[2])))")

        def resolve(status, env):
            return subprocess.run([sys.executable, "-c", probe, str(PROGRAM), json.dumps(status)],
                                  env=env, text=True, capture_output=True, check=True).stdout.strip()

        terminal = {key: value for key, value in os.environ.items()
                    if key not in ("XDG_STATE_HOME", "HOME")}
        busy = dict(terminal, XDG_STATE_HOME="/terminal-state", HOME="/home/terminal")
        sleep = shutil.which("sleep")
        for game, expected in (
                ({"XDG_STATE_HOME": "/state", "HOME": "/home/game"}, "/state/dlssnr/captures"),
                ({"XDG_STATE_HOME": "", "HOME": "/home/game"}, "/home/game/.local/state/dlssnr/captures"),
                ({"HOME": "/home/game"}, "/home/game/.local/state/dlssnr/captures"),
                ({"XDG_STATE_HOME": "", "HOME": ""}, "/tmp/dlssnr-captures"),
                ({}, "/tmp/dlssnr-captures")):
            process = subprocess.Popen([sleep, "30"], env=game)
            try:
                with self.subTest(game=game):
                    self.assertEqual(resolve({"layer_pid": str(process.pid)}, busy), expected)
            finally:
                process.kill()
                process.wait()
        # An unreadable game environment falls back to this process's by the same rule.
        # /proc/0 never exists, so PID 0 is unreadable even to root or in a PID namespace.
        for status in ({}, {"layer_pid": "0"}, {"layer_pid": "not a pid"}):
            for env, expected in (
                    (busy, "/terminal-state/dlssnr/captures"),
                    (dict(terminal, XDG_STATE_HOME="", HOME="/home/terminal"),
                     "/home/terminal/.local/state/dlssnr/captures"),
                    (terminal, "/tmp/dlssnr-captures")):
                with self.subTest(status=status, env=expected):
                    self.assertEqual(resolve(status, env), expected)

    def test_concurrent_generation_retries_capture(self):
        result = self.invoke("generation-bump")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(self.outcome()["completed"])
        self.assertEqual(self.current(), INITIAL)
        state = json.loads(self.state.read_text())
        self.assertEqual(state["captures"], 7)
        cases = json.loads((self.output / "captures.json").read_text())
        self.assertEqual(len(cases), 6)
        for case in cases:
            meta = case["manifest"]
            self.assertEqual(meta["capture_control_seq"], meta["frame_control_seq"])

    def test_wrong_applied_setting_metadata_is_rejected(self):
        result = self.invoke("wrong-metadata")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("captured apply_model differs", result.stderr)
        self.assertFalse(self.outcome()["completed"])
        self.assertTrue(self.outcome()["restored"])
        self.assertEqual(self.current(), INITIAL)

    def test_stale_token_times_out_and_restores_settings(self):
        result = self.invoke("stale")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Timed out", result.stderr)
        self.assertFalse(self.outcome()["completed"])
        self.assertTrue(self.outcome()["restored"])
        self.assertEqual(self.current(), INITIAL)

    def test_changed_held_frame_is_rejected_and_restored(self):
        result = self.invoke("changed-input")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Held input changed", result.stderr)
        self.assertFalse(self.outcome()["completed"])
        self.assertTrue(self.outcome()["restored"])
        self.assertEqual(self.current(), INITIAL)

    def test_failed_inference_response_is_rejected_and_restored(self):
        result = self.invoke("failed-response")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing successful inference sequence", result.stderr)
        self.assertTrue(self.outcome()["restored"])
        self.assertEqual(self.current(), INITIAL)

    def test_concurrent_user_change_is_preserved(self):
        result = self.invoke("concurrent")
        self.assertNotEqual(result.returncode, 0)
        outcome = self.outcome()
        self.assertFalse(outcome["restored"])
        self.assertIn("detail", outcome["restore_conflicts"])
        expected = dict(INITIAL, detail="0.73")
        self.assertEqual(self.current(), expected)

    def test_raw_worker_traces_all_pass_counts(self):
        result = self.invoke("success", trace_mode="success", max_passes=3)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.current(), INITIAL)
        outcome = self.outcome()
        self.assertTrue(outcome["completed"])
        self.assertTrue(outcome["restored"])
        self.assertEqual(outcome["raw_traces"],
                         ["raw-1-pass", "raw-2-pass", "raw-3-pass"])
        self.assertFalse(outcome["failed_raw_traces"])
        report = json.loads((self.output / "report.json").read_text())
        self.assertTrue(report["completed"])
        for passes in range(1, 4):
            name = f"raw-{passes}-pass"
            trace = report["raw_traces"][name]
            self.assertEqual(trace["metadata"]["trace_metadata_schema"], 2)
            self.assertEqual(len(trace["per_pass_changes"]), passes)
            for comparison in trace["per_pass_changes"].values():
                self.assertAlmostEqual(comparison["red_excess_delta"], 0.05, places=6)
            self.assertTrue((self.output / name / f"pass-{passes:02}-raw.pfm").is_file())
        # Traces move into the report with their completion markers removed.
        self.assertEqual([path.name for path in (self.directory / "traces").iterdir()],
                         ["owner.json"])

    def assert_preflight_failure(self, mode, message):
        result = self.invoke("success", trace_mode=mode)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(message, result.stderr.lower())
        self.assertEqual(self.current(), INITIAL)
        state = json.loads(self.state.read_text())
        self.assertEqual(state["seq"], 100)
        self.assertEqual(state["captures"], 0)
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.captures.iterdir()), [])
        self.assertFalse((self.directory / "traces" / "request").exists())

    def test_old_owner_rejected_before_mutation_or_output_creation(self):
        self.assert_preflight_failure("old-owner", "incompatible")

    def test_wrong_trace_channel_rejected_before_mutation(self):
        self.assert_preflight_failure("wrong-channel", "another shared-memory channel")

    def test_zero_owner_pid_rejected_before_mutation(self):
        self.assert_preflight_failure("invalid-owner-pid", "pid")

    def assert_failed_raw_trace(self, mode, message, passes=1):
        result = self.invoke("success", trace_mode=mode, max_passes=passes)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(message, result.stderr.lower())
        self.assertEqual(self.current(), INITIAL)
        outcome = self.outcome()
        self.assertFalse(outcome["completed"])
        self.assertTrue(outcome["restored"])
        name = f"raw-{passes}-pass"
        self.assertIn(name, outcome["failed_raw_traces"])
        retained = self.output / name
        self.assertTrue((retained / "summary.json").is_file())
        self.assertTrue((retained / f"pass-{passes:02}-input.pfm").is_file())
        self.assertTrue((retained / f"pass-{passes:02}-raw.pfm").is_file())
        self.assertTrue((self.output / "report.md").is_file())
        report = json.loads((self.output / "report.json").read_text())
        self.assertFalse(report["completed"])
        self.assertIn(message, report["error"].lower())
        self.assertIn(f"proxy_to_model-{passes}", report["comparisons"])
        self.assertNotIn(name, report["raw_traces"])
        return result, report

    def test_missing_hold_start_retains_evidence_and_partial_report(self):
        result, _ = self.assert_failed_raw_trace("missing-held-input", "incompatible")
        self.assertNotIn("input was not held", result.stderr)

    def test_missing_hold_end_is_incompatible_metadata(self):
        result, _ = self.assert_failed_raw_trace("missing-held-end", "incompatible")
        self.assertNotIn("input was not held", result.stderr)

    def test_failed_trace_reports_the_worker_error(self):
        self.assert_failed_raw_trace("failed-trace", "synthetic worker failure")

    def test_old_trace_metadata_retains_evidence(self):
        self.assert_failed_raw_trace("old-summary", "incompatible")

    def test_trace_not_held_rejects_and_restores(self):
        self.assert_failed_raw_trace("not-held", "input was not held")

    def test_trace_released_during_inference_rejects_and_restores(self):
        self.assert_failed_raw_trace("released-input", "input was not held")

    def test_changed_trace_generation_rejects_and_restores(self):
        self.assert_failed_raw_trace("changed-generation", "settings changed during raw trace")

    def test_changed_trace_tuning_rejects_and_restores(self):
        self.assert_failed_raw_trace("changed-tuning", "tuning changed during raw trace")

    def test_changed_multipass_proxy_keeps_first_valid_trace(self):
        _, report = self.assert_failed_raw_trace(
            "changed-proxy", "input proxy changed between pass counts", passes=2)
        self.assertEqual(self.outcome()["raw_traces"], ["raw-1-pass"])
        self.assertIn("raw-1-pass", report["raw_traces"])


if __name__ == "__main__":
    unittest.main()
