#!/usr/bin/env python3
"""Exercise source-only imports and filtered submodules with local Git servers."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def command(*args, cwd, check=True, env=None):
    settings = os.environ.copy()
    settings.update({'GIT_AUTHOR_NAME': 'Fixture', 'GIT_AUTHOR_EMAIL': 'fixture@example.invalid',
                     'GIT_COMMITTER_NAME': 'Fixture', 'GIT_COMMITTER_EMAIL': 'fixture@example.invalid',
                     'GIT_TERMINAL_PROMPT': '0', 'GIT_CONFIG_NOSYSTEM': '1',
                     'GIT_CONFIG_GLOBAL': os.devnull})
    settings.update(env or {})
    result = subprocess.run(args, cwd=cwd, env=settings, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    if check and result.returncode:
        raise AssertionError(f'{args}\n{result.stdout}\n{result.stderr}')
    return result


def git(*args, cwd, check=True):
    return command('git', *args, cwd=cwd, check=check).stdout.strip()


class SubmodulesTest(unittest.TestCase):
    def test_offline_import_filtered_checkout_and_prepare(self):
        with tempfile.TemporaryDirectory(prefix='dlsslop-amd-submodules-') as temporary:
            base = Path(temporary)
            upstream = base / 'upstream'
            upstream.mkdir()
            git('init', '--quiet', cwd=upstream)
            sources = {'src/layer.txt': b'original\n', 'src/kernel.hip': b'kernel\n',
                       'vendor/api.h': b'api\n'}
            for name, data in sources.items():
                target = upstream / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
            # Incompressible, unrelated content makes accidental full fetches
            # observable both in object membership and in pack byte counts.
            weights = upstream / 'weights'
            weights.mkdir()
            (weights / 'model.bin').write_bytes(os.urandom(2 * 1024 * 1024))
            git('add', '-A', cwd=upstream)
            git('commit', '--quiet', '-m', 'Pinned sources and unrelated weights', cwd=upstream)
            pin = git('rev-parse', 'HEAD', cwd=upstream)
            weight_oid = git('rev-parse', 'HEAD:weights/model.bin', cwd=upstream)
            (upstream / 'src/layer.txt').write_text('newer tip must not be checked out\n')
            git('commit', '--quiet', '-a', '-m', 'A newer unpinned tip', cwd=upstream)
            tip = git('rev-parse', 'HEAD', cwd=upstream)
            server = base / 'server.git'
            git('clone', '--quiet', '--bare', str(upstream), str(server), cwd=base)
            git('config', 'uploadpack.allowFilter', 'true', cwd=server)
            mirror = server.as_uri()

            project = base / 'source-import'
            (project / 'scripts').mkdir(parents=True)
            for name in ('init-repo.py', 'fetch-submodules.py', 'submodule_support.py', 'prepare-sources.py',
                         'update-source-patch.py'):
                shutil.copyfile(ROOT / 'scripts' / name, project / 'scripts' / name)
            (project / '.gitmodules').write_text('[submodule "amd"]\n\tpath = external/amd\n'
                                                f'\turl = {mirror}\n')
            (project / '.gitignore').write_text('__pycache__/\n/upstream-layer/\n/kernels/\n/backend/vendor/\n'
                                                '/.prepared-sources.json\n')
            targets = {'upstream-layer/example.txt': 'src/layer.txt',
                       'kernels/example.hip': 'src/kernel.hip', 'backend/vendor/api.h': 'vendor/api.h'}
            files = {name: {'repository': 'amd', 'source': source,
                            'sha256': hashlib.sha256(sources[source]).hexdigest()}
                     for name, source in targets.items()}
            lock = {'version': 1, 'repositories': {'amd': {'path': 'external/amd',
                    'url': 'https://original-provenance.invalid/upstream.git', 'commit': pin}},
                    'files': files}
            (project / 'upstreams.lock.json').write_text(json.dumps(lock))
            (project / 'patches').mkdir()
            (project / 'patches/linux-integration.patch').write_text(
                'diff --git a/upstream-layer/example.txt b/upstream-layer/example.txt\n'
                'old mode 100644\nnew mode 100755\n'
                '--- a/upstream-layer/example.txt\n+++ b/upstream-layer/example.txt\n'
                '@@ -1 +1 @@\n-original\n+patched\n')

            # No network or upstream objects are needed to create and push the
            # superproject. Move the server away to prove initialization is offline.
            offline_server = base / 'offline.git'
            server.rename(offline_server)
            git('init', '--quiet', '-b', 'main', cwd=project)
            command(sys.executable, 'scripts/init-repo.py', cwd=project)
            git('add', '-A', cwd=project)
            self.assertEqual(git('ls-files', '--stage', 'external/amd', cwd=project),
                             f'160000 {pin} 0\texternal/amd')
            self.assertFalse((project / '.git/modules').exists())
            command(sys.executable, 'scripts/init-repo.py', cwd=project)  # idempotent
            git('commit', '--quiet', '-m', 'Source import', cwd=project)
            own_remote = base / 'own-remote.git'
            git('init', '--quiet', '--bare', str(own_remote), cwd=base)
            git('symbolic-ref', 'HEAD', 'refs/heads/main', cwd=own_remote)
            git('remote', 'add', 'origin', own_remote.as_uri(), cwd=project)
            git('push', '--quiet', '-u', 'origin', 'main', cwd=project)
            clone = base / 'fresh-clone'
            git('clone', '--quiet', own_remote.as_uri(), str(clone), cwd=base)
            self.assertEqual(git('ls-tree', 'HEAD', 'external/amd', cwd=clone),
                             f'160000 commit {pin}\texternal/amd')
            self.assertFalse((clone / '.git/modules').exists())
            # Before the fetch, both staging tools stop, name the missing step
            # and leave the prepared tree and the patch alone.
            for script in ('prepare-sources.py', 'update-source-patch.py'):
                unfetched = command(sys.executable, 'scripts/' + script, cwd=clone, check=False)
                self.assertNotEqual(unfetched.returncode, 0)
                self.assertIn('run python3 scripts/fetch-submodules.py', unfetched.stderr)
            self.assertFalse((clone / 'upstream-layer').exists())
            self.assertEqual(git('status', '--porcelain', cwd=clone), '')
            offline_server.rename(server)

            # A host without filtering must fail before making a submodule or
            # transferring any commit/blob objects.
            git('config', 'uploadpack.allowFilter', 'false', cwd=server)
            unsupported = command(sys.executable, 'scripts/fetch-submodules.py', cwd=clone, check=False)
            self.assertNotEqual(unsupported.returncode, 0)
            self.assertIn('no unfiltered fetch attempted', unsupported.stderr)
            self.assertFalse((clone / '.git/modules/amd').exists())
            # Independently guard the real fetch connection in case a server
            # changes capabilities after the successful refs-only preflight.
            spec = importlib.util.spec_from_file_location('submodule_support', ROOT / 'scripts/submodule_support.py')
            support = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(support)
            receiver = base / 'rejected-fetch'
            receiver.mkdir()
            git('init', '--quiet', cwd=receiver)
            with self.assertRaisesRegex(RuntimeError, 'server dropped blob-filter support'):
                support.git('-c', 'protocol.version=2', 'fetch', '--depth=1',
                            '--filter=blob:none', mirror, pin, cwd=receiver, timeout=30)
            git('config', 'uploadpack.allowFilter', 'true', cwd=server)
            command(sys.executable, 'scripts/fetch-submodules.py', '-t', '30', cwd=clone)
            module = clone / 'external/amd'
            self.assertEqual(git('rev-parse', 'HEAD', cwd=module), pin)
            self.assertEqual(git('rev-list', '--count', 'HEAD', cwd=module), '1')
            self.assertEqual(git('rev-parse', '--is-shallow-repository', cwd=module), 'true')
            self.assertTrue((clone / '.git/modules/amd').is_dir())
            self.assertTrue((module / '.git').is_file())
            self.assertFalse(Path((module / '.git').read_text().strip()[8:]).is_absolute())
            self.assertTrue(git('submodule', 'status', cwd=clone).startswith(pin))
            self.assertEqual({path.relative_to(module).as_posix() for path in module.rglob('*')
                              if path.is_file() and path.name != '.git'}, set(sources))
            # --batch-all-objects inspects local storage without fetching missing
            # promisor objects, unlike cat-file -e on a missing object ID.
            objects = git('cat-file', '--batch-all-objects', '--batch-check=%(objectname)', cwd=module).splitlines()
            self.assertNotIn(weight_oid, objects)
            self.assertNotIn(tip, objects)
            self.assertLess(sum(path.stat().st_size for path in (clone / '.git/modules/amd/objects').rglob('*.pack')),
                            128 * 1024)
            self.assertEqual(git('config', '--get', 'remote.origin.url', cwd=module), mirror)
            # User Git configuration, attributes and GIT_* variables must not
            # reach the staging repository, where they break or corrupt the patch.
            hostile_home = base / 'hostile-home'
            (hostile_home / 'git').mkdir(parents=True)
            (hostile_home / 'git/attributes').write_text('* text eol=crlf -diff\n')
            (hostile_home / 'gitconfig').write_text('[core]\n\tautocrlf = true\n[diff]\n\tnoprefix = true\n'
                                                    '[color]\n\tui = always\n[apply]\n\twhitespace = error\n'
                                                    '[commit]\n\tgpgsign = true\n[gpg]\n\tprogram = false\n')
            hostile_index = base / 'hostile-index'
            # A temporary directory inside the work tree must not let the outer
            # repository capture the staging repository's Git commands.
            nested_temporary = clone / 'nested-temporary'
            nested_temporary.mkdir()
            hostile = {'GIT_CONFIG_GLOBAL': str(hostile_home / 'gitconfig'), 'XDG_CONFIG_HOME': str(hostile_home),
                       'GIT_CONFIG_PARAMETERS': "'diff.noprefix'='true' 'core.autocrlf'='true'",
                       'GIT_INDEX_FILE': str(hostile_index), 'TMPDIR': str(nested_temporary)}
            command(sys.executable, 'scripts/prepare-sources.py', cwd=clone, env=hostile)
            self.assertEqual((clone / 'upstream-layer/example.txt').read_bytes(), b'patched\n')
            self.assertEqual((clone / 'kernels/example.hip').read_bytes(), b'kernel\n')
            self.assertEqual((clone / 'backend/vendor/api.h').read_bytes(), b'api\n')
            # The patch alone carries file modes into the prepared tree.
            self.assertTrue(os.access(clone / 'upstream-layer/example.txt', os.X_OK))
            self.assertFalse(os.access(clone / 'kernels/example.hip', os.X_OK))
            committed = {name: (clone / name).read_bytes()
                         for name in ('upstreams.lock.json', 'patches/linux-integration.patch')}
            command(sys.executable, 'scripts/update-source-patch.py', cwd=clone, env=hostile)
            patch = (clone / 'patches/linux-integration.patch').read_text()
            self.assertTrue(patch.startswith('diff --git a/upstream-layer/example.txt b/upstream-layer/example.txt\n'
                                             'old mode 100644\nnew mode 100755\n'))
            self.assertNotIn('\x1b', patch)
            self.assertIn('\n-original\n+patched\n', patch)
            self.assertFalse(hostile_index.exists())
            nested_temporary.rmdir()
            self.assertEqual((clone / 'upstreams.lock.json').read_bytes(), committed['upstreams.lock.json'])
            # A symlink would be recorded as its target's content.
            (clone / 'kernels/link.hip').symlink_to('example.hip')
            linked = command(sys.executable, 'scripts/update-source-patch.py', cwd=clone, check=False)
            self.assertNotEqual(linked.returncode, 0)
            self.assertIn('symlink in prepared source', linked.stderr)
            self.assertEqual((clone / 'patches/linux-integration.patch').read_text(), patch)
            (clone / 'kernels/link.hip').unlink()
            command(sys.executable, 'scripts/prepare-sources.py', cwd=clone)
            self.assertEqual((clone / 'upstream-layer/example.txt').read_bytes(), b'patched\n')
            for name, data in committed.items():
                (clone / name).write_bytes(data)

            # A changed patch and lock update an unmodified prepared tree and
            # remove the files they no longer produce; local edits stay refused.
            changed = json.loads(json.dumps(lock))
            del changed['files']['backend/vendor/api.h']
            (clone / 'upstreams.lock.json').write_text(json.dumps(changed))
            (clone / 'patches/linux-integration.patch').write_bytes(
                committed['patches/linux-integration.patch'].replace(b'+patched\n', b'+patched again\n'))
            (clone / 'backend/vendor/api.h').write_text('local edit\n')
            edited = command(sys.executable, 'scripts/prepare-sources.py', cwd=clone, check=False)
            self.assertNotEqual(edited.returncode, 0)
            self.assertIn('local changes', edited.stderr)
            self.assertEqual((clone / 'backend/vendor/api.h').read_text(), 'local edit\n')
            (clone / 'backend/vendor/api.h').write_bytes(sources['vendor/api.h'])
            command(sys.executable, 'scripts/prepare-sources.py', cwd=clone)
            self.assertEqual((clone / 'upstream-layer/example.txt').read_bytes(), b'patched again\n')
            self.assertEqual((clone / 'kernels/example.hip').read_bytes(), b'kernel\n')
            self.assertFalse((clone / 'backend/vendor/api.h').exists())
            for name, data in committed.items():
                (clone / name).write_bytes(data)
            (clone / 'kernels/example.hip').write_text('local edit\n')
            edited = command(sys.executable, 'scripts/prepare-sources.py', cwd=clone, check=False)
            self.assertNotEqual(edited.returncode, 0)
            self.assertIn('local changes', edited.stderr)
            self.assertEqual((clone / 'kernels/example.hip').read_text(), 'local edit\n')
            self.assertEqual((clone / 'upstream-layer/example.txt').read_bytes(), b'patched again\n')
            (clone / 'kernels/example.hip').write_bytes(sources['src/kernel.hip'])
            command(sys.executable, 'scripts/prepare-sources.py', cwd=clone)
            self.assertEqual((clone / 'upstream-layer/example.txt').read_bytes(), b'patched\n')
            self.assertEqual((clone / 'backend/vendor/api.h').read_bytes(), b'api\n')
            # A symlink in a prepared tree is refused, never written through, even
            # when its target holds exactly what the record says was prepared.
            outside = clone.parent / 'outside.txt'
            outside.write_bytes(b'patched\n')
            (clone / 'upstream-layer/example.txt').unlink()
            (clone / 'upstream-layer/example.txt').symlink_to(outside)
            (clone / 'patches/linux-integration.patch').write_bytes(
                committed['patches/linux-integration.patch'].replace(b'+patched\n', b'+patched again\n'))
            linked = command(sys.executable, 'scripts/prepare-sources.py', cwd=clone, check=False)
            self.assertNotEqual(linked.returncode, 0)
            self.assertIn('symlink in prepared source', linked.stderr)
            self.assertEqual(outside.read_bytes(), b'patched\n')
            (clone / 'upstream-layer/example.txt').unlink()
            (clone / 'upstream-layer/example.txt').write_bytes(b'patched\n')
            (clone / 'patches/linux-integration.patch').write_bytes(committed['patches/linux-integration.patch'])
            command(sys.executable, 'scripts/prepare-sources.py', cwd=clone)
            self.assertEqual((clone / 'upstream-layer/example.txt').read_bytes(), b'patched\n')

            # Local URL overrides take precedence over .gitmodules and do not
            # alter provenance recorded in the lock. Repeated fetches are safe.
            second_mirror = base / 'second-mirror.git'
            shutil.copytree(server, second_mirror)
            git('config', 'submodule.amd.url', second_mirror.as_uri(), cwd=clone)
            command(sys.executable, 'scripts/fetch-submodules.py', cwd=clone)
            self.assertEqual(git('config', '--get', 'remote.origin.url', cwd=module), second_mirror.as_uri())
            self.assertEqual(json.loads((clone / 'upstreams.lock.json').read_text()), lock)
            git('add', '-A', cwd=clone)
            self.assertEqual(git('diff', '--cached', '--name-only', cwd=clone), '')

            (module / 'src/layer.txt').write_text('local edit\n')
            dirty = command(sys.executable, 'scripts/fetch-submodules.py', cwd=clone, check=False)
            self.assertNotEqual(dirty.returncode, 0)
            self.assertIn('local changes', dirty.stderr)
            self.assertEqual((module / 'src/layer.txt').read_text(), 'local edit\n')
            (module / 'src/layer.txt').write_bytes(sources['src/layer.txt'])
            (module / 'untracked.txt').write_text('keep me\n')
            dirty = command(sys.executable, 'scripts/fetch-submodules.py', cwd=clone, check=False)
            self.assertNotEqual(dirty.returncode, 0)
            self.assertIn('local changes', dirty.stderr)
            self.assertEqual((module / 'untracked.txt').read_text(), 'keep me\n')
            (module / 'untracked.txt').unlink()
            (clone / '.git/modules/amd/info/exclude').write_text('ignored.txt\n')
            (module / 'ignored.txt').write_text('also keep me\n')
            dirty = command(sys.executable, 'scripts/fetch-submodules.py', cwd=clone, check=False)
            self.assertNotEqual(dirty.returncode, 0)
            self.assertIn('local changes', dirty.stderr)
            self.assertEqual((module / 'ignored.txt').read_text(), 'also keep me\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('-h', '--help', action='help', help='show help (default: off)')
    parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SubmodulesTest))
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(main())
