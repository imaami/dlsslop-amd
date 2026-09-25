#!/usr/bin/env python3
"""Fetch exact pinned submodules with shallow history and sparse source files."""
import argparse
import subprocess
import sys

from submodule_support import (ROOT, check_gitlink, git, inspect_worktree,
                               load_repositories, module_gitdir)


def require_filter(url, timeout):
    # ls-remote transfers capabilities and refs, never an object pack. Git
    # otherwise ignores unsupported --filter and may download large weights.
    probe = git('-c', 'protocol.version=2', 'ls-remote', '--', url, 'HEAD',
                timeout=timeout, env={'GIT_TRACE_PACKET': '1'})
    for line in probe.stderr.splitlines():
        if 'ls-remote< fetch=' in line:
            capabilities = line.split('ls-remote< fetch=', 1)[1].split()
            if 'filter' in capabilities:
                return
    raise RuntimeError('upstream does not advertise Git protocol-v2 blob filtering; '
                       'configure a filter-capable mirror with '
                       'git config submodule.<name>.url URL; no unfiltered fetch attempted')


def sparse_pattern(path):
    # Non-cone patterns select the exact mapped files, including dotfiles,
    # without fetching every sibling in large upstream directories.
    return '/' + ''.join('\\' + char if char in '\\*?[]!#' else char for char in path)


def fetch(name, row, initialized, timeout):
    path = ROOT / row['path']
    git('submodule', 'init', '--', row['path'])
    # Git resolves relative .gitmodules URLs against the superproject remote;
    # submodule init also preserves explicit local submodule.<name>.url values.
    url = git('config', '--get', f'submodule.{name}.url').stdout.strip()
    print(f"{name}: checking filtered-fetch support", flush=True)
    require_filter(url, timeout)
    if not initialized:
        gitdir = module_gitdir(name)
        if gitdir.exists():
            raise RuntimeError(f'orphan submodule Git directory at {gitdir}; '
                               'restore its worktree before retrying')
        path.mkdir(parents=True, exist_ok=True)
        git('init', '--quiet', str(path))
        git('submodule', 'absorbgitdirs', '--', row['path'])
    git('config', 'remote.origin.url', url, cwd=path)
    git('config', 'remote.origin.promisor', 'true', cwd=path)
    git('config', 'remote.origin.partialclonefilter', 'blob:none', cwd=path)
    # Set sparse selection before either fetching or checking out the commit.
    git('sparse-checkout', 'set', '--no-cone', '--stdin', cwd=path,
        input='\n'.join(sparse_pattern(source) for source in row['sources']) + '\n',
        timeout=timeout)
    print(f"{name}: fetching pinned {row['commit']} ({len(row['sources'])} source paths)", flush=True)
    git('-c', 'protocol.version=2', 'fetch', '--depth=1', '--filter=blob:none',
        '--no-tags', '--no-recurse-submodules', 'origin', row['commit'],
        cwd=path, timeout=timeout)
    git('checkout', '--quiet', '--detach', '--no-recurse-submodules', row['commit'],
        cwd=path, timeout=timeout)
    if git('rev-parse', 'HEAD', cwd=path).stdout.strip() != row['commit']:
        raise RuntimeError(f'wrong pinned checkout for {name}')
    missing = [source for source in row['sources'] if not (path / source).is_file()]
    if missing:
        raise RuntimeError(f'missing mapped source files in {name}: {missing}')
    print(f'{name}: ready', flush=True)


def positive_seconds(value):
    seconds = int(value)
    if seconds <= 0:
        raise argparse.ArgumentTypeError('timeout must be positive')
    return seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False,
                                     epilog='Default: all repositories in upstreams.lock.json; no full-clone fallback.')
    parser.add_argument('-t', '--timeout', type=positive_seconds, default=300,
                        help='timeout per network or checkout operation, in seconds (default: %(default)s)')
    parser.add_argument('-h', '--help', action='help', help='show help (default: off)')
    args = parser.parse_args()
    try:
        repositories = load_repositories()
        initialized = {}
        # Inspect every worktree before making any changes to dependencies.
        for name, row in repositories.items():
            check_gitlink(row)
            initialized[name] = inspect_worktree(name, row, timeout=args.timeout)
        for name, row in repositories.items():
            fetch(name, row, initialized[name], args.timeout)
        print('Pinned source dependencies are ready. Run python3 scripts/prepare-sources.py.')
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.TimeoutExpired) as exc:
        print(f'fetch-submodules: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
