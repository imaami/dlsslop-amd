#!/usr/bin/env python3
"""Register pinned submodules offline after git init; no upstream download."""
import argparse
import subprocess
import sys

from submodule_support import ROOT, check_gitlink, git, inspect_worktree, load_repositories


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False,
                                     epilog='Stages only pinned gitlinks. Run git add -A separately for source files.')
    parser.add_argument('-h', '--help', action='help', help='show help (default: off)')
    parser.parse_args()
    try:
        repositories = load_repositories()
        for name, row in repositories.items():
            check_gitlink(row, required=False)
            if inspect_worktree(name, row):
                head = git('rev-parse', '--verify', 'HEAD', cwd=ROOT / row['path']).stdout.strip()
                if head != row['commit']:
                    raise RuntimeError(f"wrong submodule revision at {row['path']}")
        for row in repositories.values():
            # Git preserves an uninitialized gitlink during git add -A only
            # when its empty directory exists. No placeholder files are needed.
            (ROOT / row['path']).mkdir(parents=True, exist_ok=True)
            git('update-index', '--add', '--cacheinfo',
                f"160000,{row['commit']},{row['path']}")
        print(f'Registered {len(repositories)} pinned gitlinks offline. '
              'Run git add -A separately to stage source files.')
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.TimeoutExpired) as exc:
        print(f'init-repo: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
