#!/usr/bin/env python3
"""Package freshly compiled runtime files, notices and installation instructions."""
import argparse
import importlib.util
import io
import os
from pathlib import Path
import re
import sys
import tarfile
import tempfile
import time
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]


def load_installer(root):
    spec = importlib.util.spec_from_file_location("dlsslop_installer", root / "install.py")
    installer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(installer)
    return installer


def ci_source_url():
    repository, revision = os.environ.get("GITHUB_REPOSITORY"), os.environ.get("GITHUB_SHA")
    if not repository and not revision:
        return None
    if not (repository and revision and re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository)
            and re.fullmatch(r"[0-9a-fA-F]{40,64}", revision)):
        raise ValueError("GITHUB_REPOSITORY and GITHUB_SHA must identify the exact source revision")
    server = os.environ.get("GITHUB_SERVER_URL", "https://github.com").rstrip("/")
    if urlparse(server).scheme != "https" or not urlparse(server).netloc:
        raise ValueError("GITHUB_SERVER_URL must be an HTTPS URL")
    return f"{server}/{repository}/tree/{revision}"


def package(root, build, output, source_url):
    parsed = urlparse(source_url or "")
    if (parsed.scheme not in ("https", "http", "file")
            or (parsed.scheme in ("https", "http") and not parsed.netloc)
            or (parsed.scheme == "file" and not parsed.path.startswith("/"))
            or any(character in (source_url or "") for character in "\r\n")):
        raise ValueError("provide --source-url for the exact corresponding source, or set GITHUB_REPOSITORY and GITHUB_SHA")
    installer = load_installer(root)
    # Only the installer's allowlist can enter the archive: the checkout and
    # build directory are never enumerated, whatever else they contain.
    entries = installer.tree(root, build, (
        "dlsslop-amd corresponding source\n"
        "==============================\n\n"
        f"Exact source for this release: {source_url}\n\n"
        "This source includes the integration code, dependency pins and build\n"
        "instructions. The pinned dependencies are available through the\n"
        "repository URLs recorded in upstreams.lock.json at that revision. See\n"
        "THIRD-PARTY.txt and the accompanying licenses for component attribution.\n"
        "Model weights are a separate dependency and are not included.\n"))
    # The archive root is the installation prefix. It holds no directory
    # entries, so extracting over ~/.local never changes existing directories.
    stamp = int(os.environ.get("SOURCE_DATE_EPOCH") or time.time())
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=output.name + ".", dir=output.parent)
    os.close(descriptor)
    try:
        with tarfile.open(temporary, "w:xz") as archive:
            for name, (content, mode) in sorted(entries.items()):
                info = tarfile.TarInfo(name)
                info.size, info.mode, info.mtime = len(content), mode, stamp
                archive.addfile(info, io.BytesIO(content))
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(f"Packaged 4 native binaries, {len(installer.MODULE_NAMES)} GPU modules and runtime tools: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False, allow_abbrev=False)
    parser.add_argument("-b", "--build-dir", type=Path, default=ROOT / "build",
                        help="fresh native build directory (default: %(default)s)")
    parser.add_argument("-o", "--output", type=Path, default=ROOT / "dist/dlsslop-amd-linux-gfx1201.tar.xz",
                        help="binary release archive (default: %(default)s)")
    parser.add_argument("-s", "--source-url",
                        help="exact corresponding source URL (default: GITHUB_SERVER_URL/GITHUB_REPOSITORY/tree/GITHUB_SHA "
                             "in CI, with server https://github.com; otherwise unset and required)")
    parser.add_argument("-h", "--help", action="help", help="show this help and exit (default: off)")
    args = parser.parse_args()
    try:
        package(ROOT, args.build_dir.expanduser().resolve(), args.output.expanduser().resolve(),
                args.source_url or ci_source_url())
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
