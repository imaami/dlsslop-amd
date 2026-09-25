#!/usr/bin/python3
"""Import the model from a user-owned upstream release ZIP or asset directory.

The upstream model is not covered by the code's MIT license. This program does
not fetch or execute NVIDIA DLLs. It imports only the precise floating-point
weight files required by the native runtime, rejecting incomplete/ambiguous
packages and recording SHA-256 hashes. Download the current full Magpie or
OptiScaler package from the links in the upstream README yourself; pass its ZIP
here. Windows runtime components in the archive are not extracted.
"""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import os
import pwd
import shutil
import sys
import tempfile
import zipfile


def weight_manifest():
    """Map runtime base names to element counts (Network::WeightElements)."""
    result = {"head-matrix": 524288, "decoder39-weights": 524800,
              "post70-scales": 64, "post70-head": 96,
              "post70-ffn": 8736, "post70-attention": 8225}
    for block in list(range(31)) + list(range(40, 70)):
        c = (32 if block <= 4 else 64 if block <= 8 else 128 if block <= 14
             else 256 if block <= 22 else 512 if block <= 47 else 256 if block <= 55
             else 128 if block <= 61 else 64 if block <= 65 else 32)
        if c == 512:
            result[f"block{block}-ffwd"] = 524288
            result[f"block{block}-ffwd-projection"] = 262656
        else:
            result[f"block{block}-ffn"] = 8736 if c == 32 else 9 * c * c + c
        result[f"block{block}-attention"] = 8225 if c == 32 else 4 * c * c + (c // 32) * 4096 + c // 32 + c
    for block in range(31, 39):
        for part, count in [("expand", 4194304), ("contract", 4195328), ("qkv", 3145760), ("projection", 1049600)]:
            result[f"block{block}-{part}"] = count
    for block, c in [(4, 32), (8, 64), (14, 128), (22, 256)]:
        result[f"block{block}-ds"] = 2 * c * c
    for block, c in [(48, 256), (56, 128), (62, 64), (66, 32)]:
        result[f"block{block}-weights"] = 2 * c * c + c
    return dict(sorted(result.items()))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as src:
        while data := src.read(1024 * 1024):
            digest.update(data)
    return digest.hexdigest()


def validate_asset_dir(path):
    found = []
    missing = []
    for base, count in weight_manifest().items():
        options = [path / (base + ext) for ext in [".f32", ".f16"]]
        present = [p for p in options if p.is_file()]
        if not present:
            missing.append(base)
            continue
        # Runtime gives f32 precedence; reject either malformed representation.
        for item in present:
            expected = count * (4 if item.suffix == ".f32" else 2)
            if item.stat().st_size != expected:
                raise ValueError(f"{item}: expected {expected} bytes, found {item.stat().st_size}")
        item = present[0]
        found.append({"file": item.name, "elements": count, "bytes": item.stat().st_size, "sha256": sha256(item)})
    if missing:
        raise ValueError(f"missing {len(missing)} weights: {', '.join(missing)}")
    return found


def import_assets(args):
    if args.print_manifest:
        print(json.dumps(weight_manifest(), indent=2))
        return
    if args.check:
        records = validate_asset_dir(args.check)
        print(f"Complete model: {len(records)} weights; {sum(x['bytes'] for x in records):,} bytes")
        return
    if not args.source:
        raise ValueError("provide --source ZIP_OR_DIRECTORY, or --check ASSET_DIRECTORY")
    source = args.source.resolve()
    output = args.output.resolve()
    if source == output:
        raise ValueError("source and output must be different directories; use --check instead")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        # Preserve independently built HIP subdirectory; never overwrite weights.
        names = {base + ext for base in weight_manifest() for ext in [".f16", ".f32"]}
        if any((output / name).exists() for name in names):
            raise ValueError(f"weights already exist in {output}; choose a new output or use --check")
    wanted = {base + ext for base in weight_manifest() for ext in [".f16", ".f32"]}
    provenance = {"source": str(source), "source_sha256": sha256(source) if source.is_file() else None}
    with tempfile.TemporaryDirectory(prefix=".dlsslop-amd-assets-", dir=output.parent) as temp:
        stage = Path(temp)
        if source.is_dir():
            candidates = [p for p in source.rglob("*") if p.name in wanted and p.is_file()]
            for item in candidates:
                if item.is_symlink():
                    raise ValueError(f"refusing symlink weight: {item}")
                target = stage / item.name
                if target.exists():
                    raise ValueError(f"ambiguous duplicate weight: {item.name}; pass the exact native-game-tiled-assets directory")
                shutil.copyfile(item, target)
        else:
            with zipfile.ZipFile(source) as archive:
                for member in archive.infolist():
                    name = PurePosixPath(member.filename.replace("\\", "/")).name
                    if name not in wanted or member.is_dir():
                        continue
                    # Ignore non-runtime diagnostics and old bundled backups.
                    parts = PurePosixPath(member.filename.replace("\\", "/")).parts
                    if "native-game-tiled-assets" not in parts and len(parts) != 1:
                        continue
                    if member.file_size > 4195328 * 4:
                        raise ValueError(f"oversize weight: {member.filename}")
                    target = stage / name
                    if target.exists():
                        raise ValueError(f"ambiguous duplicate weight in archive: {name}")
                    with archive.open(member) as src, target.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
        records = validate_asset_dir(stage)
        selected = {record["file"] for record in records}
        for item in stage.iterdir():
            if item.name not in selected:
                item.unlink()
        provenance["weights"] = records
        (stage / "weights-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        (stage / "WEIGHTS-SHA256SUMS").write_text("".join(f'{r["sha256"]}  {r["file"]}\n' for r in records))
        output.mkdir(parents=True, exist_ok=True)
        for item in stage.iterdir():
            os.replace(item, output / item.name)
    print(f"Imported and checked {len(records)} weights in {output}")


def main():
    home = Path(os.environ.get("HOME") or pwd.getpwuid(os.getuid()).pw_dir)
    data_home = Path(os.environ.get("XDG_DATA_HOME") or home / ".local/share")
    parser = argparse.ArgumentParser(description=__doc__, add_help=False, allow_abbrev=False)
    parser.add_argument("-s", "--source", type=Path,
                        help="user-owned upstream full ZIP or exact native-game-tiled-assets directory "
                             "(default: unset; required unless --check or --print-manifest is used)")
    parser.add_argument("-o", "--output", type=Path, default=data_home / "dlsslop-amd/model",
                        help="import destination (default: %(default)s, from XDG_DATA_HOME or ~/.local/share; "
                             "ignored with --check or --print-manifest)")
    parser.add_argument("-c", "--check", type=Path,
                        help="validate a previously imported model without importing files (default: unset; import --source)")
    parser.add_argument("-p", "--print-manifest", action="store_true",
                        help="print required weight names and element counts without importing files (default: off)")
    parser.add_argument("-h", "--help", action="help", help="show this help and exit (default: off)")
    try:
        import_assets(parser.parse_args())
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        print(f"{Path(sys.argv[0]).name}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
