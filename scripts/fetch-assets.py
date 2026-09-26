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

# The last block of each network stage and the stage's channel width.
STAGES = ((4, 32), (8, 64), (14, 128), (22, 256), (47, 512), (55, 256), (61, 128), (65, 64), (69, 32))


def weight_manifest():
    """Map runtime base names to element counts (Network::WeightElements)."""
    result = {"head-matrix": 524288, "decoder39-weights": 524800,
              "post70-scales": 64, "post70-head": 96,
              "post70-ffn": 8736, "post70-attention": 8225}
    for block in list(range(31)) + list(range(40, 70)):
        c = next(width for last, width in STAGES if block <= last)
        if c == 512:
            result[f"block{block}-ffwd"] = 524288
            result[f"block{block}-ffwd-projection"] = 262656
        else:
            result[f"block{block}-ffn"] = 8736 if c == 32 else 9 * c * c + c
        result[f"block{block}-attention"] = 8225 if c == 32 else 4 * c * c + (c // 32) * 4096 + c // 32 + c
    for block in range(31, 39):
        for part, count in [("expand", 4194304), ("contract", 4195328), ("qkv", 3145760), ("projection", 1049600)]:
            result[f"block{block}-{part}"] = count
    # "-ds" ends each stage before the widest one; "-weights" starts each after it.
    for block, c in STAGES[:4]:
        result[f"block{block}-ds"] = 2 * c * c
    for (last, _), (_, c) in zip(STAGES[4:], STAGES[5:]):
        result[f"block{last + 1}-weights"] = 2 * c * c + c
    return dict(sorted(result.items()))


def sha256(path):
    with path.open("rb") as src:
        return hashlib.file_digest(src, "sha256").hexdigest()


def checksum_text(records):
    """WEIGHTS-SHA256SUMS in sha256sum format, as the import writes it."""
    return "".join(f'{r["sha256"]}  {r["file"]}\n' for r in records)


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
        sums = args.check / "WEIGHTS-SHA256SUMS"
        if sums.exists() and sums.read_text() != checksum_text(records):
            raise ValueError(f"{sums}: recorded hashes do not match the weights; "
                             f"run sha256sum --check WEIGHTS-SHA256SUMS in {args.check}")
        print(f"Complete model: {len(records)} weights; {sum(x['bytes'] for x in records):,} bytes")
        return
    if not args.source:
        raise ValueError("provide --source ZIP_OR_DIRECTORY, or --check ASSET_DIRECTORY")
    source = args.source.resolve()
    output = args.output.resolve()
    if source == output:
        raise ValueError("source and output must be different directories; use --check instead")
    output.parent.mkdir(parents=True, exist_ok=True)
    wanted = {base + ext for base in weight_manifest() for ext in [".f16", ".f32"]}
    # Preserve independently built HIP subdirectory; never overwrite weights.
    if any((output / name).exists() for name in wanted):
        raise ValueError(f"weights already exist in {output}; choose a new output or use --check")
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
                    path = PurePosixPath(member.filename.replace("\\", "/"))
                    # Ignore non-runtime diagnostics and old bundled backups.
                    if (path.name not in wanted or member.is_dir()
                            or ("native-game-tiled-assets" not in path.parts and len(path.parts) != 1)):
                        continue
                    if member.file_size > 4195328 * 4:
                        raise ValueError(f"oversize weight: {member.filename}")
                    target = stage / path.name
                    if target.exists():
                        raise ValueError(f"ambiguous duplicate weight in archive: {path.name}")
                    with archive.open(member) as src, target.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
        records = validate_asset_dir(stage)
        selected = {record["file"] for record in records}
        for item in stage.iterdir():
            if item.name not in selected:
                item.unlink()
        provenance["weights"] = records
        (stage / "weights-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        (stage / "WEIGHTS-SHA256SUMS").write_text(checksum_text(records))
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
                        help="validate a model's weight sizes and, when its WEIGHTS-SHA256SUMS exists, the recorded "
                             "hashes, without importing files (default: unset; import --source)")
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
