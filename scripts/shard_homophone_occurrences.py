#!/usr/bin/env python3
"""Split a complete homophone occurrence asset into reproducible zstd shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


GROUPS_ASSET = "homophone-groups.jsonl.zst"
OCCURRENCES_ASSET = "homophone-occurrences.jsonl.zst"
MANIFEST_ASSET = "homophone-manifest.json"
CHECKSUMS_ASSET = "HOMOPHONE-SHA256SUMS"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def asset_metadata(path: Path, records: int) -> dict[str, object]:
    return {
        "name": path.name,
        "records": records,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def split_occurrences(source: Path, output_dir: Path, records_per_shard: int) -> list[dict[str, object]]:
    decoder = subprocess.Popen(
        ["zstd", "-dc", "--", str(source)],
        stdout=subprocess.PIPE,
    )
    assert decoder.stdout is not None
    assets: list[dict[str, object]] = []
    shard_index = 0
    shard_records = 0
    shard_process: subprocess.Popen[bytes] | None = None
    shard_stdin = None

    try:
        for line in decoder.stdout:
            if shard_process is None or shard_records == records_per_shard:
                if shard_process is not None:
                    assert shard_stdin is not None
                    shard_stdin.close()
                    if shard_process.wait() != 0:
                        raise RuntimeError("zstd failed while closing an occurrence shard")
                    shard_path = output_dir / f"homophone-occurrences-{shard_index:05}.jsonl.zst"
                    assets.append(asset_metadata(shard_path, shard_records))
                    shard_index += 1
                shard_path = output_dir / f"homophone-occurrences-{shard_index:05}.jsonl.zst"
                shard_process = subprocess.Popen(
                    ["zstd", "-T0", "-3", "-o", str(shard_path)],
                    stdin=subprocess.PIPE,
                )
                assert shard_process.stdin is not None
                shard_stdin = shard_process.stdin
                shard_records = 0
            assert shard_stdin is not None
            shard_stdin.write(line)
            shard_records += 1
    finally:
        decoder.stdout.close()

    if shard_process is not None:
        assert shard_stdin is not None
        shard_stdin.close()
        if shard_process.wait() != 0:
            raise RuntimeError("zstd failed while closing the final occurrence shard")
        shard_path = output_dir / f"homophone-occurrences-{shard_index:05}.jsonl.zst"
        assets.append(asset_metadata(shard_path, shard_records))
    if decoder.wait() != 0:
        raise RuntimeError("zstd failed while decoding the occurrence asset")
    if not assets:
        raise RuntimeError("the occurrence asset contained no records")
    return assets


def write_checksums(output_dir: Path, assets: list[dict[str, object]]) -> None:
    manifest_path = output_dir / MANIFEST_ASSET
    lines = [f"{asset['sha256']}  {asset['name']}" for asset in assets]
    lines.append(f"{sha256_file(manifest_path)}  {MANIFEST_ASSET}")
    (output_dir / CHECKSUMS_ASSET).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--records-per-shard", type=int, default=5_000_000)
    args = parser.parse_args()
    if args.records_per_shard < 1:
        parser.error("--records-per-shard must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    input_dir = args.input_dir
    manifest = json.loads((input_dir / MANIFEST_ASSET).read_text(encoding="utf-8"))
    occurrence_asset = input_dir / OCCURRENCES_ASSET
    if not occurrence_asset.is_file():
        raise FileNotFoundError(occurrence_asset)
    if not (input_dir / GROUPS_ASSET).is_file():
        raise FileNotFoundError(input_dir / GROUPS_ASSET)

    groups_path = args.output_dir / GROUPS_ASSET
    shutil.copy2(input_dir / GROUPS_ASSET, groups_path)
    occurrence_assets = split_occurrences(
        occurrence_asset, args.output_dir, args.records_per_shard
    )

    expected_occurrences = int(manifest["corpus"]["occurrences"])
    actual_occurrences = sum(int(asset["records"]) for asset in occurrence_assets)
    if actual_occurrences != expected_occurrences:
        raise RuntimeError(
            f"occurrence count mismatch: manifest={expected_occurrences}, output={actual_occurrences}"
        )

    assets = [asset_metadata(groups_path, int(manifest["corpus"]["homophone_groups"]))]
    assets.extend(occurrence_assets)
    manifest["corpus"]["occurrence_shard_records"] = args.records_per_shard
    manifest["assets"] = assets
    (args.output_dir / MANIFEST_ASSET).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_checksums(args.output_dir, assets)


if __name__ == "__main__":
    main()
