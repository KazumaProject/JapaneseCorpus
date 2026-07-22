from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import write_json


MAX_RELEASE_ASSET_BYTES = 2 * 1024 * 1024 * 1024


def load_stats(stats_directory: Path) -> list[dict[str, Any]]:
    stats: list[dict[str, Any]] = []
    for path in sorted(stats_directory.glob("*.json")):
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
        required = {"asset", "source", "records", "characters", "bytes", "sha256"}
        missing = required - value.keys()
        if missing:
            raise RuntimeError(f"{path} is missing fields: {sorted(missing)}")
        if int(value["bytes"]) >= MAX_RELEASE_ASSET_BYTES:
            raise RuntimeError(f"{value['asset']} exceeds GitHub's 2 GiB asset limit")
        stats.append(value)
    if not stats:
        raise RuntimeError(f"No stats JSON files found in {stats_directory}")
    return stats


def build_manifest(
    stats_directory: Path,
    discovery_path: Path,
    output_path: Path,
    checksums_path: Path,
    version: str,
    pipeline_commit: str,
    built_at: str | None = None,
) -> dict[str, Any]:
    stats = load_stats(stats_directory)
    with discovery_path.open(encoding="utf-8") as stream:
        discovery = json.load(stream)

    built_at = built_at or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    wikipedia = discovery["wikipedia"]
    aozora = discovery["aozora"]
    aozora_stats = next(
        (item for item in stats if item["source"] == "aozora_bunko"), None
    )
    if aozora_stats is None:
        raise RuntimeError("Aozora Bunko statistics are missing")
    assets = [
        {
            "name": item["asset"],
            "source": item["source"],
            "source_version": item.get("source_version", ""),
            "records": int(item["records"]),
            "characters": int(item["characters"]),
            "bytes": int(item["bytes"]),
            "sha256": item["sha256"],
            "warnings": item.get("warnings", {}),
            **(
                {"ruby_annotations": int(item["ruby_annotations"])}
                if "ruby_annotations" in item
                else {}
            ),
        }
        for item in stats
    ]
    assets.sort(key=lambda item: item["name"])

    manifest = {
        "schema_version": 1,
        "release": {
            "version": version,
            "built_at": built_at,
            "pipeline_commit": pipeline_commit,
        },
        "sources": {
            "wikipedia": {
                "dump_date": wikipedia["dump_date"],
                "index_url": wikipedia["index_url"],
                "license": "CC-BY-SA-4.0",
            },
            "aozora_bunko": {
                "repository_url": aozora["repository_url"],
                "repository_commit": aozora["repository_commit"],
                "metadata_url": aozora["metadata_url"],
                "metadata_sha256": aozora_stats.get("metadata_sha256", ""),
                "license": "Public-Domain-only selection",
            },
        },
        "assets": assets,
        "totals": {
            "assets": len(assets),
            "records": sum(item["records"] for item in assets),
            "characters": sum(item["characters"] for item in assets),
            "bytes": sum(item["bytes"] for item in assets),
        },
    }
    write_json(output_path, manifest)
    with checksums_path.open("w", encoding="utf-8", newline="\n") as stream:
        for item in assets:
            stream.write(f"{item['sha256']}  {item['name']}\n")
    return manifest


def verify_remote_assets(stats_directory: Path, assets_json: Path) -> None:
    stats = load_stats(stats_directory)
    with assets_json.open(encoding="utf-8") as stream:
        remote_value = json.load(stream)
    remote_assets = {
        asset["name"]: int(asset["size"])
        for asset in remote_value.get("assets", [])
    }
    for item in stats:
        asset = item["asset"]
        if asset not in remote_assets:
            raise RuntimeError(f"Release is missing asset {asset}")
        if remote_assets[asset] != int(item["bytes"]):
            raise RuntimeError(
                f"Size mismatch for {asset}: local={item['bytes']} remote={remote_assets[asset]}"
            )
