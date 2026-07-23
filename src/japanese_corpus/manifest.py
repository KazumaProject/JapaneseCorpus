from __future__ import annotations

import hashlib
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
    dictionary_manifest_path: Path | None = None,
    dictionary_checksums_path: Path | None = None,
    english_dictionary_manifest_path: Path | None = None,
    english_dictionary_checksums_path: Path | None = None,
    ajimee_report_path: Path | None = None,
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

    dictionaries = None
    dictionary_checksum_entries: list[tuple[str, str]] = []
    if dictionary_manifest_path is not None or dictionary_checksums_path is not None:
        if dictionary_manifest_path is None or dictionary_checksums_path is None:
            raise RuntimeError(
                "Dictionary manifest and checksum paths must be provided together"
            )
        dictionaries, dictionary_checksum_entries = load_dictionary_metadata(
            dictionary_manifest_path, dictionary_checksums_path
        )

    english_dictionary = None
    english_checksum_entries: list[tuple[str, str]] = []
    if (
        english_dictionary_manifest_path is not None
        or english_dictionary_checksums_path is not None
    ):
        if (
            english_dictionary_manifest_path is None
            or english_dictionary_checksums_path is None
        ):
            raise RuntimeError(
                "English dictionary manifest and checksum paths must be provided together"
            )
        english_dictionary, english_checksum_entries = (
            load_english_dictionary_metadata(
                english_dictionary_manifest_path,
                english_dictionary_checksums_path,
            )
        )

    ajimee_report = None
    ajimee_checksum_entry: tuple[str, str] | None = None
    if ajimee_report_path is not None:
        ajimee_report = load_ajimee_report(ajimee_report_path)
        ajimee_checksum_entry = (
            _sha256(ajimee_report_path),
            "ajimee-bench-report.json",
        )

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
            **(
                {
                    "jmdict": {
                        **english_dictionary["source"],
                        "license": "CC-BY-SA-4.0",
                    }
                }
                if english_dictionary is not None
                else {}
            ),
        },
        "assets": assets,
        "totals": {
            "assets": len(assets),
            "records": sum(item["records"] for item in assets),
            "characters": sum(item["characters"] for item in assets),
            "bytes": sum(item["bytes"] for item in assets),
            **(
                {
                    "dictionary_assets": len(dictionaries["assets"]),
                    "dictionary_entries": sum(
                        int(order["retained_entries"])
                        for order in dictionaries["orders"]
                    ),
                    "dictionary_bytes": sum(
                        int(item["bytes"]) for item in dictionaries["assets"]
                    ),
                }
                if dictionaries is not None
                else {}
            ),
            **(
                {
                    "english_dictionary_assets": len(
                        english_dictionary["assets"]
                    ),
                    "english_dictionary_entries": int(
                        english_dictionary["counts"]["retained_entries"]
                    ),
                    "english_dictionary_bytes": sum(
                        int(item["bytes"])
                        for item in english_dictionary["assets"]
                    ),
                }
                if english_dictionary is not None
                else {}
            ),
        },
        **({"dictionaries": dictionaries} if dictionaries is not None else {}),
        **(
            {"english_dictionary": english_dictionary}
            if english_dictionary is not None
            else {}
        ),
        **(
            {"benchmarks": {"ajimee": ajimee_report}}
            if ajimee_report is not None
            else {}
        ),
    }
    write_json(output_path, manifest)
    with checksums_path.open("w", encoding="utf-8", newline="\n") as stream:
        for item in assets:
            stream.write(f"{item['sha256']}  {item['name']}\n")
        for digest, name in dictionary_checksum_entries:
            stream.write(f"{digest}  {name}\n")
        for digest, name in english_checksum_entries:
            stream.write(f"{digest}  {name}\n")
        if ajimee_checksum_entry is not None:
            digest, name = ajimee_checksum_entry
            stream.write(f"{digest}  {name}\n")
    return manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_dictionary_metadata(
    manifest_path: Path, checksums_path: Path
) -> tuple[dict[str, Any], list[tuple[str, str]]]:
    with manifest_path.open(encoding="utf-8") as stream:
        dictionary = json.load(stream)
    required = {
        "schema_version",
        "format",
        "tokenizer",
        "parameters",
        "corpus",
        "orders",
        "assets",
    }
    missing = required - dictionary.keys()
    if missing:
        raise RuntimeError(f"Dictionary manifest is missing fields: {sorted(missing)}")
    if dictionary["schema_version"] != 1:
        raise RuntimeError("Unsupported dictionary manifest schema")

    sums: dict[str, str] = {}
    with checksums_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.rstrip("\n")
            try:
                digest, name = line.split("  ", 1)
            except ValueError as error:
                raise RuntimeError(
                    f"Malformed dictionary checksum line {line_number}"
                ) from error
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise RuntimeError(
                    f"Invalid SHA-256 on dictionary checksum line {line_number}"
                )
            sums[name] = digest

    entries: list[tuple[str, str]] = []
    for asset in dictionary["assets"]:
        required_asset = {"name", "bytes", "sha256"}
        missing_asset = required_asset - asset.keys()
        if missing_asset:
            raise RuntimeError(
                f"Dictionary asset is missing fields: {sorted(missing_asset)}"
            )
        if int(asset["bytes"]) >= MAX_RELEASE_ASSET_BYTES:
            raise RuntimeError(f"{asset['name']} exceeds GitHub's 2 GiB asset limit")
        if sums.get(asset["name"]) != asset["sha256"]:
            raise RuntimeError(f"Checksum mismatch for dictionary asset {asset['name']}")
        entries.append((asset["sha256"], asset["name"]))

    manifest_name = "ngram-manifest.json"
    manifest_digest = _sha256(manifest_path)
    if sums.get(manifest_name) != manifest_digest:
        raise RuntimeError("Dictionary manifest checksum mismatch")
    entries.extend(
        [
            (manifest_digest, manifest_name),
            (_sha256(checksums_path), "NGRAM-SHA256SUMS"),
        ]
    )
    dictionary["metadata_assets"] = [manifest_name, "NGRAM-SHA256SUMS"]
    return dictionary, entries


def load_english_dictionary_metadata(
    manifest_path: Path, checksums_path: Path
) -> tuple[dict[str, Any], list[tuple[str, str]]]:
    manifest_name = "english-dictionary-manifest.json"
    checksums_name = "ENGLISH-DICTIONARY-SHA256SUMS"
    with manifest_path.open(encoding="utf-8") as stream:
        dictionary = json.load(stream)
    required = {
        "schema_version",
        "format",
        "source",
        "parameters",
        "counts",
        "build",
        "assets",
    }
    missing = required - dictionary.keys()
    if missing:
        raise RuntimeError(
            f"English dictionary manifest is missing fields: {sorted(missing)}"
        )
    if dictionary["schema_version"] != 1:
        raise RuntimeError("Unsupported English dictionary manifest schema")

    sums: dict[str, str] = {}
    with checksums_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.rstrip("\n")
            try:
                digest, name = line.split("  ", 1)
            except ValueError as error:
                raise RuntimeError(
                    f"Malformed English dictionary checksum line {line_number}"
                ) from error
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise RuntimeError(
                    f"Invalid SHA-256 on English dictionary checksum line {line_number}"
                )
            sums[name] = digest

    entries: list[tuple[str, str]] = []
    for asset in dictionary["assets"]:
        required_asset = {"name", "bytes", "sha256"}
        missing_asset = required_asset - asset.keys()
        if missing_asset:
            raise RuntimeError(
                "English dictionary asset is missing fields: "
                f"{sorted(missing_asset)}"
            )
        if int(asset["bytes"]) >= MAX_RELEASE_ASSET_BYTES:
            raise RuntimeError(f"{asset['name']} exceeds GitHub's 2 GiB asset limit")
        if sums.get(asset["name"]) != asset["sha256"]:
            raise RuntimeError(
                f"Checksum mismatch for English dictionary asset {asset['name']}"
            )
        entries.append((asset["sha256"], asset["name"]))

    manifest_digest = _sha256(manifest_path)
    if sums.get(manifest_name) != manifest_digest:
        raise RuntimeError("English dictionary manifest checksum mismatch")
    entries.extend(
        [
            (manifest_digest, manifest_name),
            (_sha256(checksums_path), checksums_name),
        ]
    )
    dictionary["metadata_assets"] = [manifest_name, checksums_name]
    return dictionary, entries


def load_ajimee_report(report_path: Path) -> dict[str, Any]:
    with report_path.open(encoding="utf-8") as stream:
        report = json.load(stream)
    required = {"schema_version", "benchmark", "engine", "metrics"}
    missing = required - report.keys()
    if missing:
        raise RuntimeError(f"AJIMEE-Bench report is missing fields: {sorted(missing)}")
    if report["schema_version"] != 1:
        raise RuntimeError("Unsupported AJIMEE-Bench report schema")
    if report["benchmark"].get("name") != "AJIMEE-Bench":
        raise RuntimeError("Unexpected AJIMEE-Bench report name")
    required_benchmark = {
        "name",
        "dataset",
        "repository_url",
        "commit",
        "sha256",
        "license",
        "items",
    }
    missing_benchmark = required_benchmark - report["benchmark"].keys()
    if missing_benchmark:
        raise RuntimeError(
            "AJIMEE-Bench provenance is missing fields: "
            f"{sorted(missing_benchmark)}"
        )
    if report["benchmark"].get("items") != 200:
        raise RuntimeError("AJIMEE-Bench report must contain all 200 items")
    if report["engine"].get("candidate_limit") != 1:
        raise RuntimeError("AJIMEE-Bench report must evaluate the first candidate")
    if report["engine"].get("context_mode") != "ignored":
        raise RuntimeError("Unexpected AJIMEE-Bench context mode")
    for group in ("overall", "with_context", "without_context"):
        metrics = report["metrics"].get(group, {})
        missing_metrics = {
            "items",
            "correct_at_1",
            "accuracy_at_1",
            "mean_min_cer",
        } - metrics.keys()
        if missing_metrics:
            raise RuntimeError(
                f"AJIMEE-Bench {group} metrics are missing fields: "
                f"{sorted(missing_metrics)}"
            )
        items = int(metrics["items"])
        correct = int(metrics["correct_at_1"])
        accuracy = float(metrics["accuracy_at_1"])
        mean_min_cer = float(metrics["mean_min_cer"])
        if not 0 <= correct <= items:
            raise RuntimeError(f"Invalid AJIMEE-Bench {group} correct count")
        if not 0.0 <= accuracy <= 1.0 or mean_min_cer < 0.0:
            raise RuntimeError(f"Invalid AJIMEE-Bench {group} metric value")
    if report["metrics"]["overall"]["items"] != 200:
        raise RuntimeError("AJIMEE-Bench overall metrics must contain 200 items")
    if report["metrics"]["with_context"]["items"] != 100:
        raise RuntimeError("AJIMEE-Bench contextual metrics must contain 100 items")
    if report["metrics"]["without_context"]["items"] != 100:
        raise RuntimeError("AJIMEE-Bench context-free metrics must contain 100 items")
    return report


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
