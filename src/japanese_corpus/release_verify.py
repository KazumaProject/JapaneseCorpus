"""Download and verify every asset in a published GitHub Release."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlparse

from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource


CHECKSUM_MANIFESTS = (
    "SHA256SUMS",
    "NGRAM-SHA256SUMS",
    "NGRAM-ANNOTATED-SHA256SUMS",
    "ENGLISH-DICTIONARY-SHA256SUMS",
    "HOMOPHONE-SHA256SUMS",
)
SCHEMA_NAMES = (
    "corpus-record.schema.json",
    "release-manifest.schema.json",
    "ngram-manifest.schema.json",
    "ngram-annotated-record.schema.json",
    "ngram-annotated-manifest.schema.json",
    "english-dictionary-manifest.schema.json",
    "ajimee-bench-report.schema.json",
)
JSON_METADATA_NAMES = (
    "manifest.json",
    "ngram-manifest.json",
    "ngram-annotated-manifest.json",
    "english-dictionary-manifest.json",
    "homophone-manifest.json",
    "ajimee-bench-report.json",
)
REQUIRED_METADATA = set(CHECKSUM_MANIFESTS) | set(SCHEMA_NAMES) | set(JSON_METADATA_NAMES)
MAX_RELEASE_ASSET_BYTES = 2 * 1024 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_checksums(path: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                digest, name = line.rstrip("\n").split("  ", 1)
            except ValueError as error:
                raise RuntimeError(f"Malformed checksum line {path}:{line_number}") from error
            if not re.fullmatch(r"[0-9a-f]{64}", digest) or not name:
                raise RuntimeError(f"Invalid checksum line {path}:{line_number}")
            if name in checksums and checksums[name] != digest:
                raise RuntimeError(f"Conflicting checksums for {name}")
            checksums[name] = digest
    return checksums


def validate_schema(
    schema: dict[str, Any],
    instance: Any,
    label: str,
    registry: Registry | None = None,
) -> None:
    validator = Draft202012Validator(
        schema, registry=registry or Registry(), format_checker=FormatChecker()
    )
    error = next(validator.iter_errors(instance), None)
    if error is not None:
        location = ".".join(str(part) for part in error.absolute_path) or "<root>"
        raise RuntimeError(f"{label} violates its JSON schema at {location}: {error.message}")


@contextmanager
def open_zstd(path: Path) -> Iterator[Any]:
    process = subprocess.Popen(
        ["zstd", "--decompress", "--stdout", "--", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    assert process.stderr is not None
    exhausted = False
    try:
        yield process.stdout
        exhausted = True
    finally:
        process.stdout.close()
        if not exhausted and process.poll() is None:
            process.kill()
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        process.stderr.close()
        status = process.wait()
        if exhausted and status:
            raise RuntimeError(f"zstd failed for {path}: {stderr.strip()}")


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with open_zstd(path) as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            try:
                record = json.loads(raw_line)
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise RuntimeError(f"Invalid JSONL at {path}:{line_number}") from error
            if not isinstance(record, dict):
                raise RuntimeError(f"Expected an object at {path}:{line_number}")
            yield record


def validate_annotated_record(record: dict[str, Any], label: str) -> None:
    surface = record["surface"]
    reading = record["reading"]
    order = record["order"]
    if len(surface.encode("utf-8")) > 1023 or len(reading.encode("utf-8")) > 1023:
        raise RuntimeError(f"{label} has a phrase field over 1023 UTF-8 bytes")
    if len(record["source_tokens"]) != order:
        raise RuntimeError(f"{label} source-token count does not match its order")
    if "".join(token["surface"] for token in record["source_tokens"]) != surface:
        raise RuntimeError(f"{label} source-token surfaces do not match the phrase")
    if "".join(token["reading"] for token in record["source_tokens"]) != reading:
        raise RuntimeError(f"{label} source-token readings do not match the phrase")

    for field in ("morphology", "bunsetsu"):
        cursor = 0
        segments = record[field]
        for index, segment in enumerate(segments, start=1):
            start, end = segment["start"], segment["end"]
            if start != cursor or end <= start or end > len(surface):
                raise RuntimeError(f"{label} {field} offsets are invalid at item {index}")
            if surface[start:end] != segment["surface"]:
                raise RuntimeError(f"{label} {field} surface does not match its offsets")
            cursor = end
        if not segments or cursor != len(surface):
            raise RuntimeError(f"{label} {field} offsets do not cover the phrase")


def download_asset(asset: dict[str, Any], path: Path) -> None:
    url = asset.get("browser_download_url")
    if not isinstance(url, str):
        raise RuntimeError(f"Release asset has no download URL: {asset.get('name')}")
    parsed = urlparse(url)
    if parsed.scheme == "file":
        source = Path(parsed.path)
        shutil.copyfile(source, path)
        return
    if parsed.scheme != "https" or parsed.hostname != "github.com":
        raise RuntimeError(f"Unexpected release asset URL: {url}")
    subprocess.run(
        [
            "curl",
            "--fail",
            "--location",
            "--retry",
            "5",
            "--retry-all-errors",
            "--retry-delay",
            "2",
            "--output",
            str(path),
            url,
        ],
        check=True,
    )


def verify_published_release(assets_json_path: Path, tag: str, work_dir: Path) -> dict[str, Any]:
    response = json.loads(assets_json_path.read_text(encoding="utf-8"))
    if response.get("tag_name") != tag:
        raise RuntimeError(
            f"Release API returned {response.get('tag_name')!r}, expected {tag!r}"
        )
    assets = response.get("assets")
    if not isinstance(assets, list) or not assets:
        raise RuntimeError("Published release has no assets")
    by_name: dict[str, dict[str, Any]] = {}
    for asset in assets:
        name = asset.get("name")
        if not isinstance(name, str) or Path(name).name != name:
            raise RuntimeError(f"Invalid release asset name: {name!r}")
        if name in by_name:
            raise RuntimeError(f"Duplicate release asset name: {name}")
        if re.fullmatch(r"ngram-[1-5]-[0-9]{5}\.jsonl\.zst", name):
            raise RuntimeError(f"Temporary GiNZA input leaked into release: {name}")
        if not isinstance(asset.get("size"), int) or asset["size"] >= MAX_RELEASE_ASSET_BYTES:
            raise RuntimeError(f"Invalid or oversized release asset: {name}")
        by_name[name] = asset

    missing = REQUIRED_METADATA - by_name.keys()
    if missing:
        raise RuntimeError(f"Published release is missing metadata assets: {sorted(missing)}")

    work_dir.mkdir(parents=True, exist_ok=True)
    persistent_names = REQUIRED_METADATA
    paths: dict[str, Path] = {}
    for name in persistent_names:
        path = work_dir / name
        download_asset(by_name[name], path)
        if path.stat().st_size != by_name[name]["size"]:
            raise RuntimeError(f"Downloaded asset size mismatch: {name}")
        paths[name] = path

    checksums: dict[str, str] = {}
    for name in CHECKSUM_MANIFESTS:
        for asset_name, digest in parse_checksums(paths[name]).items():
            if asset_name in checksums and checksums[asset_name] != digest:
                raise RuntimeError(f"Conflicting release checksums for {asset_name}")
            checksums[asset_name] = digest
    missing_checksummed_assets = checksums.keys() - by_name.keys()
    if missing_checksummed_assets:
        raise RuntimeError(
            f"Checksum manifests refer to missing assets: {sorted(missing_checksummed_assets)}"
        )

    schemas = {
        name.removesuffix(".schema.json"): json.loads(paths[name].read_text(encoding="utf-8"))
        for name in SCHEMA_NAMES
    }
    for schema in schemas.values():
        Draft202012Validator.check_schema(schema)
    root_manifest = json.loads(paths["manifest.json"].read_text(encoding="utf-8"))
    ngram_manifest = json.loads(paths["ngram-manifest.json"].read_text(encoding="utf-8"))
    annotated_manifest = json.loads(
        paths["ngram-annotated-manifest.json"].read_text(encoding="utf-8")
    )
    english_manifest = json.loads(
        paths["english-dictionary-manifest.json"].read_text(encoding="utf-8")
    )
    homophone_manifest = json.loads(
        paths["homophone-manifest.json"].read_text(encoding="utf-8")
    )
    ajimee_report = json.loads(paths["ajimee-bench-report.json"].read_text(encoding="utf-8"))

    ajimee_uri = "https://github.com/KazumaProject/JapaneseCorpus/blob/main/schema/ajimee-bench-report.schema.json"
    registry = Registry().with_resource(ajimee_uri, Resource.from_contents(schemas["ajimee-bench-report"]))
    validate_schema(schemas["release-manifest"], root_manifest, "manifest.json", registry)
    validate_schema(schemas["ngram-manifest"], ngram_manifest, "ngram-manifest.json")
    validate_schema(
        schemas["ngram-annotated-manifest"], annotated_manifest, "ngram-annotated-manifest.json"
    )
    validate_schema(
        schemas["english-dictionary-manifest"], english_manifest, "english-dictionary-manifest.json"
    )
    validate_schema(schemas["ajimee-bench-report"], ajimee_report, "ajimee-bench-report.json")
    if root_manifest["release"]["version"] != tag:
        raise RuntimeError("Release manifest version does not match the published tag")
    if root_manifest["dictionaries"]["orders"] != ngram_manifest["orders"]:
        raise RuntimeError("Root and dictionary manifests disagree about n-gram counts")
    if root_manifest["annotated_ngrams"]["orders"] != annotated_manifest["orders"]:
        raise RuntimeError("Root and annotated manifests disagree about n-gram counts")

    corpus_record_validator = Draft202012Validator(
        schemas["corpus-record"], format_checker=FormatChecker()
    )
    annotated_record_validator = Draft202012Validator(
        schemas["ngram-annotated-record"]
    )
    corpus_assets = {asset["name"]: asset for asset in root_manifest["assets"]}
    dictionary_assets = {
        asset["name"]: asset
        for asset in ngram_manifest["assets"]
        if asset["name"].endswith(".txt.zst")
    }
    annotated_assets = {
        asset["name"]: asset for asset in annotated_manifest["assets"]
    }
    english_assets = {
        asset["name"]: asset
        for asset in english_manifest["assets"]
        if asset["name"].startswith("mozc-english-") and asset["name"].endswith(".txt.zst")
    }
    homophone_assets = {asset["name"]: asset for asset in homophone_manifest["assets"]}
    expected_data_names = (
        set(corpus_assets)
        | set(dictionary_assets)
        | set(annotated_assets)
        | set(english_assets)
        | set(homophone_assets)
    )
    if not expected_data_names <= by_name.keys():
        raise RuntimeError(
            f"Published release is missing declared data assets: {sorted(expected_data_names - by_name.keys())}"
        )
    if not expected_data_names <= checksums.keys():
        raise RuntimeError(
            f"Data assets are missing checksums: {sorted(expected_data_names - checksums.keys())}"
        )

    ngram_counts = {order["order"]: 0 for order in ngram_manifest["orders"]}
    annotated_counts = {order["order"]: 0 for order in annotated_manifest["orders"]}
    corpus_records = 0
    english_entries = 0
    homophone_records: dict[str, int] = {}
    processed: set[str] = set()
    with tempfile.TemporaryDirectory(prefix="jcorpus-release-verify-", dir=work_dir) as temp:
        current_path = Path(temp) / "asset"
        for name, asset in by_name.items():
            path = paths.get(name)
            if path is None:
                path = current_path
                download_asset(asset, path)
            if path.stat().st_size != asset["size"]:
                raise RuntimeError(f"Downloaded asset size mismatch: {name}")
            declared_asset = (
                corpus_assets.get(name)
                or dictionary_assets.get(name)
                or annotated_assets.get(name)
                or english_assets.get(name)
                or homophone_assets.get(name)
            )
            if declared_asset is not None and int(declared_asset["bytes"]) != asset["size"]:
                raise RuntimeError(f"Manifest asset size mismatch: {name}")
            expected_digest = checksums.get(name)
            if expected_digest is not None and sha256_file(path) != expected_digest:
                raise RuntimeError(f"Downloaded asset SHA-256 mismatch: {name}")

            if name.endswith(".jsonl.zst"):
                if name in corpus_assets:
                    asset_records = 0
                    for record in iter_jsonl(path):
                        validate_schema(
                            schemas["corpus-record"],
                            record,
                            f"{name} record {asset_records + 1}",
                        )
                        asset_records += 1
                    if asset_records != int(corpus_assets[name]["records"]):
                        raise RuntimeError(f"Corpus record count mismatch: {name}")
                    corpus_records += asset_records
                elif name in annotated_assets:
                    match = re.fullmatch(r"ngram-annotated-([1-5])-([0-9]{5})\.jsonl\.zst", name)
                    if match is None:
                        raise RuntimeError(f"Invalid annotated asset name: {name}")
                    order = int(match.group(1))
                    asset_records = 0
                    for record in iter_jsonl(path):
                        validate_schema(
                            schemas["ngram-annotated-record"],
                            record,
                            f"{name} record {asset_records + 1}",
                        )
                        validate_annotated_record(record, f"{name} record {asset_records + 1}")
                        if record["order"] != order:
                            raise RuntimeError(f"Annotated row has the wrong order: {name}")
                        annotated_counts[order] += 1
                        asset_records += 1
                    if asset_records != int(annotated_assets[name]["entries"]):
                        raise RuntimeError(f"Annotated asset entry count mismatch: {name}")
                elif name in homophone_assets:
                    asset_records = sum(1 for _ in iter_jsonl(path))
                    if asset_records != int(homophone_assets[name]["records"]):
                        raise RuntimeError(f"Homophone asset record count mismatch: {name}")
                    homophone_records[name] = asset_records
                else:
                    raise RuntimeError(f"Unrecognized JSONL release asset: {name}")
                processed.add(name)
            elif name.endswith(".txt.zst"):
                if name not in dictionary_assets and name not in english_assets:
                    raise RuntimeError(f"Unrecognized dictionary release asset: {name}")
                asset_entries = 0
                with open_zstd(path) as stream:
                    for line_number, raw_line in enumerate(stream, start=1):
                        try:
                            fields = raw_line.decode("utf-8").rstrip("\n").split("\t")
                        except UnicodeDecodeError as error:
                            raise RuntimeError(f"Invalid UTF-8 in {name}:{line_number}") from error
                        if (
                            len(fields) != 5
                            or not fields[0]
                            or not fields[4]
                            or not all(re.fullmatch(r"[0-9]+", value) for value in fields[1:4])
                        ):
                            raise RuntimeError(f"Invalid Mozc row at {name}:{line_number}")
                        asset_entries += 1
                expected_entries = int(
                    (dictionary_assets.get(name) or english_assets[name])["entries"]
                )
                if asset_entries != expected_entries:
                    raise RuntimeError(f"Dictionary entry count mismatch: {name}")
                if name in dictionary_assets:
                    order = int(dictionary_assets[name]["order"])
                    ngram_counts[order] += asset_entries
                elif name in english_assets:
                    english_entries += asset_entries
                processed.add(name)
            elif name.endswith(".xml.gz"):
                with gzip.open(path, "rb") as stream:
                    while stream.read(1024 * 1024):
                        pass
            elif name.endswith(".zst"):
                with open_zstd(path) as stream:
                    while stream.read(1024 * 1024):
                        pass
            elif name.endswith(".json"):
                json.loads(path.read_text(encoding="utf-8"))
            if name not in paths:
                path.unlink(missing_ok=True)

    if processed.intersection(corpus_assets) != set(corpus_assets):
        raise RuntimeError("Not every corpus shard was verified")
    if processed.intersection(dictionary_assets) != set(dictionary_assets):
        raise RuntimeError("Not every Mozc dictionary shard was verified")
    if processed.intersection(annotated_assets) != set(annotated_assets):
        raise RuntimeError("Not every annotated n-gram shard was verified")
    if processed.intersection(english_assets) != set(english_assets):
        raise RuntimeError("Not every English dictionary shard was verified")
    if processed.intersection(homophone_assets) != set(homophone_assets):
        raise RuntimeError("Not every homophone shard was verified")
    for order in ngram_manifest["orders"]:
        if ngram_counts[int(order["order"])] != int(order["retained_entries"]):
            raise RuntimeError(f"Mozc order {order['order']} total entry count mismatch")
    for order in annotated_manifest["orders"]:
        if annotated_counts[int(order["order"])] != int(order["retained_entries"]):
            raise RuntimeError(f"Annotated order {order['order']} total entry count mismatch")
        if annotated_counts[int(order["order"])] != ngram_counts[int(order["order"] )]:
            raise RuntimeError(f"Annotated and dictionary order {order['order']} counts differ")
    if english_entries != int(english_manifest["counts"]["retained_entries"]):
        raise RuntimeError("English dictionary total entry count mismatch")
    if int(root_manifest["totals"]["annotated_ngram_entries"]) != sum(annotated_counts.values()):
        raise RuntimeError("Root annotated n-gram total entry count mismatch")
    if int(root_manifest["totals"]["dictionary_entries"]) != sum(ngram_counts.values()):
        raise RuntimeError("Root n-gram dictionary total entry count mismatch")
    if corpus_records != int(root_manifest["totals"]["records"]):
        raise RuntimeError("Total corpus record count mismatch")
    if sum(homophone_records.values()) != int(homophone_manifest["corpus"]["homophone_groups"] + homophone_manifest["corpus"]["occurrences"]):
        raise RuntimeError("Homophone total record count mismatch")
    accuracy = float(ajimee_report["metrics"]["overall"]["accuracy_at_1"])
    min_cer = float(ajimee_report["metrics"]["overall"]["mean_min_cer"])
    if accuracy < 0.50 or min_cer > 0.09:
        raise RuntimeError("AJIMEE-Bench results do not meet release thresholds")

    return {
        "release": tag,
        "assets": len(assets),
        "corpus_records": corpus_records,
        "ngram_entries_by_order": ngram_counts,
        "annotated_entries_by_order": annotated_counts,
        "homophone_assets": len(homophone_records),
        "ajimee_accuracy_at_1": accuracy,
        "ajimee_mean_min_cer": min_cer,
    }
