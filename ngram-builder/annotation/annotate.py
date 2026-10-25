#!/usr/bin/env python3
"""Add deterministic, stand-alone GiNZA bunsetsu annotations to n-gram rows."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterator

import ginza
import jsonschema
import spacy

GINZA_VERSION = "5.2.0"
MODEL_VERSION = "5.2.0"
MODEL_WHEEL_SHA256 = "c31a2ac57a7c2a1440f7e58111e37306c20dfaf08fd5a3f6f065948ff126cb30"
MAX_PART_RECORDS = 100_000
INPUT_PATTERN = re.compile(r"ngram-([1-5])-([0-9]{5})\.jsonl\.zst$")
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def minimum_count_for_order(order: int, unigram_threshold: int, ngram_threshold: int) -> int:
    if order == 1:
        return unigram_threshold
    if 2 <= order <= 5:
        return ngram_threshold
    raise ValueError(f"unsupported n-gram order: {order}")


def decode_lines(path: Path) -> Iterator[dict[str, Any]]:
    process = subprocess.Popen(
        ["zstd", "--decompress", "--stdout", "--", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    assert process.stderr is not None
    try:
        for line_number, raw_line in enumerate(process.stdout, start=1):
            try:
                record = json.loads(raw_line)
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise RuntimeError(f"{path}:{line_number}: invalid UTF-8 JSON") from error
            if not isinstance(record, dict):
                raise RuntimeError(f"{path}:{line_number}: expected a JSON object")
            yield record
    finally:
        process.stdout.close()
    stderr = process.stderr.read().decode("utf-8", errors="replace")
    process.stderr.close()
    status = process.wait()
    if status:
        raise RuntimeError(f"zstd failed for {path}: {stderr.strip()}")


def analysis_samples(path: Path, threshold: int) -> Iterator[tuple[str, dict[str, Any]]]:
    for index, record in enumerate(decode_lines(path), start=1):
        try:
            order = record["order"]
            tokens = record["source_tokens"]
            surface = record["surface"]
            reading = record["reading"]
            count = record["count"]
            if not isinstance(order, int) or not 1 <= order <= 5:
                raise ValueError("order must be between 1 and 5")
            if not isinstance(tokens, list) or len(tokens) != order:
                raise ValueError("source token count does not match order")
            if not isinstance(count, int) or count < threshold:
                raise ValueError("entry is below the configured frequency threshold")
            if "".join(token["surface"] for token in tokens) != surface:
                raise ValueError("source token surfaces do not match the n-gram surface")
            if "".join(token["reading"] for token in tokens) != reading:
                raise ValueError("source token readings do not match the n-gram reading")
            if len(surface.encode("utf-8")) > 1023 or len(reading.encode("utf-8")) > 1023:
                raise ValueError("n-gram field exceeds 1023 UTF-8 bytes")
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f"{path}: record {index}: {error}") from error
        yield surface, record


def bunsetsu_spans(doc: Any) -> Iterator[Any]:
    labels = ginza.bunsetu_bi_labels(doc)
    if len(labels) != len(doc) or any(label not in {"B", "I"} for label in labels):
        raise RuntimeError("GiNZA returned invalid bunsetsu boundary labels")

    boundaries = [0]
    for token in doc:
        if token.i > 0 and (labels[token.i] == "B" or token.is_sent_start):
            boundaries.append(token.i)
    boundaries.append(len(doc))

    for start, end in zip(boundaries, boundaries[1:]):
        if end > start:
            yield doc[start:end]


def annotate_record(
    doc: Any,
    source: dict[str, Any],
    validator: jsonschema.Draft202012Validator,
) -> dict[str, Any]:
    surface = source["surface"]
    if doc.text != surface or not doc.has_annotation("SENT_START"):
        raise RuntimeError("GiNZA returned a document that does not match its input")

    morphology = [
        {
            "surface": token.text,
            "lemma": token.lemma_,
            "pos": token.pos_,
            "tag": token.tag_,
            "reading": ginza.reading_form(token, True),
            "start": token.idx,
            "end": token.idx + len(token.text),
        }
        for token in doc
        if not token.is_space
    ]
    cursor = 0
    for token in morphology:
        start, end = token["start"], token["end"]
        gap = surface[cursor:start]
        if (
            start < cursor
            or end <= start
            or (gap and not gap.isspace())
            or surface[start:end] != token["surface"]
        ):
            raise RuntimeError(
                "GiNZA morphology offsets are invalid or leave non-whitespace text uncovered"
            )
        cursor = end
    trailing_text = surface[cursor:]
    if not morphology or (trailing_text and not trailing_text.isspace()):
        raise RuntimeError("GiNZA morphology does not cover all non-whitespace text")

    bunsetsu = []
    cursor = 0
    for span in bunsetsu_spans(doc):
        start, end = span.start_char, span.end_char
        if start != cursor or end <= start or surface[start:end] != span.text:
            raise RuntimeError("GiNZA bunsetsu offsets are invalid or discontinuous")
        bunsetsu.append({"surface": span.text, "start": start, "end": end})
        cursor = end
    if not bunsetsu or cursor != len(surface):
        raise RuntimeError("GiNZA bunsetsu do not cover the complete phrase")

    annotated = {
        **source,
        "analysis_scope": "standalone_phrase",
        "morphology": morphology,
        "bunsetsu": bunsetsu,
    }
    validator.validate(annotated)
    return annotated


def write_annotated_part(
    source_path: Path,
    output_path: Path,
    threshold: int,
    nlp: Any,
    validator: jsonschema.Draft202012Validator,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("wb") as compressed_output:
        process = subprocess.Popen(
            ["zstd", "--threads=0", "-9", "--stdout"],
            stdin=subprocess.PIPE,
            stdout=compressed_output,
            stderr=subprocess.PIPE,
        )
        assert process.stdin is not None
        assert process.stderr is not None
        try:
            samples = analysis_samples(source_path, threshold)
            for doc, source in nlp.pipe(
                samples, as_tuples=True, batch_size=128, n_process=1
            ):
                annotated = annotate_record(doc, source, validator)
                process.stdin.write(
                    (json.dumps(annotated, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
                        "utf-8"
                    )
                )
                count += 1
        except Exception:
            process.stdin.close()
            process.kill()
            process.wait()
            output_path.unlink(missing_ok=True)
            raise
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        process.stderr.close()
        status = process.wait()
        if status:
            output_path.unlink(missing_ok=True)
            raise RuntimeError(f"zstd failed for {output_path}: {stderr.strip()}")
    return count


def annotate(input_dir: Path, output_dir: Path, ngram_manifest_path: Path, pipeline_commit: str) -> None:
    for package, expected in (("ginza", GINZA_VERSION), ("ja_ginza", MODEL_VERSION)):
        actual = importlib.metadata.version(package.replace("_", "-"))
        if actual != expected:
            raise RuntimeError(f"expected {package} {expected}, found {actual}")

    ngram_manifest = json.loads(ngram_manifest_path.read_text(encoding="utf-8"))
    record_schema = json.loads(
        (REPOSITORY_ROOT / "schema/ngram-annotated-record.schema.json").read_text(
            encoding="utf-8"
        )
    )
    manifest_schema = json.loads(
        (REPOSITORY_ROOT / "schema/ngram-annotated-manifest.schema.json").read_text(
            encoding="utf-8"
        )
    )
    record_validator = jsonschema.Draft202012Validator(record_schema)
    manifest_validator = jsonschema.Draft202012Validator(manifest_schema)
    jsonschema.Draft202012Validator.check_schema(record_schema)
    jsonschema.Draft202012Validator.check_schema(manifest_schema)
    unigram_threshold = int(ngram_manifest["parameters"]["unigram_min_count"])
    ngram_threshold = int(ngram_manifest["parameters"]["ngram_min_count"])
    expected_counts = {
        int(item["order"]): int(item["retained_entries"])
        for item in ngram_manifest["orders"]
    }
    if sorted(expected_counts) != list(range(1, 6)):
        raise RuntimeError("n-gram manifest must describe orders 1 through 5")
    files = sorted(input_dir.glob("ngram-*.jsonl.zst"))
    if not files:
        raise RuntimeError(f"no n-gram analysis input files found in {input_dir}")
    for stale in output_dir.glob("ngram-annotated-*.jsonl.zst"):
        stale.unlink()

    nlp = spacy.load("ja_ginza")
    if "bunsetu_recognizer" not in nlp.pipe_names:
        raise RuntimeError("the pinned GiNZA model does not contain bunsetu_recognizer")
    orders: dict[int, dict[str, int]] = {
        order: {"records": 0, "assets": 0} for order in range(1, 6)
    }
    assets: list[dict[str, Any]] = []

    for source_path in files:
        match = INPUT_PATTERN.fullmatch(source_path.name)
        if match is None:
            raise RuntimeError(f"unexpected analysis input filename: {source_path.name}")
        order, part = int(match.group(1)), int(match.group(2))
        threshold = minimum_count_for_order(order, unigram_threshold, ngram_threshold)
        output_path = output_dir / f"ngram-annotated-{order}-{part:05}.jsonl.zst"
        count = write_annotated_part(
            source_path, output_path, threshold, nlp, record_validator
        )
        if count > MAX_PART_RECORDS:
            raise RuntimeError(f"annotated shard contains too many records: {output_path}")
        orders[order]["records"] += count
        orders[order]["assets"] += 1
        assets.append(
            {
                "name": output_path.name,
                "entries": count,
                "bytes": output_path.stat().st_size,
                "sha256": sha256_file(output_path),
            }
        )
        print(f"annotated {output_path.name}: {count} records", file=sys.stderr)

    if any(orders[order]["assets"] == 0 for order in range(1, 6)):
        raise RuntimeError("at least one n-gram order has no analysis input assets")
    for order in range(1, 6):
        if orders[order]["records"] != expected_counts[order]:
            raise RuntimeError(
                f"annotated order {order} contains {orders[order]['records']} entries; "
                f"dictionary manifest expects {expected_counts[order]}"
            )
    manifest = {
        "schema_version": 1,
        "format": {
            "name": "JapaneseCorpus annotated n-gram JSONL",
            "encoding": "UTF-8/LF",
            "compression": "Zstandard",
            "offset_unit": "Unicode code points",
            "analysis_scope": "standalone_phrase",
        },
        "analyzer": {
            "implementation": "GiNZA",
            "version": GINZA_VERSION,
            "model": "ja_ginza",
            "model_version": MODEL_VERSION,
            "model_artifact_sha256": MODEL_WHEEL_SHA256,
            "spacy_version": importlib.metadata.version("spacy"),
        },
        "parameters": {
            "unigram_min_count": ngram_manifest["parameters"]["unigram_min_count"],
            "ngram_min_count": threshold,
            "maximum_phrase_field_bytes": 1023,
            "pipeline_commit": pipeline_commit,
            "ngram_manifest_sha256": sha256_file(ngram_manifest_path),
        },
        "orders": [
            {
                "order": order,
                "assets": orders[order]["assets"],
                "retained_entries": orders[order]["records"],
            }
            for order in range(1, 6)
        ],
        "assets": sorted(assets, key=lambda asset: asset["name"]),
    }
    manifest_validator.validate(manifest)
    manifest_path = output_dir / "ngram-annotated-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    checksum_paths = sorted(output_dir.glob("ngram-annotated-*.jsonl.zst"))
    checksum_paths.append(manifest_path)
    checksum_path = output_dir / "NGRAM-ANNOTATED-SHA256SUMS"
    with checksum_path.open("w", encoding="utf-8", newline="\n") as stream:
        for path in checksum_paths:
            stream.write(f"{sha256_file(path)}  {path.name}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ngram-manifest", type=Path, required=True)
    parser.add_argument("--pipeline-commit", required=True)
    args = parser.parse_args()
    annotate(args.input_dir, args.output_dir, args.ngram_manifest, args.pipeline_commit)


if __name__ == "__main__":
    main()
