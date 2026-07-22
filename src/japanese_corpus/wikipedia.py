from __future__ import annotations

import bz2
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .common import normalize_text, sha256_file, write_json, write_json_line
from .compression import ZstdTextWriter


def build_wikipedia(
    input_path: Path,
    output_path: Path,
    stats_path: Path,
    dump_date: str,
    limit: int = 0,
) -> dict[str, Any]:
    counters: Counter[str] = Counter()

    with bz2.open(input_path, "rt", encoding="utf-8", errors="strict") as source:
        with ZstdTextWriter(output_path) as output:
            for raw_line in source:
                if not raw_line.strip():
                    continue
                try:
                    document = json.loads(raw_line)
                except json.JSONDecodeError:
                    counters["invalid_json"] += 1
                    continue
                if "index" in document:
                    continue
                if document.get("namespace") != 0:
                    counters["non_main_namespace"] += 1
                    continue

                page_id = document.get("page_id")
                title = str(document.get("title") or "").strip()
                text = normalize_text(str(document.get("text") or ""))
                if page_id in (None, "") or not title or not text:
                    counters["missing_required_field"] += 1
                    continue

                record = {
                    "schema_version": 1,
                    "source": "wikipedia",
                    "document_id": str(page_id),
                    "title": title,
                    "url": f"https://ja.wikipedia.org/?curid={page_id}",
                    "text": text,
                    "metadata": {
                        "dump_date": dump_date,
                        "timestamp": str(document.get("timestamp") or ""),
                    },
                }
                write_json_line(output, record)
                counters["records"] += 1
                counters["characters"] += len(text)
                if limit and counters["records"] >= limit:
                    break

    stats = {
        "schema_version": 1,
        "asset": output_path.name,
        "source": "wikipedia",
        "source_version": dump_date,
        "records": counters["records"],
        "characters": counters["characters"],
        "bytes": output_path.stat().st_size,
        "sha256": sha256_file(output_path),
        "warnings": {
            key: value
            for key, value in sorted(counters.items())
            if key not in {"records", "characters"} and value
        },
    }
    write_json(stats_path, stats)
    return stats
