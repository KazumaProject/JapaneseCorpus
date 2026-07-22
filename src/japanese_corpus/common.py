from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable, TextIO


CONTROL_CHARACTER_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
EXCESS_BLANK_LINES_RE = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    """Apply conservative normalization without changing Japanese wording."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFC", text)
    text = CONTROL_CHARACTER_RE.sub("", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = EXCESS_BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


def write_json_line(stream: TextIO, value: dict[str, Any]) -> None:
    json.dump(value, stream, ensure_ascii=False, separators=(",", ":"))
    stream.write("\n")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unique_preserving_order(values: Iterable[Any]) -> list[Any]:
    result: list[Any] = []
    seen: set[Any] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
