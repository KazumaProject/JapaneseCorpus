from __future__ import annotations

import csv
import html
import io
import re
import unicodedata
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import unquote, urlparse

from .common import (
    normalize_text,
    sha256_file,
    unique_preserving_order,
    write_json,
    write_json_line,
)
from .compression import ZstdTextWriter


SEPARATOR_RE = re.compile(r"^-{10,}\s*$")
AOZORA_NOTE_RE = re.compile(r"※?［＃[^］]*］")
KANJI_RUBY_BASE_RE = re.compile(r"[一-龯々〆ヵヶ]")
FOOTER_PREFIXES = ("底本：", "青空文庫作成ファイル：")


def read_metadata_rows(metadata_zip: Path) -> list[dict[str, str]]:
    with zipfile.ZipFile(metadata_zip) as archive:
        csv_names = sorted(
            name for name in archive.namelist() if name.lower().endswith(".csv")
        )
        if len(csv_names) != 1:
            raise RuntimeError(
                f"Expected exactly one CSV in {metadata_zip}, found {len(csv_names)}"
            )
        with archive.open(csv_names[0]) as raw_stream:
            text_stream = io.TextIOWrapper(raw_stream, encoding="utf-8-sig", newline="")
            return [dict(row) for row in csv.DictReader(text_stream)]


def group_works(rows: Iterable[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        work_id = (row.get("作品ID") or "").strip()
        if work_id:
            grouped[work_id].append(row)
    return dict(grouped)


def is_public_domain_work(rows: list[dict[str, str]]) -> bool:
    return bool(rows) and all(
        (row.get("作品著作権フラグ") or "").strip() == "なし"
        and (row.get("人物著作権フラグ") or "").strip() == "なし"
        for row in rows
    )


def source_url_path(text_url: str) -> str | None:
    parsed = urlparse(text_url)
    if parsed.hostname not in {"aozora.gr.jp", "www.aozora.gr.jp"}:
        return None
    url_path = unquote(parsed.path).lstrip("/")
    if not url_path.lower().endswith(".zip"):
        return None
    return url_path


def source_archive_path(source_root: Path, text_url: str) -> Path | None:
    url_path = source_url_path(text_url)
    if url_path is None:
        return None
    path = source_root / url_path
    return path if path.is_file() else None


def read_source_archive(source_root: Path, text_url: str) -> bytes | None:
    archive_path = source_archive_path(source_root, text_url)
    if archive_path is None:
        return None
    try:
        with zipfile.ZipFile(archive_path) as archive:
            text_names = sorted(
                name
                for name in archive.namelist()
                if not name.endswith("/") and name.lower().endswith(".txt")
            )
            if len(text_names) != 1:
                return None
            return archive.read(text_names[0])
    except zipfile.BadZipFile:
        return None


def selected_source_paths(metadata_zip: Path) -> list[str]:
    paths: set[str] = set()
    for work_rows in group_works(read_metadata_rows(metadata_zip)).values():
        if not is_public_domain_work(work_rows):
            continue
        for row in work_rows:
            text_url = (row.get("テキストファイルURL") or "").strip()
            path = source_url_path(text_url)
            if path is not None:
                paths.add(path)
    return sorted(paths)


def decode_aozora_text(data: bytes, declared_encoding: str) -> tuple[str, str]:
    normalized_encoding = declared_encoding.lower().replace("-", "")
    candidates = ["utf-8-sig", "cp932"] if "utf8" in normalized_encoding else ["cp932", "utf-8-sig"]
    last_error: UnicodeDecodeError | None = None
    for encoding in candidates:
        try:
            return data.decode(encoding), encoding
        except UnicodeDecodeError as error:
            last_error = error
    assert last_error is not None
    raise last_error


def strip_aozora_metadata(text: str) -> str:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    separators = [index for index, line in enumerate(lines) if SEPARATOR_RE.match(line)]
    start = 0
    if len(separators) >= 2:
        between = "\n".join(lines[separators[0] + 1 : separators[1]])
        start = (
            separators[1] + 1
            if "テキスト中に現れる記号" in between
            else separators[0] + 1
        )
    elif separators:
        start = separators[0] + 1

    end = len(lines)
    for index in range(start, len(lines)):
        if lines[index].lstrip().startswith(FOOTER_PREFIXES):
            end = index
            break

    body_lines: list[str] = []
    previous_blank = False
    for line in lines[start:end]:
        line = AOZORA_NOTE_RE.sub("", html.unescape(line)).rstrip()
        is_blank = not line.strip()
        if is_blank and previous_blank:
            continue
        body_lines.append("" if is_blank else line)
        previous_blank = is_blank
    return normalize_text("\n".join(body_lines))


def extract_ruby(text: str) -> tuple[str, list[dict[str, Any]], int]:
    """Remove Aozora ruby markup and retain readings with code-point offsets."""
    text = unicodedata.normalize("NFC", text)
    output: list[str] = []
    annotations: list[dict[str, Any]] = []
    malformed = 0
    index = 0

    while index < len(text):
        character = text[index]
        if character == "｜":
            ruby_open = text.find("《", index + 1)
            newline = text.find("\n", index + 1)
            if ruby_open != -1 and (newline == -1 or ruby_open < newline):
                ruby_close = text.find("》", ruby_open + 1)
                if ruby_close != -1:
                    base = text[index + 1 : ruby_open]
                    reading = text[ruby_open + 1 : ruby_close].strip()
                    if base and reading:
                        start = len(output)
                        output.extend(base)
                        annotations.append(
                            {"start": start, "end": len(output), "reading": reading}
                        )
                        index = ruby_close + 1
                        continue
            output.append(character)
            index += 1
            continue

        if character == "《":
            ruby_close = text.find("》", index + 1)
            if ruby_close != -1:
                reading = text[index + 1 : ruby_close].strip()
                end = len(output)
                start = end
                while start > 0 and KANJI_RUBY_BASE_RE.fullmatch(output[start - 1]):
                    start -= 1
                if start < end and reading:
                    annotations.append(
                        {"start": start, "end": end, "reading": reading}
                    )
                else:
                    malformed += 1
                index = ruby_close + 1
                continue

        output.append(character)
        index += 1

    return "".join(output), annotations, malformed


def people_for_work(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    people: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        person_id = (row.get("人物ID") or "").strip()
        name = " ".join(
            part
            for part in [
                (row.get("姓") or "").strip(),
                (row.get("名") or "").strip(),
            ]
            if part
        )
        role = (row.get("役割フラグ") or "").strip()
        key = (person_id, name, role)
        if key not in seen:
            seen.add(key)
            people.append({"person_id": person_id, "name": name, "role": role})
    people.sort(key=lambda person: (person["role"], person["person_id"], person["name"]))
    return people


def build_aozora(
    metadata_zip: Path,
    source_root: Path,
    output_path: Path,
    stats_path: Path,
    source_commit: str,
    limit: int = 0,
) -> dict[str, Any]:
    rows = read_metadata_rows(metadata_zip)
    works = group_works(rows)
    counters: Counter[str] = Counter()

    with ZstdTextWriter(output_path) as output:
        for work_id in sorted(works):
            work_rows = works[work_id]
            if not is_public_domain_work(work_rows):
                counters["excluded_by_copyright"] += 1
                continue

            first = work_rows[0]
            text_urls = sorted(
                unique_preserving_order(
                    (row.get("テキストファイルURL") or "").strip()
                    for row in work_rows
                    if (row.get("テキストファイルURL") or "").strip()
                )
            )
            if not text_urls:
                counters["missing_text_url"] += 1
                continue
            supported_urls = [url for url in text_urls if source_url_path(url) is not None]
            if not supported_urls:
                counters["unsupported_text_url"] += 1
                continue
            source_bytes = read_source_archive(source_root, supported_urls[0])
            if source_bytes is None:
                counters["missing_text_file"] += 1
                continue

            try:
                decoded, actual_encoding = decode_aozora_text(
                    source_bytes,
                    (first.get("テキストファイル符号化方式") or "").strip(),
                )
            except UnicodeDecodeError:
                counters["decode_error"] += 1
                continue

            stripped = strip_aozora_metadata(decoded)
            text, ruby, malformed_ruby = extract_ruby(stripped)
            if malformed_ruby:
                counters["malformed_ruby"] += malformed_ruby
            if not text:
                counters["empty_after_cleaning"] += 1
                continue

            record: dict[str, Any] = {
                "schema_version": 1,
                "source": "aozora_bunko",
                "document_id": work_id,
                "title": (first.get("作品名") or "").strip(),
                "url": (first.get("図書カードURL") or "").strip(),
                "text": text,
                "metadata": {
                    "work_id": work_id,
                    "people": people_for_work(work_rows),
                    "orthography": (first.get("文字遣い種別") or "").strip(),
                    "published_at": (first.get("公開日") or "").strip(),
                    "updated_at": (first.get("最終更新日") or "").strip(),
                    "text_encoding": actual_encoding,
                },
            }
            if ruby:
                record["annotations"] = {"ruby": ruby}
            write_json_line(output, record)
            counters["records"] += 1
            counters["characters"] += len(text)
            counters["ruby_annotations"] += len(ruby)
            if limit and counters["records"] >= limit:
                break

    stats = {
        "schema_version": 1,
        "asset": output_path.name,
        "source": "aozora_bunko",
        "source_version": source_commit,
        "metadata_sha256": sha256_file(metadata_zip),
        "records": counters["records"],
        "characters": counters["characters"],
        "bytes": output_path.stat().st_size,
        "sha256": sha256_file(output_path),
        "warnings": {
            key: value
            for key, value in sorted(counters.items())
            if key not in {"records", "characters", "ruby_annotations"} and value
        },
        "ruby_annotations": counters["ruby_annotations"],
    }
    write_json(stats_path, stats)
    return stats
