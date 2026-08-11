from __future__ import annotations

import hashlib
import io
import json
import subprocess
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, TextIO

from .common import write_json, write_json_line
from .compression import ZstdTextWriter


GROUPS_ASSET = "homophone-groups.jsonl.zst"
OCCURRENCES_ASSET = "homophone-occurrences.jsonl.zst"
MANIFEST_ASSET = "homophone-manifest.json"
CHECKSUMS_ASSET = "HOMOPHONE-SHA256SUMS"
SENTENCE_TERMINATORS = frozenset("。！？!?")
MAX_TOKENIZER_CHUNK_CHARACTERS = 12_000
NATURALNESS_POLICY_VERSION = "conservative-v1"
DEFAULT_MIN_NATURAL_OCCURRENCES = 2
DEFAULT_MIN_NATURAL_SENTENCES = 2
MAX_NATURAL_SURFACE_CHARACTERS = 24
MAX_NATURAL_READING_CHARACTERS = 24
NATURAL_CONTENT_POS = frozenset(
    {"名詞", "動詞", "形容詞", "形状詞", "副詞", "連体詞"}
)
NATURAL_NOISE_MARKERS = (
    "http://",
    "https://",
    "www.",
    "[[",
    "]]",
    "{{",
    "}}",
    "<ref",
    "</ref>",
    "ISBN",
    "==",
    "|",
)


@dataclass
class Token:
    surface: str
    reading: str
    lemma: str
    pos: str
    subpos: str
    subsubpos: str
    start: int
    end: int
    reading_source: str = "sudachi"
    is_oov: bool = False


@dataclass
class CandidateStats:
    count: int = 0
    lemmas: Counter[str] = field(default_factory=Counter)
    parts_of_speech: Counter[str] = field(default_factory=Counter)
    sources: Counter[str] = field(default_factory=Counter)
    reading_sources: Counter[str] = field(default_factory=Counter)
    document_count: int = 0
    sentence_count: int = 0
    last_document_key: str = ""
    last_sentence_id: str = ""

    def observe(
        self,
        token: Token,
        source: str,
        document_id: str,
        sentence_id: str,
    ) -> None:
        # _iter_records emits complete documents and sentences in source order,
        # so distinct-evidence counts do not need per-candidate ID sets.
        self.count += 1
        self.lemmas[token.lemma] += 1
        self.parts_of_speech[
            "/".join((token.pos, token.subpos, token.subsubpos))
        ] += 1
        self.sources[source] += 1
        self.reading_sources[token.reading_source] += 1
        document_key = f"{source}\x1f{document_id}"
        if document_key != self.last_document_key:
            self.document_count += 1
            self.last_document_key = document_key
        if sentence_id != self.last_sentence_id:
            self.sentence_count += 1
            self.last_sentence_id = sentence_id


@dataclass
class BuildCounters:
    documents: int = 0
    sentences: int = 0
    valid_tokens: int = 0
    natural_candidate_tokens: int = 0
    reading_sources: Counter[str] = field(default_factory=Counter)
    naturalness_rejections: Counter[str] = field(default_factory=Counter)


def build_homophones(
    input_paths: list[Path],
    output_dir: Path,
    *,
    min_group_size: int = 2,
    min_candidate_count: int = 1,
    dictionary_version: str = "SudachiDict",
    pipeline_commit: str = "working-tree",
    limit_documents: int = 0,
    min_natural_occurrences: int = DEFAULT_MIN_NATURAL_OCCURRENCES,
    min_natural_sentences: int = DEFAULT_MIN_NATURAL_SENTENCES,
) -> dict[str, Any]:
    """Build a curated context corpus containing natural homophone groups.

    The source corpus is read twice. The first pass creates an exact
    reading-to-surface index after deterministic naturalness screening; the
    second pass writes only occurrences whose reading has at least
    min_group_size natural, independently attested candidate forms.
    """

    if not input_paths:
        raise ValueError("at least one input path is required")
    if min_group_size < 2:
        raise ValueError("min_group_size must be at least 2")
    if min_candidate_count < 1:
        raise ValueError("min_candidate_count must be positive")
    if limit_documents < 0:
        raise ValueError("limit_documents must not be negative")
    if min_natural_occurrences < 1:
        raise ValueError("min_natural_occurrences must be positive")
    if min_natural_sentences < 1:
        raise ValueError("min_natural_sentences must be positive")

    for input_path in input_paths:
        if not input_path.is_file():
            raise FileNotFoundError(input_path)

    tokenizer, split_mode = _create_tokenizer()
    sorted_inputs = sorted(input_paths)
    counts: dict[str, dict[str, CandidateStats]] = defaultdict(dict)
    counters = BuildCounters()

    for record in _iter_records(sorted_inputs, limit_documents):
        counters.documents += 1
        for sentence_index, (start, end) in enumerate(
            split_sentence_spans(record["text"])
        ):
            counters.sentences += 1
            sentence = record["text"][start:end]
            sentence_id = f"{record['document_id']}:s{sentence_index:06d}"
            sentence_reason = _natural_sentence_rejection_reason(sentence)
            for token in tokenize_record(
                record, start, sentence, tokenizer, split_mode
            ):
                counters.valid_tokens += 1
                counters.reading_sources[token.reading_source] += 1
                if not _contains_kanji(token.surface):
                    continue
                rejection_reason = sentence_reason or _natural_token_rejection_reason(
                    token
                )
                if rejection_reason:
                    counters.naturalness_rejections[rejection_reason] += 1
                    continue
                counters.natural_candidate_tokens += 1
                candidates = counts[token.reading]
                candidate = candidates.setdefault(token.surface, CandidateStats())
                candidate.observe(
                    token,
                    record["source"],
                    record["document_id"],
                    sentence_id,
                )

    groups, quality = _build_groups(
        counts,
        min_group_size,
        min_candidate_count,
        min_natural_occurrences,
        min_natural_sentences,
    )
    group_occurrences = sum(
        int(group["total_occurrences"]) for group in groups.values()
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    groups_path = output_dir / GROUPS_ASSET
    with ZstdTextWriter(groups_path) as stream:
        for group in groups.values():
            write_json_line(stream, group)

    occurrences_path = output_dir / OCCURRENCES_ASSET
    occurrence_count = _write_occurrences(
        sorted_inputs,
        limit_documents,
        tokenizer,
        split_mode,
        groups,
        occurrences_path,
        min_natural_occurrences=min_natural_occurrences,
        min_natural_sentences=min_natural_sentences,
    )
    if occurrence_count != group_occurrences:
        raise RuntimeError(
            "occurrence recount mismatch: "
            f"group index has {group_occurrences}, output has {occurrence_count}"
        )

    assets = [
        _asset_metadata(groups_path, len(groups)),
        _asset_metadata(occurrences_path, occurrence_count),
    ]
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "format": {
            "name": "Japanese homophone context corpus",
            "record_type": "JSONL; one group or occurrence per line",
            "encoding": "UTF-8",
            "compression": "zstd for JSONL assets",
            "offsets": (
                "Unicode code-point [start, end) offsets within the source document"
            ),
        },
        "selection": {
            "reading_normalization": (
                "Sudachi reading normalized with Unicode NFKC and "
                "katakana-to-hiragana conversion; exact Aozora ruby overrides when available"
            ),
            "surface_grouping": (
                "exact kanji-containing surface form; lemma and part of speech are retained"
            ),
            "min_group_size": min_group_size,
            "min_candidate_count": min_candidate_count,
            "candidate_definition": "A surface form with an attested token occurrence",
            "naturalness": {
                "policy": NATURALNESS_POLICY_VERSION,
                "min_occurrences_per_candidate": min_natural_occurrences,
                "min_sentences_per_candidate": min_natural_sentences,
                "content_pos": sorted(NATURAL_CONTENT_POS),
                "sentence_filter": "reject markup, URL-like, control-heavy, and excessively long contexts",
                "surface_filter": "reject OOV, affix/function, mixed ASCII/digit, and non-Japanese noise",
                "variant_filter": "keep the most frequent representative when kana/okurigana-only variants share a kanji skeleton",
                "lemma_filter": "require distinct dominant dictionary lemmas within each group",
            },
        },
        "tokenizer": {
            "implementation": "SudachiPy",
            "split_mode": "C",
            "dictionary": "SudachiDict-core or compatible Sudachi dictionary",
            "dictionary_version": dictionary_version,
        },
        "corpus": {
            "input_assets": [_input_asset_metadata(path) for path in sorted_inputs],
            "documents": counters.documents,
            "sentences": counters.sentences,
            "valid_tokens": counters.valid_tokens,
            "natural_candidate_tokens": counters.natural_candidate_tokens,
            "reading_source_counts": dict(sorted(counters.reading_sources.items())),
            "homophone_groups": len(groups),
            "candidate_forms": sum(
                int(group["candidate_count"]) for group in groups.values()
            ),
            "occurrences": occurrence_count,
            "pipeline_commit": pipeline_commit,
            **({"document_limit": limit_documents} if limit_documents else {}),
        },
        "quality": {
            **quality,
            "token_rejection_reasons": dict(
                sorted(counters.naturalness_rejections.items())
            ),
        },
        "assets": assets,
    }
    manifest_path = output_dir / MANIFEST_ASSET
    write_json(manifest_path, manifest)
    _write_checksums(output_dir, manifest)
    return manifest


def split_sentence_spans(text: str) -> list[tuple[int, int]]:
    """Return trimmed sentence spans while retaining punctuation and offsets."""

    spans: list[tuple[int, int]] = []
    start = 0
    for index, character in enumerate(text):
        if character not in SENTENCE_TERMINATORS and character != "\n":
            continue
        end = index if character == "\n" else index + 1
        _append_trimmed_span(text, start, end, spans)
        start = end
    _append_trimmed_span(text, start, len(text), spans)
    return spans


def tokenize(sentence: str, tokenizer: Any, split_mode: Any) -> Iterator[Token]:
    for morpheme in tokenizer.tokenize(sentence, split_mode):
        surface = morpheme.surface()
        parts = morpheme.part_of_speech()
        pos = parts[0] if parts else "UNK"
        if pos in {"記号", "補助記号"} or not _contains_japanese_letter(surface):
            continue

        reading = normalize_reading(morpheme.reading_form())
        if not reading:
            continue
        lemma = morpheme.dictionary_form() or surface
        is_oov_method = getattr(morpheme, "is_oov", None)
        is_oov = bool(is_oov_method()) if callable(is_oov_method) else False
        yield Token(
            surface=surface,
            reading=reading,
            lemma=lemma,
            pos=pos,
            subpos=parts[1] if len(parts) > 1 else "UNK",
            subsubpos=parts[2] if len(parts) > 2 else "UNK",
            start=morpheme.begin(),
            end=morpheme.end(),
            is_oov=is_oov,
        )


def tokenize_record(
    record: dict[str, Any],
    sentence_start: int,
    sentence: str,
    tokenizer: Any,
    split_mode: Any,
) -> Iterator[Token]:
    ruby_readings = _ruby_readings(record)
    for token in tokenize(sentence, tokenizer, split_mode):
        ruby_key = (
            sentence_start + token.start,
            sentence_start + token.end,
        )
        ruby_reading = ruby_readings.get(ruby_key)
        if ruby_reading:
            token.reading = ruby_reading
            token.reading_source = "aozora_ruby"
        yield token


def normalize_reading(reading: str) -> str:
    normalized = unicodedata.normalize("NFKC", reading)
    return "".join(_katakana_to_hiragana(character) for character in normalized)


def _ruby_readings(record: dict[str, Any]) -> dict[tuple[int, int], str]:
    result: dict[tuple[int, int], str] = {}
    annotations_value = record.get("annotations", {})
    if not isinstance(annotations_value, dict):
        return result
    annotations = annotations_value.get("ruby", [])
    if not isinstance(annotations, list):
        return result
    for annotation in annotations:
        if not isinstance(annotation, dict):
            continue
        try:
            start = int(annotation["start"])
            end = int(annotation["end"])
            reading = normalize_reading(str(annotation["reading"]))
        except (KeyError, TypeError, ValueError):
            continue
        if start >= 0 and start < end and reading:
            result[(start, end)] = reading
    return result


def _create_tokenizer() -> tuple[Any, Any]:
    try:
        from sudachipy import Dictionary, SplitMode
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "build-homophones requires SudachiPy and SudachiDict-core; "
            "install with pip install -e '.[homophones]'"
        ) from error
    return Dictionary().create(), SplitMode.C


def _build_groups(
    counts: dict[str, dict[str, CandidateStats]],
    min_group_size: int,
    min_candidate_count: int,
    min_natural_occurrences: int,
    min_natural_sentences: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    raw_candidate_forms = sum(len(candidates) for candidates in counts.values())
    candidate_rejection_reasons: Counter[str] = Counter()
    group_rejection_reasons: Counter[str] = Counter()
    rejected_candidate_forms = 0
    rejected_groups = 0
    variant_collapsed_forms = 0
    candidate_forms_after_evidence_filter = 0
    for reading in sorted(counts):
        candidates: list[dict[str, Any]] = []
        for surface in sorted(counts[reading]):
            stats = counts[reading][surface]
            if stats.count < min_candidate_count:
                rejected_candidate_forms += 1
                candidate_rejection_reasons["below_min_candidate_count"] += 1
                continue
            if stats.count < min_natural_occurrences:
                rejected_candidate_forms += 1
                candidate_rejection_reasons["insufficient_occurrences"] += 1
                continue
            if stats.sentence_count < min_natural_sentences:
                rejected_candidate_forms += 1
                candidate_rejection_reasons["insufficient_sentences"] += 1
                continue
            candidate_forms_after_evidence_filter += 1
            dominant_lemma = _dominant_counter_key(stats.lemmas)
            candidates.append(
                {
                    "surface": surface,
                    "occurrence_count": stats.count,
                    "document_count": stats.document_count,
                    "sentence_count": stats.sentence_count,
                    "lemma_count": len(stats.lemmas),
                    "dominant_lemma": dominant_lemma,
                    "lemmas": sorted(stats.lemmas),
                    "parts_of_speech": sorted(stats.parts_of_speech),
                    "source_counts": dict(sorted(stats.sources.items())),
                    "reading_source_counts": dict(
                        sorted(stats.reading_sources.items())
                    ),
                }
            )
        candidates, collapsed = _collapse_orthographic_variants(candidates)
        variant_collapsed_forms += collapsed
        if len(candidates) < min_group_size:
            rejected_groups += 1
            group_rejection_reasons["too_few_candidates"] += 1
            continue
        if len({candidate["dominant_lemma"] for candidate in candidates}) < min_group_size:
            rejected_groups += 1
            group_rejection_reasons["same_dominant_lemma"] += 1
            continue
        groups[reading] = {
            "schema_version": 1,
            "group_id": f"homophone:{reading}",
            "reading": reading,
            "candidate_count": len(candidates),
            "total_occurrences": sum(
                int(candidate["occurrence_count"]) for candidate in candidates
            ),
            "candidates": candidates,
        }
    quality = {
        "raw_reading_groups": len(counts),
        "raw_candidate_forms": raw_candidate_forms,
        "candidate_forms_after_evidence_filter": candidate_forms_after_evidence_filter,
        "rejected_candidate_forms": rejected_candidate_forms,
        "rejected_groups": rejected_groups,
        "variant_collapsed_forms": variant_collapsed_forms,
        "candidate_rejection_reasons": dict(sorted(candidate_rejection_reasons.items())),
        "group_rejection_reasons": dict(sorted(group_rejection_reasons.items())),
        "selected_groups": len(groups),
        "selected_candidate_forms": sum(
            int(group["candidate_count"]) for group in groups.values()
        ),
        "selected_occurrences": sum(
            int(group["total_occurrences"]) for group in groups.values()
        ),
    }
    return groups, quality


def _collapse_orthographic_variants(
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_key[_orthographic_key(candidate["surface"])].append(candidate)

    selected: list[dict[str, Any]] = []
    collapsed = 0
    for variants in by_key.values():
        variants.sort(
            key=lambda candidate: (
                -int(candidate["occurrence_count"]),
                -int(candidate["document_count"]),
                -int(candidate["sentence_count"]),
                candidate["surface"],
            )
        )
        selected.append(variants[0])
        collapsed += len(variants) - 1
    selected.sort(key=lambda candidate: candidate["surface"])
    return selected, collapsed


def _dominant_counter_key(counter: Counter[str]) -> str:
    if not counter:
        return ""
    return min(counter, key=lambda value: (-counter[value], value))


def _orthographic_key(surface: str) -> str:
    """Fold kana/okurigana so spelling variants do not become homophones."""

    normalized = unicodedata.normalize("NFKC", surface)
    return "".join(
        character
        for character in normalized
        if not ("\u3041" <= character <= "\u309f")
        and not ("\u30a0" <= character <= "\u30ff")
    )


def _natural_token_rejection_reason(token: Token) -> str | None:
    if token.is_oov:
        return "unknown_token"
    if token.pos not in NATURAL_CONTENT_POS:
        return "non_content_pos"
    if token.pos == "名詞" and token.subpos in {"数詞", "形式名詞"}:
        return "non_lexical_noun"
    if not _is_natural_surface(token.surface):
        return "surface_noise"
    if len(token.surface) > MAX_NATURAL_SURFACE_CHARACTERS:
        return "surface_too_long"
    if len(token.reading) > MAX_NATURAL_READING_CHARACTERS:
        return "reading_too_long"
    return None


def _natural_sentence_rejection_reason(sentence: str) -> str | None:
    if not sentence or len(sentence) > 2_000:
        return "context_too_long"
    if any(marker in sentence for marker in NATURAL_NOISE_MARKERS):
        return "noisy_context"
    if sum(character.isascii() and character.isalnum() for character in sentence) > max(
        20, len(sentence) // 4
    ):
        return "ascii_heavy_context"
    if sum(character.isdigit() for character in sentence) > max(12, len(sentence) // 5):
        return "numeric_heavy_context"
    return None


def _is_natural_surface(surface: str) -> bool:
    if not surface or surface != surface.strip():
        return False
    for character in surface:
        if (
            "\u3041" <= character <= "\u309f"
            or "\u30a0" <= character <= "\u30ff"
            or "\u3400" <= character <= "\u4dbf"
            or "\u4e00" <= character <= "\u9fff"
            or "\uf900" <= character <= "\ufaff"
            or character in "々〆"
        ):
            continue
        return False
    return True


def _write_occurrences(
    input_paths: list[Path],
    limit_documents: int,
    tokenizer: Any,
    split_mode: Any,
    groups: dict[str, dict[str, Any]],
    output_path: Path,
    *,
    min_natural_occurrences: int,
    min_natural_sentences: int,
) -> int:
    candidate_surfaces = {
        reading: {candidate["surface"] for candidate in group["candidates"]}
        for reading, group in groups.items()
    }
    occurrence_count = 0
    with ZstdTextWriter(output_path) as stream:
        for record in _iter_records(input_paths, limit_documents):
            text = record["text"]
            for sentence_index, (start, end) in enumerate(split_sentence_spans(text)):
                sentence = text[start:end]
                sentence_start = start
                sentence_end = end
                sentence_id = f"{record['document_id']}:s{sentence_index:06d}"
                sentence_reason = _natural_sentence_rejection_reason(sentence)
                for token_index, token in enumerate(
                    tokenize_record(
                        record, start, sentence, tokenizer, split_mode
                    )
                ):
                    if not _contains_kanji(token.surface):
                        continue
                    if sentence_reason or _natural_token_rejection_reason(token):
                        continue
                    if token.reading not in candidate_surfaces:
                        continue
                    if token.surface not in candidate_surfaces[token.reading]:
                        continue
                    occurrence = {
                        "schema_version": 1,
                        "group_id": groups[token.reading]["group_id"],
                        "reading": token.reading,
                        "reading_source": token.reading_source,
                        "surface": token.surface,
                        "lemma": token.lemma,
                        "pos": token.pos,
                        "subpos": token.subpos,
                        "subsubpos": token.subsubpos,
                        "source": record["source"],
                        "document_id": record["document_id"],
                        "title": record["title"],
                        "url": record["url"],
                        "sentence_id": sentence_id,
                        "sentence": sentence,
                        "sentence_start": sentence_start,
                        "sentence_end": sentence_end,
                        "target_start": sentence_start + token.start,
                        "target_end": sentence_start + token.end,
                        "token_index": token_index,
                        "left_context": sentence[: token.start],
                        "right_context": sentence[token.end :],
                    }
                    write_json_line(stream, occurrence)
                    occurrence_count += 1
    return occurrence_count


def _iter_records(
    input_paths: list[Path], limit_documents: int
) -> Iterator[dict[str, Any]]:
    seen_documents = 0
    for path in input_paths:
        for record in _iter_jsonl(path):
            required = ("source", "document_id", "title", "url", "text")
            missing = [field for field in required if field not in record]
            if missing:
                raise ValueError(f"{path} record is missing fields: {missing}")
            yield record
            seen_documents += 1
            if limit_documents and seen_documents >= limit_documents:
                return


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    process: subprocess.Popen[bytes] | None = None
    stream: TextIO
    if path.suffix == ".zst":
        process = subprocess.Popen(
            ["zstd", "--quiet", "--decompress", "--stdout", str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        if process.stdout is None:
            raise RuntimeError("zstd did not expose stdout")
        stream = io.TextIOWrapper(process.stdout, encoding="utf-8")
    else:
        stream = path.open(encoding="utf-8")

    try:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON in {path} line {line_number}") from error
    finally:
        stream.close()
        if process is not None:
            return_code = process.wait()
            # A document limit intentionally closes the decompressor before
            # EOF; zstd reports SIGPIPE in that case.
            if return_code not in (0, -13):
                raise RuntimeError(
                    f"zstd failed while reading {path}: exit code {return_code}"
                )


def _append_trimmed_span(
    text: str, start: int, end: int, output: list[tuple[int, int]]
) -> None:
    if start >= end:
        return
    value = text[start:end]
    left = len(value) - len(value.lstrip())
    right = len(value) - len(value.rstrip())
    trimmed_start = start + left
    trimmed_end = end - right
    while trimmed_end - trimmed_start > MAX_TOKENIZER_CHUNK_CHARACTERS:
        chunk_end = trimmed_start + MAX_TOKENIZER_CHUNK_CHARACTERS
        output.append((trimmed_start, chunk_end))
        trimmed_start = chunk_end
    if trimmed_start < trimmed_end:
        output.append((trimmed_start, trimmed_end))


def _contains_japanese_letter(value: str) -> bool:
    return any(
        (
            "\u3041" <= character <= "\u309f"
            or "\u30a0" <= character <= "\u30ff"
            or "\u3400" <= character <= "\u4dbf"
            or "\u4e00" <= character <= "\u9fff"
            or "\uf900" <= character <= "\ufaff"
            or character in "々〆"
        )
        for character in value
    )


def _contains_kanji(value: str) -> bool:
    return any(
        "\u3400" <= character <= "\u4dbf"
        or "\u4e00" <= character <= "\u9fff"
        or "\uf900" <= character <= "\ufaff"
        for character in value
    )


def _katakana_to_hiragana(character: str) -> str:
    if "\u30a1" <= character <= "\u30f6":
        return chr(ord(character) - 0x60)
    if character in "ヽヾ":
        return chr(ord(character) - 0x60)
    return character


def _asset_metadata(path: Path, records: int) -> dict[str, Any]:
    return {
        "name": path.name,
        "records": records,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _input_asset_metadata(path: Path) -> dict[str, Any]:
    return {
        "name": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_checksums(output_dir: Path, manifest: dict[str, Any]) -> None:
    manifest_path = output_dir / MANIFEST_ASSET
    lines = [
        f"{asset['sha256']}  {asset['name']}"
        for asset in manifest["assets"]
    ]
    lines.append(f"{_sha256_file(manifest_path)}  {MANIFEST_ASSET}")
    (output_dir / CHECKSUMS_ASSET).write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )
