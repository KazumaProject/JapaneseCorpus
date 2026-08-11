from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from japanese_corpus.homophones import (
    build_homophones,
    normalize_reading,
    split_sentence_spans,
)

from helpers import read_zstd_jsonl

try:
    import sudachipy  # noqa: F401
except ModuleNotFoundError:
    HAS_SUDACHI = False
else:
    HAS_SUDACHI = True


@unittest.skipUnless(HAS_SUDACHI, "SudachiPy is an optional homophone dependency")
class HomophoneCorpusTest(unittest.TestCase):
    def test_builds_groups_occurrences_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            source = directory / "source.jsonl"
            source.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "source": "test",
                        "document_id": "1",
                        "title": "同音語",
                        "url": "https://example.test/1",
                        "text": (
                            "両国は交渉を続けた。高尚な話を聞いた。"
                            "交渉は難航し、高尚な理念が必要だ。"
                        ),
                        "metadata": {},
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            output_dir = directory / "homophones"
            manifest = build_homophones([source], output_dir)
            groups = read_zstd_jsonl(output_dir / "homophone-groups.jsonl.zst")
            occurrences = read_zstd_jsonl(
                output_dir / "homophone-occurrences.jsonl.zst"
            )
            checksums = (output_dir / "HOMOPHONE-SHA256SUMS").read_text(
                encoding="utf-8"
            )

        group = next(item for item in groups if item["reading"] == "こうしょう")
        self.assertEqual(
            {candidate["surface"] for candidate in group["candidates"]},
            {"交渉", "高尚"},
        )
        self.assertEqual(group["total_occurrences"], 4)
        self.assertEqual(manifest["corpus"]["occurrences"], 4)
        self.assertEqual(len(occurrences), 4)
        for occurrence in occurrences:
            self.assertEqual(
                occurrence["sentence"][
                    occurrence["target_start"] - occurrence["sentence_start"] :
                    occurrence["target_end"] - occurrence["sentence_start"]
                ],
                occurrence["surface"],
            )
        self.assertIn("homophone-manifest.json", checksums)

    def test_normalization_and_sentence_offsets(self) -> None:
        self.assertEqual(normalize_reading("コウショウ"), "こうしょう")
        text = " 交渉。 \n高尚！ "
        self.assertEqual(
            [text[start:end] for start, end in split_sentence_spans(text)],
            ["交渉。", "高尚！"],
        )

        long_text = "漢字" * 7000
        chunks = split_sentence_spans(long_text)
        self.assertGreater(len(chunks), 1)
        self.assertEqual("".join(long_text[start:end] for start, end in chunks), long_text)

    def test_uses_exact_aozora_ruby_when_it_matches_a_token(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            source = directory / "aozora.jsonl"
            source.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "source": "aozora_bunko",
                        "document_id": "ruby",
                        "title": "ルビ",
                        "url": "https://example.test/ruby",
                        "text": "山。産。",
                        "metadata": {"orthography": "新字新仮名"},
                        "annotations": {
                            "ruby": [{"start": 0, "end": 1, "reading": "さん"}]
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            manifest = build_homophones(
                [source],
                directory / "output",
                min_natural_occurrences=1,
                min_natural_sentences=1,
            )
            occurrences = read_zstd_jsonl(
                directory / "output" / "homophone-occurrences.jsonl.zst"
            )

        self.assertEqual(manifest["corpus"]["occurrences"], 2)
        ruby_occurrence = next(item for item in occurrences if item["surface"] == "山")
        self.assertEqual(ruby_occurrence["reading"], "さん")
        self.assertEqual(ruby_occurrence["reading_source"], "aozora_ruby")


if __name__ == "__main__":
    unittest.main()
