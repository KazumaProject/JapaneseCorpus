from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from japanese_corpus.common import write_json
from japanese_corpus.manifest import build_manifest, verify_remote_assets


class ManifestTest(unittest.TestCase):
    def test_builds_manifest_checksums_and_verifies_remote_assets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            stats_directory = directory / "stats"
            stats_directory.mkdir()
            write_json(
                stats_directory / "wiki.json",
                {
                    "asset": "wikipedia.jsonl.zst",
                    "source": "wikipedia",
                    "source_version": "20260719",
                    "records": 2,
                    "characters": 20,
                    "bytes": 10,
                    "sha256": "a" * 64,
                    "warnings": {},
                },
            )
            write_json(
                stats_directory / "aozora.json",
                {
                    "asset": "aozora.jsonl.zst",
                    "source": "aozora_bunko",
                    "source_version": "b" * 40,
                    "records": 1,
                    "characters": 5,
                    "bytes": 8,
                    "sha256": "c" * 64,
                    "warnings": {},
                },
            )
            discovery = directory / "discovery.json"
            write_json(
                discovery,
                {
                    "wikipedia": {
                        "dump_date": "20260719",
                        "index_url": "https://example.test/wiki/",
                    },
                    "aozora": {
                        "repository_url": "https://example.test/aozora.git",
                        "repository_commit": "b" * 40,
                        "metadata_url": "https://example.test/metadata.zip",
                    },
                },
            )
            manifest_path = directory / "manifest.json"
            checksums_path = directory / "SHA256SUMS"
            manifest = build_manifest(
                stats_directory,
                discovery,
                manifest_path,
                checksums_path,
                "v2026.0722.1",
                "d" * 40,
                "2026-07-22T00:00:00Z",
            )
            assets_json = directory / "assets.json"
            assets_json.write_text(
                json.dumps(
                    {
                        "assets": [
                            {"name": "wikipedia.jsonl.zst", "size": 10},
                            {"name": "aozora.jsonl.zst", "size": 8},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            verify_remote_assets(stats_directory, assets_json)

            self.assertIn(
                "wikipedia.jsonl.zst", checksums_path.read_text(encoding="utf-8")
            )

        self.assertEqual(manifest["totals"]["records"], 3)

    def test_embeds_and_checks_dictionary_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            stats_directory = directory / "stats"
            stats_directory.mkdir()
            for name, source in (("wiki", "wikipedia"), ("aozora", "aozora_bunko")):
                write_json(
                    stats_directory / f"{name}.json",
                    {
                        "asset": f"{name}.jsonl.zst",
                        "source": source,
                        "records": 1,
                        "characters": 1,
                        "bytes": 1,
                        "sha256": "a" * 64,
                    },
                )
            discovery = directory / "discovery.json"
            write_json(
                discovery,
                {
                    "wikipedia": {"dump_date": "20260719", "index_url": "https://w"},
                    "aozora": {
                        "repository_url": "https://a",
                        "repository_commit": "b" * 40,
                        "metadata_url": "https://m",
                    },
                },
            )

            dictionary_asset = b"compressed dictionary"
            dictionary_digest = hashlib.sha256(dictionary_asset).hexdigest()
            dictionary_manifest_path = directory / "ngram-manifest.json"
            write_json(
                dictionary_manifest_path,
                {
                    "schema_version": 1,
                    "format": {},
                    "tokenizer": {},
                    "parameters": {},
                    "corpus": {},
                    "orders": [
                        {"order": 1, "retained_entries": 7},
                        {"order": 2, "retained_entries": 5},
                        {"order": 3, "retained_entries": 3},
                    ],
                    "assets": [
                        {
                            "name": "mozc-unigram-00000.txt.zst",
                            "bytes": len(dictionary_asset),
                            "sha256": dictionary_digest,
                        }
                    ],
                },
            )
            manifest_digest = hashlib.sha256(
                dictionary_manifest_path.read_bytes()
            ).hexdigest()
            dictionary_sums_path = directory / "NGRAM-SHA256SUMS"
            dictionary_sums_path.write_text(
                f"{dictionary_digest}  mozc-unigram-00000.txt.zst\n"
                f"{manifest_digest}  ngram-manifest.json\n",
                encoding="utf-8",
            )

            english_asset = b"compressed English dictionary"
            english_digest = hashlib.sha256(english_asset).hexdigest()
            english_manifest_path = directory / "english-dictionary-manifest.json"
            write_json(
                english_manifest_path,
                {
                    "schema_version": 1,
                    "format": {},
                    "source": {
                        "name": "JMdict_e",
                        "url": "https://example.test/JMdict_e.gz",
                        "created": "2026-07-22",
                        "etag": '"abc"',
                        "last_modified": "Wed, 22 Jul 2026 03:30:21 GMT",
                        "bytes": 100,
                        "sha256": "e" * 64,
                        "license": "CC-BY-SA-4.0",
                    },
                    "parameters": {},
                    "counts": {
                        "jmdict_entries": 10,
                        "katakana_entries": 4,
                        "unique_readings": 5,
                        "retained_entries": 9,
                    },
                    "build": {},
                    "assets": [
                        {
                            "name": "mozc-english-unigram-00000.txt.zst",
                            "bytes": len(english_asset),
                            "sha256": english_digest,
                        }
                    ],
                },
            )
            english_manifest_digest = hashlib.sha256(
                english_manifest_path.read_bytes()
            ).hexdigest()
            english_sums_path = directory / "ENGLISH-DICTIONARY-SHA256SUMS"
            english_sums_path.write_text(
                f"{english_digest}  mozc-english-unigram-00000.txt.zst\n"
                f"{english_manifest_digest}  english-dictionary-manifest.json\n",
                encoding="utf-8",
            )

            ajimee_report_path = directory / "ajimee-bench-report.json"
            write_json(
                ajimee_report_path,
                {
                    "schema_version": 1,
                    "benchmark": {
                        "name": "AJIMEE-Bench",
                        "dataset": "JWTD_v2/v1/evaluation_items.json",
                        "repository_url": "https://github.com/azooKey/AJIMEE-Bench",
                        "commit": "f" * 40,
                        "sha256": "e" * 64,
                        "license": "CC-BY-SA-3.0",
                        "items": 200,
                    },
                    "engine": {
                        "candidate_limit": 1,
                        "context_mode": "ignored",
                        "dictionary_entries": 24,
                    },
                    "metrics": {
                        "overall": {
                            "items": 200,
                            "correct_at_1": 107,
                            "accuracy_at_1": 0.535,
                            "mean_min_cer": 0.0715,
                        },
                        "with_context": {
                            "items": 100,
                            "correct_at_1": 44,
                            "accuracy_at_1": 0.44,
                            "mean_min_cer": 0.0991,
                        },
                        "without_context": {
                            "items": 100,
                            "correct_at_1": 63,
                            "accuracy_at_1": 0.63,
                            "mean_min_cer": 0.0439,
                        },
                    },
                },
            )

            manifest = build_manifest(
                stats_directory,
                discovery,
                directory / "manifest.json",
                directory / "SHA256SUMS",
                "v1",
                "c" * 40,
                "2026-07-22T00:00:00Z",
                dictionary_manifest_path,
                dictionary_sums_path,
                english_manifest_path,
                english_sums_path,
                ajimee_report_path,
            )

            self.assertEqual(manifest["totals"]["dictionary_entries"], 15)
            self.assertEqual(
                manifest["dictionaries"]["metadata_assets"],
                ["ngram-manifest.json", "NGRAM-SHA256SUMS"],
            )
            checksums = (directory / "SHA256SUMS").read_text(encoding="utf-8")
            self.assertIn("mozc-unigram-00000.txt.zst", checksums)
            self.assertIn("NGRAM-SHA256SUMS", checksums)
            self.assertEqual(manifest["totals"]["english_dictionary_entries"], 9)
            self.assertEqual(
                manifest["english_dictionary"]["metadata_assets"],
                [
                    "english-dictionary-manifest.json",
                    "ENGLISH-DICTIONARY-SHA256SUMS",
                ],
            )
            self.assertEqual(manifest["sources"]["jmdict"]["created"], "2026-07-22")
            self.assertIn("mozc-english-unigram-00000.txt.zst", checksums)
            self.assertEqual(
                manifest["benchmarks"]["ajimee"]["metrics"]["overall"][
                    "correct_at_1"
                ],
                107,
            )
            self.assertIn("ajimee-bench-report.json", checksums)


if __name__ == "__main__":
    unittest.main()
