from __future__ import annotations

import bz2
import json
import tempfile
import unittest
from pathlib import Path

from japanese_corpus.wikipedia import build_wikipedia

from helpers import read_zstd_jsonl


class BuildWikipediaTest(unittest.TestCase):
    def test_builds_main_namespace_records(self) -> None:
        documents = [
            {"index": {"_id": "1"}},
            {
                "namespace": 0,
                "page_id": 123,
                "title": "日本語",
                "text": "本文です。\r\n\r\n\r\n次です。",
                "timestamp": "2026-07-19T00:00:00Z",
            },
            {"index": {"_id": "2"}},
            {"namespace": 1, "page_id": 456, "title": "ノート", "text": "除外"},
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            source = directory / "source.json.bz2"
            output = directory / "wikipedia.jsonl.zst"
            stats = directory / "stats.json"
            with bz2.open(source, "wt", encoding="utf-8") as stream:
                for document in documents:
                    stream.write(json.dumps(document, ensure_ascii=False) + "\n")

            result = build_wikipedia(source, output, stats, "20260719")
            records = read_zstd_jsonl(output)

        self.assertEqual(result["records"], 1)
        self.assertEqual(result["warnings"]["non_main_namespace"], 1)
        self.assertEqual(records[0]["document_id"], "123")
        self.assertEqual(records[0]["text"], "本文です。\n\n次です。")
        self.assertEqual(records[0]["url"], "https://ja.wikipedia.org/?curid=123")


if __name__ == "__main__":
    unittest.main()
