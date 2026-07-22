from __future__ import annotations

import unittest

from japanese_corpus.discover import discover_jmdict, discover_wikipedia


class DiscoverWikipediaTest(unittest.TestCase):
    def test_selects_newest_completed_dump_and_all_shards(self) -> None:
        root = "https://example.test/dumps/"
        pages = {
            root: '<a href="20260719/">new</a><a href="20260712/">old</a>',
            root + "20260719/index_name=jawiki_content/": (
                '<a href="jawiki_content-20260719-00000.json.bz2">shard</a>'
            ),
            root + "20260712/index_name=jawiki_content/": (
                '<a href="_SUCCESS">ok</a>'
                '<a href="jawiki_content-20260712-00001.json.bz2">1</a>'
                '<a href="jawiki_content-20260712-00000.json.bz2">0</a>'
            ),
        }

        result = discover_wikipedia(root, pages.__getitem__)

        self.assertEqual(result["dump_date"], "20260712")
        self.assertEqual(
            [shard["sequence"] for shard in result["shards"]], ["00000", "00001"]
        )

    def test_rejects_root_without_dates(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "No dated dumps"):
            discover_wikipedia("https://example.test/", lambda _: "empty")


class DiscoverJmdictTest(unittest.TestCase):
    def test_records_remote_version_headers(self) -> None:
        result = discover_jmdict(
            "https://example.test/JMdict_e.gz",
            lambda _: {
                "etag": '"abc123"',
                "last-modified": "Wed, 22 Jul 2026 03:30:21 GMT",
                "content-length": "10512215",
            },
        )

        self.assertEqual(result["etag"], '"abc123"')
        self.assertEqual(result["bytes"], 10512215)

    def test_requires_a_version_header(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "no ETag or Last-Modified"):
            discover_jmdict("https://example.test/JMdict_e.gz", lambda _: {})


if __name__ == "__main__":
    unittest.main()
