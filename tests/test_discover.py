from __future__ import annotations

import unittest

from japanese_corpus.discover import discover_wikipedia


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


if __name__ == "__main__":
    unittest.main()
