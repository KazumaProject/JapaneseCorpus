from __future__ import annotations

import json
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


if __name__ == "__main__":
    unittest.main()
