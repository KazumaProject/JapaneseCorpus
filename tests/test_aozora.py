from __future__ import annotations

import csv
import io
import tempfile
import unittest
import zipfile
from pathlib import Path

from japanese_corpus.aozora import (
    build_aozora,
    extract_ruby,
    selected_source_paths,
    strip_aozora_metadata,
)

from helpers import read_zstd_jsonl


CSV_FIELDS = [
    "作品ID",
    "作品名",
    "作品著作権フラグ",
    "公開日",
    "最終更新日",
    "図書カードURL",
    "人物ID",
    "姓",
    "名",
    "役割フラグ",
    "人物著作権フラグ",
    "文字遣い種別",
    "テキストファイルURL",
    "テキストファイル符号化方式",
]


def metadata_row(**overrides: str) -> dict[str, str]:
    row = {
        "作品ID": "000001",
        "作品名": "試験作品",
        "作品著作権フラグ": "なし",
        "公開日": "2026-01-01",
        "最終更新日": "2026-01-02",
        "図書カードURL": "https://www.aozora.gr.jp/cards/000001/card1.html",
        "人物ID": "000001",
        "姓": "試験",
        "名": "太郎",
        "役割フラグ": "著者",
        "人物著作権フラグ": "なし",
        "文字遣い種別": "新字新仮名",
        "テキストファイルURL": "https://www.aozora.gr.jp/cards/000001/files/1_txt_1.zip",
        "テキストファイル符号化方式": "ShiftJIS",
    }
    row.update(overrides)
    return row


class AozoraCleaningTest(unittest.TestCase):
    def test_strips_metadata_and_preserves_body(self) -> None:
        source = (
            "作品\n著者\n"
            "-------------------------------------------------------\n"
            "【テキスト中に現れる記号について】\n注記\n"
            "-------------------------------------------------------\n\n"
            "本文［＃改ページ］です。\n\n\n次。\n"
            "底本：試験本\n入力：人\n"
        )
        self.assertEqual(strip_aozora_metadata(source), "本文です。\n\n次。")

    def test_extracts_explicit_and_implicit_ruby(self) -> None:
        text, ruby, malformed = extract_ruby("｜今日《きょう》は山《やま》。")
        self.assertEqual(text, "今日は山。")
        self.assertEqual(malformed, 0)
        self.assertEqual(
            ruby,
            [
                {"start": 0, "end": 2, "reading": "きょう"},
                {"start": 3, "end": 4, "reading": "やま"},
            ],
        )


class BuildAozoraTest(unittest.TestCase):
    def test_filters_copyrighted_work_and_builds_public_domain_work(self) -> None:
        rows = [
            metadata_row(),
            metadata_row(
                人物ID="000002", 姓="翻訳", 名="花子", 役割フラグ="翻訳者"
            ),
            metadata_row(
                作品ID="000002",
                作品名="除外作品",
                作品著作権フラグ="あり",
                人物ID="000003",
                人物著作権フラグ="あり",
                テキストファイルURL=(
                    "https://www.aozora.gr.jp/cards/000003/files/2_txt_1.zip"
                ),
            ),
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            metadata_zip = directory / "metadata.zip"
            csv_buffer = io.StringIO(newline="")
            writer = csv.DictWriter(csv_buffer, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
            with zipfile.ZipFile(metadata_zip, "w") as archive:
                archive.writestr("metadata.csv", "\ufeff" + csv_buffer.getvalue())

            text_path = directory / "source/cards/000001/files/1_txt_1.zip"
            text_path.parent.mkdir(parents=True)
            with zipfile.ZipFile(text_path, "w") as archive:
                archive.writestr(
                    "1_txt_1.txt",
                    (
                    "作品\r\n著者\r\n"
                    "-------------------------------------------------------\r\n"
                    "【テキスト中に現れる記号について】\r\n"
                    "-------------------------------------------------------\r\n"
                    "｜今日《きょう》は山《やま》。\r\n"
                    "底本：試験本\r\n"
                    ).encode("cp932"),
                )
            output = directory / "aozora.jsonl.zst"
            stats_path = directory / "stats.json"

            self.assertEqual(
                selected_source_paths(metadata_zip),
                ["cards/000001/files/1_txt_1.zip"],
            )

            stats = build_aozora(
                metadata_zip,
                directory / "source",
                output,
                stats_path,
                "a" * 40,
            )
            records = read_zstd_jsonl(output)

        self.assertEqual(stats["records"], 1)
        self.assertEqual(stats["warnings"]["excluded_by_copyright"], 1)
        self.assertEqual(records[0]["text"], "今日は山。")
        self.assertEqual(len(records[0]["metadata"]["people"]), 2)
        self.assertEqual(len(records[0]["annotations"]["ruby"]), 2)


if __name__ == "__main__":
    unittest.main()
