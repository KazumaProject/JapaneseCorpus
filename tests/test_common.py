from __future__ import annotations

import unittest

from japanese_corpus.common import normalize_text


class NormalizeTextTest(unittest.TestCase):
    def test_normalizes_line_endings_controls_and_blank_lines(self) -> None:
        self.assertEqual(
            normalize_text("  日本語  \r\n\x00\r\n\r\n\r\n本文\t  "),
            "日本語\n\n本文",
        )

    def test_uses_nfc(self) -> None:
        self.assertEqual(normalize_text("カ\u3099"), "ガ")


if __name__ == "__main__":
    unittest.main()
