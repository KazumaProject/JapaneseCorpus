import unittest

from annotate import minimum_count_for_order
from japanese_corpus.release_verify import validate_annotated_record


class MinimumCountForOrderTest(unittest.TestCase):
    def test_unigrams_use_the_unigram_threshold(self) -> None:
        self.assertEqual(minimum_count_for_order(1, 16, 32), 16)

    def test_phrase_orders_use_the_ngram_threshold(self) -> None:
        for order in range(2, 6):
            with self.subTest(order=order):
                self.assertEqual(minimum_count_for_order(order, 16, 32), 32)

    def test_rejects_orders_outside_release_format(self) -> None:
        with self.assertRaises(ValueError):
            minimum_count_for_order(6, 16, 32)

    def test_post_download_check_uses_unicode_codepoint_offsets(self) -> None:
        record = {
            "order": 1,
            "count": 16,
            "total_observations": 100,
            "reading": "ねこ",
            "surface": "猫",
            "left_id": 1,
            "right_id": 1,
            "cost": 100,
            "source_tokens": [
                {"surface": "猫", "reading": "ねこ", "left_id": 1, "right_id": 1}
            ],
            "analysis_scope": "standalone_phrase",
            "morphology": [
                {
                    "surface": "猫",
                    "lemma": "猫",
                    "pos": "NOUN",
                    "tag": "名詞-普通名詞-一般",
                    "reading": "ネコ",
                    "start": 0,
                    "end": 1,
                }
            ],
            "bunsetsu": [{"surface": "猫", "start": 0, "end": 1}],
        }
        validate_annotated_record(record, "fixture")
        record["bunsetsu"][0]["end"] = 2
        with self.assertRaisesRegex(RuntimeError, "bunsetsu offsets"):
            validate_annotated_record(record, "fixture")


if __name__ == "__main__":
    unittest.main()
