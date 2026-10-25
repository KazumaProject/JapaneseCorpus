import json
from pathlib import Path
import unittest

import jsonschema
import spacy

from annotate import annotate_record, minimum_count_for_order
from japanese_corpus.release_verify import validate_annotated_record


class AnnotateRecordTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        repository_root = Path(__file__).resolve().parents[2]
        schema = json.loads(
            (repository_root / "schema/ngram-annotated-record.schema.json").read_text(
                encoding="utf-8"
            )
        )
        cls.validator = jsonschema.Draft202012Validator(schema)
        cls.nlp = spacy.load("ja_ginza")

    def test_trailing_ideographic_space_is_not_required_to_be_a_morpheme(self) -> None:
        surface = "ルーマニア　"
        source = {
            "order": 1,
            "count": 46,
            "total_observations": 1_298_029_758,
            "reading": "るーまにあ",
            "surface": surface,
            "left_id": 1928,
            "right_id": 1928,
            "cost": 13724,
            "source_tokens": [
                {
                    "surface": surface,
                    "reading": "るーまにあ",
                    "left_id": 1928,
                    "right_id": 1928,
                }
            ],
        }

        annotated = annotate_record(self.nlp(surface), source, self.validator)

        self.assertEqual(annotated["morphology"][0]["surface"], "ルーマニア")
        self.assertEqual(
            (annotated["morphology"][0]["start"], annotated["morphology"][0]["end"]),
            (0, 5),
        )
        self.assertEqual(
            annotated["bunsetsu"], [{"surface": surface, "start": 0, "end": 6}]
        )


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
