import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(name):
    return [json.loads(line) for line in (ROOT / "data" / name).read_text().splitlines()]


class DatasetIntegrityTests(unittest.TestCase):
    def test_scored_synthetic_rows_are_exactly_unique(self):
        key_functions = {
            "stage1_eval.jsonl": lambda row: row["text"],
            "stage2_eval.jsonl": lambda row: row["chunk"],
            "stage3_eval.jsonl": lambda row: (row["context"], row["response"]),
            "stage3_hard_eval.jsonl": lambda row: (row["context"], row["response"]),
            "pii_eval.jsonl": lambda row: row["text"],
        }
        for filename, key in key_functions.items():
            with self.subTest(filename=filename):
                rows = load(filename)
                self.assertEqual(len(rows), len({key(row) for row in rows}))

    def test_easy_and_hard_fact_contexts_are_disjoint(self):
        easy = {row["context"] for row in load("stage3_eval.jsonl")}
        hard = {row["context"] for row in load("stage3_hard_eval.jsonl")}
        self.assertTrue(easy.isdisjoint(hard))


if __name__ == "__main__":
    unittest.main()
