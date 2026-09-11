"""Run independently with python3 -m unittest discover -s dashboard."""

import unittest
from pathlib import Path

from offline_help import answer_help, refine_query


class OfflineHelpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = (Path(__file__).resolve().parents[1] / "docs/offline-help.md").read_text()

    def test_exception_name_finds_missing_module(self):
        result = answer_help(self.document, "ModuleNotFoundError")
        self.assertIn("Missing module", result["matches"][0][0])
        self.assertGreater(len(result["trace"]), 1)

    def test_permission_exception(self):
        result = answer_help(self.document, "PermissionError")
        self.assertIn("Permission error", result["matches"][0][0])

    def test_typo(self):
        query, trace = refine_query(self.document, "metadtaa")
        self.assertIn("metadata", query)
        self.assertTrue(trace)

    def test_no_invented_answer(self):
        self.assertEqual(answer_help(self.document, "xyzzy123")['matches'], [])

    def test_source_lines(self):
        for title, body, line in answer_help(self.document, "recording format")['matches']:
            self.assertEqual(self.document.splitlines()[line - 1], "## " + title)
            self.assertIn(body, self.document)

    def test_bounded_and_deterministic(self):
        query = "PermissionError " * 1000
        result = answer_help(self.document, query)
        self.assertEqual(result, answer_help(self.document, query))
        self.assertLessEqual(len(result['matches']), 4)
        self.assertLessEqual(len(result['trace']), 4)

    def test_empty_question(self):
        self.assertEqual(answer_help(self.document, "how do i")['matches'], [])
