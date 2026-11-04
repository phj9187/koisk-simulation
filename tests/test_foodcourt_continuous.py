import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from tools.run_foodcourt_until_complete import run_until_complete
from kiosk_foodcourt.cli import atomic, run_job, signature
from kiosk_foodcourt.models import Settings


class ContinuousFoodcourtTests(unittest.TestCase):
    def settings(self):
        return Settings(menu_counts=[12], attention_modes=["bounded"], visitors=[20],
                        repetitions=1, generations=2, population=3, train_days=1,
                        validation_days=1, evaluation_days=1, validation_every=1, workers=1)

    def test_resume_preserves_manifest_and_matches_uninterrupted_result(self):
        s = self.settings()
        with tempfile.TemporaryDirectory() as resumed, tempfile.TemporaryDirectory() as fresh:
            manifest = Path(resumed) / "manifest.json"
            atomic(manifest, {"signature": signature(s), "settings": s.to_dict()})
            before = manifest.read_bytes()
            with patch("kiosk_foodcourt.cli.time.monotonic", side_effect=[0, 0, 2]):
                run_job((s.to_dict(), resumed, (0, 12, "bounded", 20), 1))
            self.assertEqual(json.loads(next(Path(resumed).rglob("checkpoint.json")).read_text())["generation"], 1)
            self.assertEqual(run_until_complete(s, resumed), 1)
            self.assertEqual(run_until_complete(s, fresh), 1)
            self.assertEqual(before, manifest.read_bytes())
            left = json.loads(next(Path(resumed).rglob("result.json")).read_text())
            right = json.loads(next(Path(fresh).rglob("result.json")).read_text())
            self.assertEqual(left, right)
            # A repeat invocation skips completed training entirely.
            with patch("tools.run_foodcourt_until_complete.run_job", side_effect=AssertionError("retrained")):
                self.assertEqual(run_until_complete(s, resumed), 1)

    def test_changed_scientific_configuration_is_still_rejected(self):
        s = self.settings()
        with tempfile.TemporaryDirectory() as root:
            atomic(Path(root) / "manifest.json", {"signature": "different", "settings": s.to_dict()})
            with self.assertRaises(ValueError):
                run_until_complete(s, root)

    def test_parallel_runner_has_no_finite_deadline(self):
        s = self.settings()
        s.workers = 2
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(run_until_complete(s, root), 1)
            self.assertTrue(list(Path(root).rglob("result.json")))


if __name__ == "__main__":
    unittest.main()
