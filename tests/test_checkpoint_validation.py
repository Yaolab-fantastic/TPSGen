import csv
import tempfile
import unittest
from pathlib import Path

from tpsgen.evaluation.checkpoint_validation import evaluate_checkpoint_predictions


class TestCheckpointValidation(unittest.TestCase):
    def _write(self, path: Path) -> None:
        fields = ["record_id"] + [
            f"{kind}_{tissue}"
            for tissue in ("root", "stem", "leaf", "fruit")
            for kind in ("label", "prediction")
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for index, value in enumerate((1.0, 2.0, 4.0)):
                row = {"record_id": f"r{index}"}
                for tissue in ("root", "stem", "leaf", "fruit"):
                    row[f"label_{tissue}"] = value
                    row[f"prediction_{tissue}"] = 2 * value + 1
                writer.writerow(row)

    def test_reports_distinct_r_squared_definitions_and_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "evaluation.csv"
            output = Path(directory) / "metrics.json"
            self._write(source)
            result = evaluate_checkpoint_predictions(source, output, expected_records=3)
        fruit = result["metrics"]["fruit"]
        self.assertAlmostEqual(fruit["pearson_r"], 1.0)
        self.assertAlmostEqual(fruit["linear_regression_r_squared"], 1.0)
        self.assertEqual(len(result["input_sha256"]), 64)

    def test_expected_record_count_is_verified(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "evaluation.csv"
            self._write(source)
            with self.assertRaisesRegex(ValueError, "Expected 3703 records, found 3"):
                evaluate_checkpoint_predictions(source, Path(directory) / "out.json", expected_records=3703)

    def test_released_activity_validation_is_3703_record_cohort(self) -> None:
        root = Path(__file__).resolve().parents[1]
        directory = root / "data" / "results" / "activity_validation_20260924"
        source = directory / "observed_vs_predicted.csv"
        manifest = directory / "validation_manifest.json"
        if not source.exists() or not manifest.exists():
            self.fail("released activity validation evidence is missing")
        result = evaluate_checkpoint_predictions(source, directory / "_test_audit.json", expected_records=3703)
        self.assertEqual(result["cohort_role"], "checkpoint_evaluation; independence not inferred")
        self.assertAlmostEqual(result["metrics"]["fruit"]["pearson_r"], 0.7924056785, places=6)
        (directory / "_test_audit.json").unlink()


if __name__ == "__main__":
    unittest.main()
