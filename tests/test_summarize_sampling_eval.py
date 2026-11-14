"""Sampling tables retain the evaluator's test-time means and step ordering."""

from __future__ import annotations

import contextlib
import csv
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import summarize_sampling_eval as summary


def evaluation(steps, value, **changes):
    return {
        "method": "otpfm_imf",
        "sampler": "flowmap",
        "n_steps_per_interval": steps,
        "protocol": "interval",
        "timepoint": "test_times",
        "n_checkpoints": 3,
        "w1_mean": value,
        "w1_sd": 9.0,
        "w1_se": 5.0,
        **changes,
    }


def write_summary(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_table(path):
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


class SamplingSummaryTests(unittest.TestCase):
    def test_flat_sweep_preserves_mean_values_and_ignores_other_metrics(self):
        values = dict(
            zip(summary.STEPS, (48.123456789, 47.0, 46.0, 45.0, 44.0, 43.0, 42.0))
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for steps, value in reversed(list(values.items())):
                rows.extend(
                    [
                        evaluation(steps, 999.0, timepoint="D8"),
                        evaluation(steps, 998.0, timepoint="hparam_val_times"),
                        evaluation(steps, 997.0, protocol="rollout"),
                        evaluation(steps, value),
                    ]
                )
            write_summary(root / "summary.csv", rows)
            (root / "per_draw.csv").write_text("not an evaluation summary\n")
            row = summary.sampling_row(root)
            self.assertEqual(
                list(row),
                [
                    "method",
                    "1 sample",
                    "2 samples",
                    "4 samples",
                    "8 samples",
                    "16 samples",
                    "32 samples",
                    "64 samples",
                ],
            )
            self.assertEqual(row["method"], "otpfm_imf")
            for steps, column in zip(summary.STEPS, summary.COLUMNS[1:]):
                self.assertEqual(row[column], values[steps])

    def test_nested_step_folders_produce_the_same_table_and_cli_uses_only_stdlib(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index, steps in enumerate(summary.STEPS):
                name = f"steps_{steps}" if index % 2 else f"flowmap_{steps}_steps"
                write_summary(
                    root / name / "summary.csv", [evaluation(steps, 10.0 + index)]
                )
            completed = subprocess.run(
                [
                    sys.executable,
                    "-S",
                    str(REPO_ROOT / "scripts" / "summarize_sampling_eval.py"),
                    str(root),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertIn("Saved otpfm_imf", completed.stdout)
            header, rows = read_table(root / "sampling_summary.csv")
            self.assertEqual(header, list(summary.COLUMNS))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["method"], "otpfm_imf")
            self.assertEqual(
                [float(rows[0][column]) for column in summary.COLUMNS[1:]],
                [10.0 + index for index in range(7)],
            )

    def test_partial_sweep_leaves_missing_counts_blank_and_labels_are_csv_quoted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_summary(
                root / "summary.csv", [evaluation(1, 0.0), evaluation(64, 4.25)]
            )
            before = (root / "summary.csv").read_bytes()
            output = root / "tables" / "result.csv"
            label = 'Bio prior, "OT"'
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                io.StringIO()
            ) as stderr:
                summary.main([str(root), "--method", label, "--output", str(output)])
            _, rows = read_table(output)
            self.assertEqual(rows[0]["method"], label)
            self.assertEqual(rows[0]["1 sample"], "0.0")
            self.assertEqual(rows[0]["64 samples"], "4.25")
            self.assertTrue(
                all(rows[0][f"{step} samples"] == "" for step in (2, 4, 8, 16, 32))
            )
            self.assertIn("[2, 4, 8, 16, 32]", stderr.getvalue())
            self.assertEqual((root / "summary.csv").read_bytes(), before)

    def test_euler_and_euler_maruyama_summaries_and_explicit_sampler_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for sampler in ("euler", "euler_maruyama"):
                with self.subTest(sampler=sampler):
                    write_summary(
                        root / "summary.csv",
                        [
                            evaluation(
                                1,
                                12.5,
                                sampler=sampler,
                                method="sf2m" if sampler == "euler_maruyama" else "mfm",
                            )
                        ],
                    )
                    self.assertEqual(summary.sampling_row(root)["1 sample"], 12.5)
            write_summary(
                root / "summary.csv",
                [evaluation(1, 7.0), evaluation(1, 12.5, sampler="euler")],
            )
            with self.assertRaisesRegex(ValueError, "multiple samplers"):
                summary.sampling_row(root)
            self.assertEqual(
                summary.sampling_row(root, sampler="flowmap")["1 sample"], 7.0
            )
            self.assertEqual(
                summary.sampling_row(root, sampler="euler")["1 sample"], 12.5
            )

    def test_multiple_methods_or_duplicate_step_results_are_not_silently_combined(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_summary(
                root / "summary.csv",
                [evaluation(1, 7.0), evaluation(2, 12.5, method="another_method")],
            )
            with self.assertRaisesRegex(ValueError, "multiple methods"):
                summary.sampling_row(root)
            write_summary(root / "summary.csv", [evaluation(1, 7.0)])
            write_summary(root / "seed_1" / "summary.csv", [evaluation(1, 8.0)])
            with self.assertRaisesRegex(ValueError, "multiple results for 1 steps"):
                summary.sampling_row(root)

    def test_missing_aggregate_old_output_and_invalid_values_report_clear_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(FileNotFoundError, "No evaluation summary.csv"):
                summary.sampling_row(root)
            (root / "summary.csv").write_text("method,w1_mean\notpfm,12.5\n")
            with self.assertRaisesRegex(ValueError, "missing evaluation columns"):
                summary.sampling_row(root)
            for changes, message in (
                ({"timepoint": "D8"}, "No interval/test_times"),
                ({"n_steps_per_interval": 50}, "No interval/test_times"),
                ({"n_steps_per_interval": "bad"}, "Invalid step count"),
                ({"w1_mean": "bad"}, "Invalid mean EMD"),
                ({"w1_mean": float("nan")}, "Non-finite mean EMD"),
                ({"w1_mean": float("inf")}, "Non-finite mean EMD"),
            ):
                with self.subTest(changes=changes):
                    write_summary(root / "summary.csv", [evaluation(1, 7.0, **changes)])
                    with self.assertRaisesRegex(ValueError, message):
                        summary.sampling_row(root)

    def test_csv_output_cannot_overwrite_its_evaluation_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "summary.csv"
            write_summary(path, [evaluation(1, 7.0)])
            before = path.read_bytes()
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(
                SystemExit
            ):
                summary.main([str(root), "--output", str(path)])
            self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
