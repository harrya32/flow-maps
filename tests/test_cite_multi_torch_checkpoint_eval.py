"""OTP-FM/MFM CITE/Multi evaluation restores preprocessing and flow weights."""

from __future__ import annotations

import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
for directory in (
    REPO_ROOT,
    REPO_ROOT / "py",
    REPO_ROOT / "tests",
    REPO_ROOT / "metric-flow-matching",
):
    sys.path.insert(0, str(directory))

from common import cite_multi, maizels_torch_checkpoint as loader
from launchers import eval_cite_multi
from mfm.flow_matchers.ema import EMA
from mfm.networks.flow_networks.mlp import VelocityNet
import otpfm_baseline  # Register the upstream experiment import paths.
from experiments.train import create_model, create_potentials, load_config
from test_cite_multi_checkpoint_eval import fixture_data, write_dataset


def save_otpfm(root, name="cite", omitted="4", clock="equal_time"):
    run = root / "otpfm_run"
    path = run / "models" / "best.pt"
    path.parent.mkdir(parents=True)
    times = cite_multi.normalized_time_map(clock)
    retained = [day for day in cite_multi.TIMEPOINTS if day != omitted]
    config = {
        **load_config("citeseq", "w2inf"),
        "dataset_name": name,
        "pca_dim": 100,
        "heldout_day": omitted,
        "cite_multi_time_mode": clock,
        "retained_days": retained,
        "time_map": times,
        "hidden_dim": 8,
        "num_hidden_layers": 1,
        "x_emb_dim": 4,
        "t_emb_dim": 4,
        "dropout": 0.0,
        "tks": [times[retained[1]]],
        "strength": 1.0,
        "evaluation_weights": "ema",
        "normalization_mean": [np.linspace(-1.0, 1.0, 100).tolist()],
        "normalization_scale": 3.0,
    }
    model = create_model(config, 100, create_potentials(config), "cpu")
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        for network, value in ((model.flownet, 1.0), (model.flownet_ema, 2.0)):
            output = [
                module
                for module in network.modules()
                if isinstance(module, torch.nn.Linear)
            ][-1]
            output.bias.fill_(value)
    torch.save({"model_state_dict": model.state_dict(), "global_step": 17}, path)
    (run / "experiment.json").write_text(json.dumps(config))
    return path


def save_mfm(
    root, name="cite", omitted="4", clock="equal_time", whiten=False, seed=4, ema=True
):
    run = root / f"mfm_run_seed{seed}"
    path = run / "checkpoints" / name / "run_id" / "flow_model" / "best.ckpt"
    path.parent.mkdir(parents=True)
    config = {
        "data_type": "scrna",
        "data_name": name,
        "dim": 100,
        "mfm": True,
        "hidden_dims_flow": [4],
        "activation_flow": "selu",
        "whiten": whiten,
        "t_exclude": [1, 2],
        "t_exclude_current": cite_multi.TIMEPOINTS.index(omitted),
        "cite_multi_time_mode": clock,
        "ema_decay": 0.99 if ema else None,
    }
    model = VelocityNet(dim=100, hidden_dims=[4], activation="selu", batch_norm=False)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.model[-1].bias.fill_(1.0)
    if ema:
        model = EMA(model, decay=0.99)
        with torch.no_grad():
            model.shadow_params[-1].fill_(float(seed + 1))
    torch.save(
        {
            "state_dict": {
                **{
                    f"flow_net.{key}": value
                    for key, value in model.state_dict().items()
                },
                "flow_matcher.geopath_net.unneeded_weight": torch.ones(7),
            },
            "global_step": 23,
        },
        path,
    )
    (run / "sweep_config.yaml").write_text(yaml.safe_dump(config))
    return path


def read_csv(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


class TorchCheckpointTests(unittest.TestCase):
    def test_all_datasets_folds_and_clocks_evaluate_all_marginals_without_training(
        self,
    ):
        _, populations = fixture_data()
        for backend in ("otpfm", "mfm"):
            for name in ("cite", "multi"):
                for omitted in ("3", "4"):
                    for clock in cite_multi.TIME_MODES:
                        with self.subTest(
                            backend=backend, name=name, omitted=omitted, clock=clock
                        ), tempfile.TemporaryDirectory() as tmp:
                            root = Path(tmp)
                            write_dataset(root, name)
                            path = (
                                save_otpfm(root, name, omitted, clock)
                                if backend == "otpfm"
                                else save_mfm(root, name, omitted, clock)
                            )
                            out = root / "eval"
                            with contextlib.ExitStack() as stack:
                                wrapped = stack.enter_context(
                                    mock.patch.object(
                                        loader,
                                        "load_checkpoint",
                                        wraps=loader.load_checkpoint,
                                    )
                                )
                                stack.enter_context(
                                    mock.patch.object(
                                        cite_multi,
                                        "timepoint_pool_splits",
                                        side_effect=AssertionError(
                                            "Training pools rebuilt"
                                        ),
                                    )
                                )
                                stack.enter_context(
                                    contextlib.redirect_stdout(io.StringIO())
                                )
                                eval_cite_multi.main(
                                    [
                                        "--backend",
                                        backend,
                                        "--checkpoint",
                                        f"{backend}:4:{path}",
                                        "--dataset_location",
                                        str(root),
                                        "--device",
                                        "cpu",
                                        "--batch_size",
                                        "2",
                                        "--n_steps_per_interval",
                                        "1",
                                        "4",
                                        "--out_dir",
                                        str(out),
                                    ]
                                )
                            self.assertEqual(wrapped.call_count, 1)
                            rows = read_csv(out / "per_checkpoint.csv")
                            sampler = "flowmap" if backend == "otpfm" else "euler"
                            self.assertEqual(len(rows), 10)
                            self.assertEqual(
                                {row["sampler"] for row in rows}, {sampler}
                            )
                            self.assertEqual(
                                {row["protocol"] for row in rows}, {"interval"}
                            )
                            metadata = json.loads((out / "metadata.json").read_text())
                            self.assertEqual(metadata["dataset"], name)
                            self.assertEqual(metadata["test_times"], [omitted])
                            self.assertEqual(metadata["metric_space"], "raw PCA100")
                            for steps in ("1", "4"):
                                by_day = {
                                    row["timepoint"]: row
                                    for row in rows
                                    if row["n_steps_per_interval"] == steps
                                }
                                for day in cite_multi.TIMEPOINTS:
                                    previous = (
                                        max(
                                            (
                                                left
                                                for left in metadata[
                                                    "retained_timepoints"
                                                ]
                                                if float(left) < float(day)
                                            ),
                                            key=float,
                                        )
                                        if day != "2"
                                        else "2"
                                    )
                                    dt = (
                                        metadata["normalized_times"][day]
                                        - metadata["normalized_times"][previous]
                                    )
                                    velocity = 6.0 if backend == "otpfm" else 5.0
                                    expected = np.linalg.norm(
                                        populations[day][0].astype(float)
                                        - populations[previous][0]
                                        - velocity * dt
                                    )
                                    self.assertAlmostEqual(
                                        float(by_day[day]["w1"]), expected, places=4
                                    )
                                    self.assertEqual(
                                        by_day[day]["source_timepoint"], previous
                                    )
                                    self.assertEqual(
                                        int(by_day[day]["source_n"]),
                                        len(populations[previous]),
                                    )
                                    self.assertEqual(
                                        int(by_day[day]["target_n"]),
                                        len(populations[day]),
                                    )
                                self.assertEqual(
                                    by_day["test_times"]["w1"], by_day[omitted]["w1"]
                                )
                            records = json.loads((out / "metrics.json").read_text())
                            self.assertEqual(len(records), 2)
                            self.assertTrue(
                                all(
                                    f"final_eval/{sampler}_mean_emd_test_times"
                                    in record["metrics"]
                                    for record in records
                                )
                            )

    def test_otpfm_can_select_euler_and_instantaneous_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_dataset(root, "cite")
            path = save_otpfm(root)
            out = root / "eval"
            with contextlib.redirect_stdout(io.StringIO()):
                eval_cite_multi.main(
                    [
                        "--backend",
                        "otpfm",
                        "--checkpoint",
                        f"otpfm:4:{path}",
                        "--dataset_location",
                        str(root),
                        "--device",
                        "cpu",
                        "--euler",
                        "--no-ema",
                        "--n_steps_per_interval",
                        "4",
                        "--out_dir",
                        str(out),
                    ]
                )
            rows = {
                row["timepoint"]: row for row in read_csv(out / "per_checkpoint.csv")
            }
            # D3 (0.25) -> D4 (0.5), raw velocity 1 * normalization scale 3.
            self.assertAlmostEqual(float(rows["4"]["w1"]), 7.5, places=4)
            self.assertEqual({row["sampler"] for row in rows.values()}, {"euler"})
            self.assertEqual(
                json.loads((out / "metadata.json").read_text())["checkpoints"][0][
                    "parameters"
                ],
                "instantaneous",
            )

    def test_mfm_whitening_is_reconstructed_and_emd_is_scored_in_raw_pca(self):
        from sklearn.preprocessing import StandardScaler

        data, _ = fixture_data()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_dataset(root, "multi", data)
            path = save_mfm(
                root, name="multi", clock="real_time", whiten=True, ema=False
            )
            out = root / "eval"
            with contextlib.redirect_stdout(io.StringIO()):
                eval_cite_multi.main(
                    [
                        "--backend",
                        "mfm",
                        "--checkpoint",
                        f"mfm:4:{path}",
                        "--dataset_location",
                        str(root),
                        "--device",
                        "cpu",
                        "--n_steps_per_interval",
                        "2",
                        "8",
                        "--out_dir",
                        str(out),
                    ]
                )
            metadata = json.loads((out / "metadata.json").read_text())
            scale = StandardScaler().fit(data["x"]).scale_
            np.testing.assert_allclose(
                metadata["checkpoints"][0]["whitening_scale"], scale
            )
            rows = read_csv(out / "per_checkpoint.csv")
            for row in rows:
                if row["timepoint"] == "4":
                    expected = np.linalg.norm(np.full(100, 0.25) - 0.2 * scale)
                    self.assertAlmostEqual(float(row["w1"]), expected, places=4)

    def test_multiple_seeds_are_summarized_separately_from_sampling_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_dataset(root, "cite")
            first, second = save_mfm(root, seed=0), save_mfm(root, seed=4)
            out = root / "eval"
            with contextlib.redirect_stdout(io.StringIO()):
                eval_cite_multi.main(
                    [
                        "--backend",
                        "mfm",
                        "--checkpoint",
                        f"mfm:0:{first}",
                        "--checkpoint",
                        f"mfm:4:{second}",
                        "--dataset_location",
                        str(root),
                        "--device",
                        "cpu",
                        "--n_steps_per_interval",
                        "1",
                        "4",
                        "--out_dir",
                        str(out),
                    ]
                )
            rows = read_csv(out / "per_checkpoint.csv")
            summary = read_csv(out / "summary.csv")
            self.assertEqual(len(rows), 20)
            self.assertEqual(len(summary), 10)
            for row in summary:
                self.assertEqual(int(row["n_checkpoints"]), 2)
                values = [
                    float(value["w1"])
                    for value in rows
                    if value["timepoint"] == row["timepoint"]
                    and value["n_steps_per_interval"] == row["n_steps_per_interval"]
                ]
                self.assertAlmostEqual(float(row["w1_mean"]), np.mean(values))
                self.assertAlmostEqual(
                    float(row["w1_se"]), np.std(values, ddof=1) / np.sqrt(2)
                )


if __name__ == "__main__":
    unittest.main()
