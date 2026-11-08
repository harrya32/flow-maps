"""Maizels OTP-FM coupling, held-out cells, and evaluation integration."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from otpfm_baseline.data import prepare_data
from otpfm_baseline.evaluation import classifier_paths, evaluate
from otpfm_baseline.train import experiment_config, main, parse_args, run_experiment
from common import maizels
from experiments.citeseq.data import compute_ot_alignments
from experiments.train import create_model, create_potentials, load_config
from otpfm.potentials import W2InfPotential, W2Potential


def fixture_data():
    rng = np.random.default_rng(13)
    days = np.repeat(maizels.TIMEPOINTS, 40)
    x = np.concatenate(
        [
            rng.normal(maizels.parse_timepoint(day), 0.2, (40, 50))
            for day in maizels.TIMEPOINTS
        ]
    ).astype(np.float32)
    return {
        "x": x,
        "timepoints": days,
        "time_values": np.array([maizels.parse_timepoint(day) for day in days]),
        "cell_types": np.full(len(days), maizels.CLASS_NAMES[0], dtype=object),
        "obs_names": np.arange(len(days)).astype(str),
    }


def prepared(
    config, *, schedule="d3_d3p8_d8", clock="real_time", smoke=False, raw=None
):
    with patch.object(
        maizels, "load_pca50_dataset", return_value=raw or fixture_data()
    ):
        return prepare_data(
            "maizels",
            None,
            clock,
            "/private/tmp/fixture_maizels.csv.gz",
            config,
            maizels_schedule=schedule,
            smoke_test=smoke,
        )


class RecordingSampler:
    def __init__(self):
        self.calls = []

    def sample(
        self,
        source,
        *,
        start_time=0.0,
        end_time=1.0,
        n_steps=50,
        sampler="direct",
        paths=False,
    ):
        self.calls.append(
            (source.copy(), start_time, end_time, n_steps, sampler, paths)
        )
        offset = {"direct": 0.0, "flowmap": 1.0, "euler": 2.0}[sampler]
        if paths:
            return source[:, None] + np.linspace(0, offset, n_steps + 1)[None, :, None]
        return source + offset


class OTPFMMaizelsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_defaults_reuse_cite_hyperparameters_with_50_dimensions(self):
        for argv, preset in (
            ([], "W2Inf"),
            (["--ot_coupling"], "W2"),
            (["--potential", "w2"], None),
            (["--potential", "w2", "--ot_coupling"], None),
        ):
            with self.subTest(argv=argv):
                args = parse_args(["--dataset_name", "maizels", *argv])
                config = experiment_config(args)
                expected = load_config("citeseq", preset)
                for key, value in expected.items():
                    if key not in (
                        "pca_dim",
                        "solver_type",
                        "potential",
                        "ot_coupling",
                    ):
                        self.assertEqual(config[key], value, key)
                self.assertEqual(config["pca_dim"], 50)
                self.assertEqual(args.maizels_schedule, "d3_d3p8_d8")
                self.assertEqual(args.maizels_time_mode, "real_time")

    def test_retained_tuples_and_normalization_exclude_validation_and_omitted_cells(
        self,
    ):
        config = experiment_config(parse_args(["--dataset_name", "maizels"]))
        for schedule, retained in (
            ("d3_d3p8_d8", ("D3", "D3.8", "D8")),
            ("d3_d8", ("D3", "D8")),
        ):
            for clock in ("real_time", "equal_time"):
                with self.subTest(schedule=schedule, clock=clock):
                    data = prepared(config, schedule=schedule, clock=clock)
                    self.assertEqual(data.retained, retained)
                    self.assertEqual(
                        data.train_dataset.train_times,
                        [data.day_indices[d] for d in retained],
                    )
                    self.assertEqual(data.time_map["D3"], 0.0)
                    self.assertEqual(data.time_map["D8"], 1.0)
                    if len(retained) == 3:
                        self.assertAlmostEqual(
                            data.time_map["D3.8"], 0.16 if clock == "real_time" else 0.5
                        )
                    train = np.concatenate(
                        [data.timepoint_splits[d]["train_x"] for d in retained]
                    )
                    np.testing.assert_allclose(
                        data.scaler.mean_, train.mean(axis=0, keepdims=True)
                    )
                    changed = fixture_data()
                    for day in maizels.TIMEPOINTS:
                        pool = data.timepoint_splits[day]
                        row_ids = np.flatnonzero(changed["timepoints"] == day)
                        altered = (
                            row_ids[pool["holdout_idx"]] if day in retained else row_ids
                        )
                        changed["x"][altered] += 10000
                        if day in retained:
                            self.assertFalse(
                                set(pool["train_idx"]) & set(pool["holdout_idx"])
                            )
                            for dataset, key in (
                                (data.train_dataset, "train_x"),
                                (data.val_dataset, "holdout_x"),
                            ):
                                values = dataset.data_by_time[
                                    data.day_indices[day]
                                ].numpy()
                                np.testing.assert_allclose(
                                    data.scaler.inverse_transform(values),
                                    pool[key],
                                    atol=1e-6,
                                )
                    second = prepared(
                        config, schedule=schedule, clock=clock, raw=changed
                    )
                    np.testing.assert_array_equal(
                        data.scaler.mean_, second.scaler.mean_
                    )
                    self.assertEqual(data.scaler.scale_, second.scaler.scale_)
                    batch = next(iter(data.train_loader))
                    self.assertEqual(len(batch), len(retained))
                    self.assertEqual(batch[0].shape[-1], 50)

    def test_ot_chains_are_built_separately_for_training_and_validation(self):
        config = experiment_config(
            parse_args(["--dataset_name", "maizels", "--ot_coupling"])
        )
        for schedule in ("d3_d3p8_d8", "d3_d8"):
            with self.subTest(schedule=schedule), patch(
                "otpfm_baseline.data.compute_ot_alignments", wraps=compute_ot_alignments
            ) as solve:
                data = prepared(config, schedule=schedule)
            self.assertEqual(solve.call_count, 2)
            for call, key, dataset in zip(
                solve.call_args_list,
                ("train_x", "holdout_x"),
                (data.train_dataset, data.val_dataset),
            ):
                pcs, labels, times = call.args
                raw = np.concatenate(
                    [data.timepoint_splits[d][key] for d in data.retained]
                )
                np.testing.assert_allclose(
                    pcs, data.scaler.transform(raw).astype(np.float32)
                )
                self.assertEqual(
                    set(labels), {data.day_indices[d] for d in data.retained}
                )
                self.assertEqual(times, dataset.train_times)
                self.assertEqual(call.kwargs, {"method": "emd", "reg": 0.01})
                self.assertTrue(dataset.use_ot)
                self.assertEqual(len(dataset), len(data.timepoint_splits["D3"][key]))
                for k, edge in enumerate(zip(times[:-1], times[1:])):
                    np.testing.assert_array_equal(
                        dataset.ot_chains[:, k + 1],
                        dataset.ot_alignments[edge][dataset.ot_chains[:, k]],
                    )

    def test_smoke_ot_caps_populations_and_preserves_sinkhorn_options(self):
        config = experiment_config(
            parse_args(
                [
                    "--dataset_name",
                    "maizels",
                    "--ot_coupling",
                    "--ot_method",
                    "sinkhorn",
                    "--ot_reg",
                    "0.05",
                    "--batch_size",
                    "4",
                ]
            )
        )
        with patch(
            "otpfm_baseline.data.compute_ot_alignments", wraps=compute_ot_alignments
        ) as solve:
            data = prepared(config, smoke=True)
        self.assertEqual(len(data.train_loader), 1)
        self.assertEqual(len(data.val_loader), 1)
        for call in solve.call_args_list:
            _, labels, times = call.args
            self.assertTrue(all(np.count_nonzero(labels == t) <= 4 for t in times))
            self.assertEqual(call.kwargs, {"method": "sinkhorn", "reg": 0.05})

    def test_upstream_full_strength_losses_and_endpoint_case_backpropagate(self):
        for potential, cls in (("w2inf", W2InfPotential), ("w2", W2Potential)):
            for schedule in ("d3_d3p8_d8", "d3_d8"):
                with self.subTest(potential=potential, schedule=schedule):
                    config = experiment_config(
                        parse_args(
                            [
                                "--dataset_name",
                                "maizels",
                                "--potential",
                                potential,
                                "--batch_size",
                                "4",
                            ]
                        )
                    )
                    data = prepared(config, schedule=schedule, smoke=True)
                    config.update(
                        hidden_dim=8,
                        num_hidden_layers=1,
                        dropout=0.0,
                        lossfn="mse",
                        tks=[data.time_map[d] for d in data.retained[1:-1]],
                    )
                    potentials = create_potentials(config)
                    self.assertEqual(len(potentials), len(data.retained) - 2)
                    if potentials:
                        self.assertIsInstance(next(iter(potentials.values())), cls)
                    model = create_model(config, 50, potentials, "cpu")
                    batch = torch.stack(next(iter(data.train_loader))).transpose(0, 1)
                    loss = model.forward_with_loss(
                        batch, otp_alpha=1.0, do_otp=bool(potentials)
                    )
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward()
                    grads = [
                        p.grad for p in model.flownet.parameters() if p.grad is not None
                    ]
                    self.assertTrue(grads)
                    self.assertTrue(all(torch.isfinite(g).all() for g in grads))

    def test_evaluation_uses_nearest_retained_source_raw_units_and_both_classifiers(
        self,
    ):
        config = experiment_config(parse_args(["--dataset_name", "maizels"]))
        data = prepared(config)
        sampler = RecordingSampler()

        def score(paths, source_types, **kwargs):
            self.assertEqual(paths.shape, (4, 50, 50))
            self.assertEqual(len(source_types), 4)
            self.assertEqual(kwargs["lineage_transition_mode"], "descendant")
            self.assertEqual(kwargs["prob_threshold"], 0.0)
            self.assertEqual(kwargs["margin_threshold"], 0.0)
            return {"valid": np.array([True, False, True, False])}

        with patch(
            "cellmnn_baseline.evaluation.wasserstein.exact_emd",
            side_effect=lambda x, y: float(x.mean()),
        ), patch(
            "cellmnn_baseline.evaluation.rbf_mmd2", return_value=0.125
        ), patch.object(
            maizels, "check_paths_with_classifier", side_effect=score
        ) as classify:
            metrics, figures = evaluate(
                data,
                sampler,
                {"observed_days": "observed.npz", "all_days": "all.npz"},
                include_observed=True,
                make_plot=False,
            )
        self.assertEqual(classify.call_count, 6)
        self.assertEqual(figures, {})
        expected_source = data.timepoint_splits["D3.8"]["x"]
        self.assertAlmostEqual(
            metrics["distribution_eval/D6_flowmap_emd"],
            float((expected_source + 1).mean()),
        )
        self.assertEqual(
            metrics["OTPFM/test_EMD"], metrics["distribution_eval/flowmap_emd_mean"]
        )
        self.assertNotIn("cellmnn/test_EMD", metrics)
        self.assertEqual(
            metrics["final_eval/observed_evaluation_mean_emd"],
            metrics["final_eval/observed_flowmap_mean_emd"],
        )
        self.assertEqual(
            metrics["final_eval/D3_to_D8_rollout_evaluation_emd"],
            metrics["final_eval/D3_to_D8_rollout_flowmap_emd"],
        )
        self.assertEqual(
            metrics[
                "maizels/full_data_classifier/model_flowmap_invalid_trajectory_pct"
            ],
            50.0,
        )
        omitted = [d for d in maizels.TIMEPOINTS if d not in data.retained]
        for day in omitted:
            left = "D3" if maizels.parse_timepoint(day) < 3.8 else "D3.8"
            calls = [
                c
                for c in sampler.calls
                if c[4] == "flowmap" and c[2] == data.time_map[day]
            ]
            self.assertEqual(len(calls), 1)
            source, start, _, steps, _, paths = calls[0]
            np.testing.assert_array_equal(source, data.timepoint_splits[left]["x"])
            self.assertEqual(start, data.time_map[left])
            self.assertEqual(steps, 50)
            self.assertFalse(paths)
        for source, _, _, steps, _, paths in sampler.calls:
            if paths:
                np.testing.assert_array_equal(
                    source, data.timepoint_splits["D3"]["holdout_x"]
                )
                self.assertEqual(steps, 50)

    def test_classifier_selection_matches_shared_maizels_schedules(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            for name in (
                "celltype_classifier_pca50",
                "celltype_classifier_pca50_d3_d3p8_d8",
                "custom",
            ):
                (root / f"{name}.npz").touch()
            with patch("cellmnn_baseline.evaluation.REPO_ROOT", root):
                for schedule, observed in (
                    ("d3_d3p8_d8", "celltype_classifier_pca50_d3_d3p8_d8.pt"),
                    ("d3_d8", "celltype_classifier_pca50.pt"),
                ):
                    paths = classifier_paths("maizels", None, maizels_schedule=schedule)
                    self.assertEqual(paths["observed_days"], root / observed)
                    self.assertEqual(
                        paths["all_days"], root / "celltype_classifier_pca50.pt"
                    )
                custom = classifier_paths(
                    "maizels", None, classifier_path=str(root / "custom.npz")
                )
                self.assertEqual(custom["observed_days"], root / "custom.npz")

    def test_runner_restores_val_loss_checkpoint_and_logs_maizels_artifacts(self):
        for schedule in ("d3_d3p8_d8", "d3_d8"):
            with self.subTest(
                schedule=schedule
            ), tempfile.TemporaryDirectory() as temporary:
                args = parse_args(
                    [
                        "--dataset_name",
                        "maizels",
                        "--maizels_schedule",
                        schedule,
                        "--smoke_test",
                        "--wandb_mode",
                        "disabled",
                        "--working_dir",
                        temporary,
                        "--batch_size",
                        "4",
                        "--eval_batch_size",
                        "4",
                    ]
                )
                config = experiment_config(args)
                config.update(hidden_dim=8, num_hidden_layers=1, dropout=0.0, epochs=2)

                def factory(*factory_args):
                    model = create_model(*factory_args)
                    training_forward = model.forward_with_loss
                    validation_losses = iter((3.0, 1.0, 2.0))

                    def forward(xs, *loss_args, **loss_kwargs):
                        self.assertEqual(
                            loss_kwargs.get("do_otp"), schedule == "d3_d3p8_d8"
                        )
                        if not model.training:
                            return torch.tensor(next(validation_losses))
                        return training_forward(xs, *loss_args, **loss_kwargs)

                    model.forward_with_loss = forward
                    return model

                paths = {
                    key: Path(temporary) / f"{key}.npz"
                    for key in ("observed_days", "all_days")
                }
                with patch.object(
                    maizels, "load_pca50_dataset", return_value=fixture_data()
                ), patch(
                    "otpfm_baseline.train.create_model", side_effect=factory
                ), patch(
                    "otpfm_baseline.train.classifier_paths", return_value=paths
                ), patch.object(
                    maizels,
                    "check_paths_with_classifier",
                    side_effect=lambda p, t, **kw: {"valid": np.ones(len(p), bool)},
                ), patch(
                    "otpfm_baseline.train.wandb.require"
                ):
                    metrics = run_experiment(args, config, None, 42)
                output = next(Path(temporary).iterdir())
                self.assertTrue((output / "models" / "best.pt").is_file())
                self.assertEqual(
                    json.loads((output / "final_metrics.json").read_text()), metrics
                )
                self.assertEqual(metrics["final_eval/best_step"], 1)
                self.assertEqual(metrics["final_eval/best_validation_loss"], 1.0)
                losses = pd.read_csv(output / "losses.csv")
                self.assertEqual(int(losses["train_loss"].count()), 2)
                self.assertEqual(losses["val_loss"].tolist(), [3.0, 1.0, 2.0])
                metadata = json.loads((output / "experiment.json").read_text())
                self.assertEqual(metadata["pca_dim"], 50)
                self.assertEqual(metadata["checkpoint_monitor"], "val_loss")
                self.assertEqual(metadata["primary_sampler"], "flowmap")
                self.assertEqual(metadata["do_otp"], schedule == "d3_d3p8_d8")
                if metadata["tks"]:
                    self.assertAlmostEqual(metadata["tks"][0], 0.16)
                for name in ("lineage_trajectories", "intermediate_distributions"):
                    self.assertTrue((output / f"final_{name}.png").is_file())

    def test_maizels_runs_one_schedule_per_seed_without_cite_folds(self):
        with patch("otpfm_baseline.train.run_experiment") as run:
            main(["--dataset_name", "maizels", "--seeds", "42", "43"])
        self.assertEqual(run.call_count, 2)
        self.assertEqual(
            [call.args[2:] for call in run.call_args_list], [(None, 42), (None, 43)]
        )


if __name__ == "__main__":
    unittest.main()
