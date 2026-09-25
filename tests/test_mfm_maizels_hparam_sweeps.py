from pathlib import Path

import pytest

from scripts import mfm_maizels_sweep_utils as sweep
from scripts import sweep_maizels_mfm_hparams as mfm_sweep
from scripts import sweep_maizels_sf2m_hparams as sf2m_sweep


def test_default_sf2m_grid_fixes_sigma_and_lr_and_sweeps_score_weight():
    args = sf2m_sweep.parse_args(["--dataset-path", "unused.csv.gz"])
    grid = sweep.build_grid(sf2m_sweep.parameter_grid(args))

    assert len(grid) == 3
    assert {row["flow_lr"] for row in grid} == {1e-3}
    assert {row["sf2m_sigma"] for row in grid} == {0.2}
    assert {row["sf2m_score_weight"] for row in grid} == {0.3, 1.0, 3.0}
    assert {row["flow_weight_decay"] for row in grid} == {1e-5}


def test_default_mfm_grid_fixes_flow_lr_and_sweeps_rho_and_kappa():
    args = mfm_sweep.parse_args(["--dataset-path", "unused.csv.gz"])
    grid = sweep.build_grid(mfm_sweep.parameter_grid(args))

    assert len(grid) == 9
    assert {row["flow_lr"] for row in grid} == {1e-3}
    assert {row["rho"] for row in grid} == {-1.5, -2.75, -5.0}
    assert {row["kappa"] for row in grid} == {1.0, 1.5, 2.0}
    assert {row["geopath_lr"] for row in grid} == {1e-4}
    assert {row["metric_lr"] for row in grid} == {1e-2}
    assert {row["n_centers"] for row in grid} == {150}


def test_derived_configs_apply_single_seed_and_method_hparams(tmp_path):
    base_path = sweep.resolve_config_path(
        "metric-flow-matching/configs/single_cell/50dims/" "sf2m_maizels_3marginal.yaml"
    )
    base = sweep.load_base_config(base_path, "sf2m")
    config = sweep.derived_config(
        base,
        method="sf2m",
        setting_id="setting-a",
        seed=7,
        hparams={
            "flow_lr": 3e-4,
            "flow_weight_decay": 0.0,
            "sf2m_sigma": 0.1,
            "sf2m_score_weight": 3.0,
            "sf2m_time_eps": 1e-3,
        },
        runtime_overrides={"max_steps": 20, "batch_size": 8},
        schedule="d3_d3p8_d8",
        time_mode="real_time",
        hparam_val_times=("D3.4", "D6"),
        shared_cache_root=tmp_path,
    )

    assert config["seeds"] == [7]
    assert config["t_exclude"] is None
    assert config["flow_lr"] == 3e-4
    assert config["sf2m_sigma"] == 0.1
    assert config["sf2m_score_weight"] == 3.0
    assert config["max_steps"] == 20
    assert config["batch_size"] == 8
    assert config["maizels_hparam_val_times"] == ["D3.4", "D6"]


def test_derived_config_does_not_override_run_specific_cli_paths(tmp_path):
    base = {
        "working_dir": "wrong-output",
        "maizels_dataset_path": "wrong-data.csv.gz",
        "maizels_classifier_path": "wrong-classifier.pt",
        "final_metrics_path": "wrong-metrics.json",
    }
    config = sweep.derived_config(
        base,
        method="mfm",
        setting_id="setting-a",
        seed=0,
        hparams={},
        runtime_overrides={},
        schedule="d3_d3p8_d8",
        time_mode="real_time",
        hparam_val_times=("D3.4", "D6"),
        shared_cache_root=tmp_path,
    )

    assert "working_dir" not in config
    assert "maizels_dataset_path" not in config
    assert "maizels_classifier_path" not in config
    assert "final_metrics_path" not in config


def test_method_validation_rejects_the_wrong_base_config():
    sf2m_path = sweep.resolve_config_path(
        "metric-flow-matching/configs/single_cell/50dims/" "sf2m_maizels_3marginal.yaml"
    )
    mfm_path = sweep.resolve_config_path(
        "metric-flow-matching/configs/single_cell/50dims/"
        "i-mfm_maizels_3marginal.yaml"
    )

    with pytest.raises(ValueError, match="MFM sweep"):
        sweep.load_base_config(sf2m_path, "mfm")
    with pytest.raises(ValueError, match="SF2M sweep"):
        sweep.load_base_config(mfm_path, "sf2m")


def test_dry_run_builds_single_seed_commands_with_metrics_export(capsys, tmp_path):
    common = [
        "--dataset-path",
        str(tmp_path / "dataset.csv.gz"),
        "--seeds",
        "4",
        "--output-dir",
        str(tmp_path / "outputs"),
        "--n-pairs",
        "64",
        "--dry-run",
    ]
    assert (
        sf2m_sweep.main(
            common
            + [
                "--flow-learning-rates",
                "0.001",
                "--flow-weight-decays",
                "0.00001",
                "--sf2m-sigmas",
                "0.2",
                "--sf2m-score-weights",
                "1",
                "--sf2m-time-epsilons",
                "0.001",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "SF2M: 1 settings x 1 seeds = 1 runs" in output
    assert "--final_metrics_path" in output
    assert "--config_path" in output
    assert '"maizels_n_pairs": 64' in output

    assert (
        mfm_sweep.main(
            common
            + [
                "--flow-learning-rates",
                "0.001",
                "--flow-weight-decays",
                "0.00001",
                "--geopath-learning-rates",
                "0.0001",
                "--geopath-weight-decays",
                "0.00001",
                "--metric-learning-rates",
                "0.01",
                "--rhos=-2.75",
                "--kappas",
                "1.5",
                "--alpha-metrics",
                "1",
                "--n-centers",
                "150",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "MFM: 1 settings x 1 seeds = 1 runs" in output
    assert "--final_metrics_path" in output


def test_summary_ranks_complete_settings_by_validation_day_emd():
    objective = "final_eval/euler_mean_emd_hparam_val_times"
    settings = [
        {"setting_id": "a", "method": "mfm"},
        {"setting_id": "b", "method": "mfm"},
    ]
    rows = []
    for setting_id, values in (("a", (2.0, 4.0)), ("b", (1.0, 3.0))):
        for seed, value in enumerate(values):
            rows.append(
                {
                    "setting_id": setting_id,
                    "status": "complete",
                    "seed": seed,
                    objective: value,
                }
            )

    summaries = sweep.summarize_settings(rows, settings, (0, 1), objective)
    by_id = {row["setting_id"]: row for row in summaries}
    assert by_id["a"]["objective_mean"] == 3.0
    assert by_id["a"]["objective_rank"] == 2
    assert by_id["b"]["objective_mean"] == 2.0
    assert by_id["b"]["objective_rank"] == 1
