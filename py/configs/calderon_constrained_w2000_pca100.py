"""Calderon constrained, interpolant-filtered OT flow map at weight 2000."""

from pathlib import Path

from . import calderon_constrained_pca100

ROOT = Path(__file__).resolve().parents[2]


def get_config(dataset_location="", output_folder="", classifier_path="", classifier_python="", seed=0,
               evaluation_classifier_path=""):
    cfg = calderon_constrained_pca100.get_config(
        dataset_location, output_folder, classifier_path, classifier_python, seed,
        evaluation_classifier_path=evaluation_classifier_path)
    cfg.constraints.weight = 2000.0
    if not output_folder:
        cfg.logging.output_folder = str(
            ROOT / "outputs" / f"calderon_symmetric_4_8_12_constrained_filtered_ot_flow_map_w2000_seed{seed}")
    cfg.logging.wandb_name = f"calderon_symmetric_4_8_12_pca100_constrained_filtered_ot_flow_map_w2000_seed{seed}"
    return cfg
