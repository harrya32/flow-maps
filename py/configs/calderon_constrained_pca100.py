"""Calderon Maizels-ID-6 constrained flow map with filtered OT pairs."""

from pathlib import Path

from ml_collections import ConfigDict

from . import calderon_filtered_pca100, maizels_pca50

ROOT = Path(__file__).resolve().parents[2]


def get_config(dataset_location="", output_folder="", classifier_path="", classifier_python="", seed=0,
               evaluation_classifier_path=""):
    cfg = calderon_filtered_pca100.get_config(dataset_location, output_folder, classifier_path, classifier_python, seed,
                                            evaluation_classifier_path=evaluation_classifier_path)
    reference = maizels_pca50.get_config(6, hparam_val_times=(), seed=seed, maizels_ot_coupling="minibatch_ot")
    cfg.constraints = ConfigDict(reference.constraints.to_dict())
    # Endpoint IDs enter only the loss; the flow network remains unconditional.
    cfg.problem.interp_uses_labels = True
    # Full-grid inference is faster on this GPU than early rejection; preserve
    # the exact filtering execution used by the existing filtered baseline.
    cfg.problem.interpolant_check_chunk_size = 50
    cfg.evaluation.validation_objective = "plain_ot_lsd_without_constraint"
    cfg.logging.output_folder = str(Path(output_folder).expanduser().resolve()) if output_folder else str(
        ROOT / "outputs" / f"calderon_symmetric_4_8_12_constrained_filtered_ot_flow_map_seed{seed}")
    cfg.logging.wandb_name = f"calderon_symmetric_4_8_12_pca100_constrained_filtered_ot_flow_map_seed{seed}"
    return cfg
