"""Calderon Maizels-ID-5 flow map with endpoint/interpolant-filtered OT."""

from pathlib import Path

from . import calderon_pca100, maizels_pca50

ROOT = Path(__file__).resolve().parents[2]


def get_config(dataset_location="", output_folder="", classifier_path="", classifier_python="", seed=0,
               evaluation_classifier_path=""):
    cfg = calderon_pca100.get_config(dataset_location, output_folder, classifier_path, classifier_python, seed,
                                   evaluation_classifier_path=evaluation_classifier_path)
    reference = maizels_pca50.get_config(5, hparam_val_times=(), seed=seed, maizels_ot_coupling="minibatch_ot")
    cfg.problem.maizels_pair_mode = "ot_endpoint_interpolant"
    cfg.problem.n_interpolant_check_times = reference.problem.n_interpolant_check_times
    cfg.problem.classifier_prob_threshold = reference.problem.classifier_prob_threshold
    cfg.problem.classifier_margin_threshold = reference.problem.classifier_margin_threshold
    cfg.problem.classifier_batch_size = reference.problem.classifier_batch_size
    cfg.problem.ot_minibatch_infeasible_fallback = reference.problem.ot_minibatch_infeasible_fallback
    # Same fixed unfiltered adjacent-interval validation as the baseline.
    # Intermediate marginals and test trajectories do not select a model.
    cfg.evaluation.validation_pair_mode = "ot_plain"
    cfg.logging.output_folder = str(Path(output_folder).expanduser().resolve()) if output_folder else str(
        ROOT / "outputs" / f"calderon_symmetric_4_8_12_filtered_ot_flow_map_seed{seed}")
    cfg.logging.wandb_name = f"calderon_symmetric_4_8_12_pca100_filtered_ot_flow_map_seed{seed}"
    return cfg
