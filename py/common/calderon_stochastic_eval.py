"""Native Maizels SSFM evaluation on Calderon windows and tissue ensembles."""

from . import calderon_stochastic_data as data_backend
from . import maizels_stochastic_eval as shared

save_full_data_trajectory_plot = shared.save_full_data_trajectory_plot


def distribution_metrics(model, params, cfg, **kwargs):
    return shared.distribution_metrics(model, params, cfg, data_backend=data_backend,
                                       include_direct=False, include_mmd=False, **kwargs)


def observed_distribution_metrics(model, params, cfg, **kwargs):
    return shared.observed_distribution_metrics(model, params, cfg, data_backend=data_backend, **kwargs)


def lineage_metrics(model, params, cfg, **kwargs):
    return shared.lineage_metrics(model, params, cfg, data_backend=data_backend, **kwargs)


def pushforward_plot_data(model, params, cfg, **kwargs):
    return shared.pushforward_plot_data(model, params, cfg, data_backend=data_backend,
                                        include_direct=False, **kwargs)


def save_pushforward_plot(plot_data, output_path, **kwargs):
    return shared.save_pushforward_plot(plot_data, output_path, composed_only=True, **kwargs)


def full_data_trajectory_plot_data(model, params, cfg, **kwargs):
    return shared.full_data_trajectory_plot_data(model, params, cfg, data_backend=data_backend, **kwargs)


def final_evaluation(model, params, cfg, output_dir):
    return shared.final_evaluation(model, params, cfg, output_dir,
                                   data_backend=data_backend, trajectory_tag="heldout_4_6h",
                                   include_direct=False, include_mmd=False)
