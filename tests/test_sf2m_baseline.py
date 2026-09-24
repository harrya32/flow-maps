from argparse import Namespace
from pathlib import Path
import sys

import pytest
import torch


pytest.importorskip("pytorch_lightning")

REPO_ROOT = Path(__file__).resolve().parents[1]
MFM_ROOT = REPO_ROOT / "metric-flow-matching"
if str(MFM_ROOT) not in sys.path:
    sys.path.insert(0, str(MFM_ROOT))

from mfm.flow_matchers.flow_net_train import (  # noqa: E402
    SF2MFlowNetTrainTrajectory,
)
from mfm.flow_matchers import flow_net_train  # noqa: E402
from mfm.flow_matchers.maizels_eval import (  # noqa: E402
    evaluation_rollout,
    evaluation_sampler_name,
)
from mfm.flow_matchers.models.sf2m import (  # noqa: E402
    IntervalSchrodingerBridgeFlowMatcher,
)


class _ZeroField(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(dim))

    def forward(self, time, state):
        del time
        return torch.zeros_like(state) + self.bias


def _trainer_args(**updates):
    values = {
        "flow_optimizer": "adamw",
        "flow_lr": 1e-3,
        "flow_weight_decay": 0.0,
        "whiten": False,
        "working_dir": ".",
        "data_type": "scrna",
        "seed_current": 7,
        "sf2m_sigma": 0.2,
        "sf2m_score_weight": 1.0,
    }
    values.update(updates)
    return Namespace(**values)


def test_interval_matcher_uses_torchcfm_and_global_time_scaling(monkeypatch):
    matcher = IntervalSchrodingerBridgeFlowMatcher(
        sigma=0.2,
        ot_method="exact",
        time_eps=0.0,
    )

    class FakeNativeMatcher:
        def sample_location_and_conditional_flow(
            self, x0, x1, t=None, return_noise=False
        ):
            assert return_noise
            return t, (x0 + x1) / 2.0, torch.full_like(x0, 2.0), torch.ones_like(x0)

        @staticmethod
        def compute_sigma_t(t):
            return torch.full_like(t, 0.25)

    monkeypatch.setattr(
        matcher, "_native_matcher", lambda duration: FakeNativeMatcher()
    )
    monkeypatch.setattr(
        torch, "rand", lambda *args, **kwargs: torch.full(args, 0.5, **kwargs)
    )
    x0 = torch.zeros(4, 3)
    x1 = torch.ones(4, 3)

    time, state, velocity, score_scale, noise = matcher.sample_location_flow_and_score(
        x0, x1, 0.2, 0.6
    )
    assert torch.allclose(time, torch.full((4,), 0.4))
    assert torch.allclose(state, torch.full_like(state, 0.5))
    assert torch.allclose(velocity, torch.full_like(velocity, 5.0))
    assert score_scale.shape == (4, 1)
    assert torch.all(noise == 1.0)


def test_native_matcher_scales_sigma_with_interval_duration():
    from torchcfm.conditional_flow_matching import (
        SchrodingerBridgeConditionalFlowMatcher,
    )

    matcher = IntervalSchrodingerBridgeFlowMatcher(sigma=0.2)
    native = matcher._native_matcher(0.25)
    assert isinstance(native, SchrodingerBridgeConditionalFlowMatcher)
    assert native.sigma == pytest.approx(0.1)
    assert native.ot_sampler.reg == pytest.approx(0.02)


def test_sf2m_trains_velocity_and_score_fields():
    class FakeMatcher:
        alpha = 0

        @staticmethod
        def sample_location_flow_and_score(x0, x1, t_min, t_max):
            batch = x0.shape[0]
            time = torch.full(
                (batch,), (float(t_min) + float(t_max)) / 2.0, dtype=x0.dtype
            )
            state = (x0 + x1) / 2.0
            velocity = x1 - x0
            scale = torch.ones(batch, 1, dtype=x0.dtype)
            noise = torch.ones_like(x0)
            return time, state, velocity, scale, noise

    model = SF2MFlowNetTrainTrajectory(
        flow_matcher=FakeMatcher(),
        flow_net=_ZeroField(2),
        score_net=_ZeroField(2),
        skipped_time_points=[1],
        args=_trainer_args(),
    )
    model.timesteps = [0.0, 0.25, 1.0]
    total, velocity, score = model._compute_sf2m_loss(
        [torch.zeros(5, 2), torch.zeros(5, 2), torch.ones(5, 2)]
    )
    assert float(velocity) == pytest.approx(1.0)
    assert float(score) == pytest.approx(1.0)
    assert float(total) == pytest.approx(2.0)
    total.backward()
    assert model.flow_net.bias.grad is not None
    assert model.score_net.bias.grad is not None


def test_sf2m_euler_maruyama_rollout_is_seeded():
    model = SF2MFlowNetTrainTrajectory(
        flow_matcher=Namespace(alpha=0),
        flow_net=_ZeroField(2),
        score_net=_ZeroField(2),
        skipped_time_points=[],
        args=_trainer_args(),
    )
    initial = torch.zeros(6, 2)
    first = model.evaluation_trajectory(
        initial, start_time=0.0, end_time=1.0, n_steps=10, seed=19
    )
    second = model.evaluation_trajectory(
        initial, start_time=0.0, end_time=1.0, n_steps=10, seed=19
    )
    different = model.evaluation_trajectory(
        initial, start_time=0.0, end_time=1.0, n_steps=10, seed=20
    )
    assert first.shape == (11, 6, 2)
    assert torch.equal(first, second)
    assert not torch.equal(first, different)

    final, batch_major = evaluation_rollout(
        model, initial, 1.0, 10, start_time=0.0, seed=19
    )
    assert evaluation_sampler_name(model) == "euler_maruyama"
    assert batch_major.shape == (6, 11, 2)
    assert torch.equal(final, batch_major[:, -1])


def test_sf2m_test_step_logs_shared_and_sampler_specific_metrics(monkeypatch):
    monkeypatch.setattr(
        flow_net_train, "wasserstein_distance", lambda *args, **kwargs: 1.25
    )
    monkeypatch.setattr(flow_net_train, "rbf_mmd2", lambda *args, **kwargs: 0.125)
    model = SF2MFlowNetTrainTrajectory(
        flow_matcher=Namespace(alpha=0),
        flow_net=_ZeroField(2),
        score_net=_ZeroField(2),
        skipped_time_points=[1],
        args=_trainer_args(),
    )
    model._trainer = Namespace(
        datamodule=Namespace(
            data_type="scrna",
            data_name="cite",
            times=torch.tensor([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]),
            timepoint_splits={key: {} for key in ("2", "3", "4", "7")},
        )
    )
    logged = {}
    monkeypatch.setattr(
        model,
        "log",
        lambda name, value, **kwargs: logged.update({name: float(value)}),
    )

    model.test_step([torch.zeros(3, 2) for _ in range(4)], 0)

    assert logged["test_EMD"] == 1.25
    assert logged["final_eval/euler_mean_emd"] == 1.25
    assert logged["final_eval/euler_mean_rbf_mmd2"] == 0.125
    assert logged["final_eval/euler_maruyama_mean_emd"] == 1.25
    assert logged["final_eval/euler_maruyama_mean_rbf_mmd2"] == 0.125


@pytest.mark.parametrize(
    "relative_path",
    [
        "configs/single_cell/100dims/sf2m_cite.yaml",
        "configs/single_cell/100dims/sf2m_multi.yaml",
        "configs/single_cell/50dims/sf2m_maizels.yaml",
        "configs/single_cell/50dims/sf2m_maizels_3marginal.yaml",
    ],
)
def test_sf2m_configs_use_internal_ot_once(relative_path):
    import yaml

    config = yaml.safe_load((MFM_ROOT / relative_path).read_text())
    assert config["sf2m"] is True
    assert config["mfm"] is False
    assert str(config["optimal_transport_method"]).lower() == "none"
    assert config["max_steps"] == 10_000
    assert config["val_check_interval"] == 100
