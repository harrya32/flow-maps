"""Training loops for SF2M, SSFM, published CLIFT, and Maizels-style CLIFT."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from .classifier import load_classifier, sha256
from .diffusion import instantaneous_variance_torch, resolve_prior
from .models import (
    DirectStrongMap,
    Fields,
    bridge_losses,
    build_pair_banks,
    coarsen,
    composition_batch,
    consistency_loss,
    diagonal_bridge_statistics,
    draw_atoms,
    endpoint_lineage_nll,
    load_fields,
    load_map,
    load_pair_banks,
    map_fate_batch,
    map_fate_loss,
    save_fields,
    save_map,
    update_ema,
)


METHODS = ("sf2m", "ssfm", "clift", "real_clift")


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def select_device(requested="auto"):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _model_statistics(data):
    points = data["train_points"].reshape(-1, 2).astype(np.float64)
    mean = points.mean(0)
    scale = points.std(0)
    if np.any(scale < 1e-8):
        raise ValueError("Training coordinate scale is degenerate.")
    return mean, scale


def _clip(model, maximum):
    torch.nn.utils.clip_grad_norm_(model.parameters(), float(maximum))


def _clip_map(model, maximum):
    # Preserve the upstream optimizer convention: the local fields and finite
    # response are clipped independently rather than as one combined vector.
    torch.nn.utils.clip_grad_norm_(_field_parameters(model), float(maximum))
    torch.nn.utils.clip_grad_norm_(model.finite.parameters(), float(maximum))


def real_clift_horizon_max(settings, step):
    """Largest normalized off-diagonal horizon at a given optimizer step."""
    local = float(settings["local_step_fraction"])
    maximum = float(settings["max_horizon_fraction"])
    curriculum_steps = max(
        1.0,
        float(settings["horizon_curriculum_fraction"]) * float(settings["steps"]),
    )
    progress = min(max(float(step) / curriculum_steps, 0.0), 1.0)
    progress = progress ** float(settings["horizon_curriculum_power"])
    return local * (maximum / local) ** progress


def _sample_real_clift_bridge_states(
    model,
    bank,
    count,
    lower_horizon,
    upper_horizon,
    log_uniform,
    time_epsilon,
    rng,
):
    """Sample stochastic-interpolant states and their exact conditional drifts."""
    chunks = []
    intervals = len(bank.times) - 1
    counts = [count // intervals] * intervals
    for index in range(count % intervals):
        counts[index] += 1
    for interval, n in enumerate(counts):
        if not n:
            continue
        left, right = [float(value) for value in bank.times[interval : interval + 2]]
        duration = right - left
        source, target = bank.pair(interval, n, rng, model.mean.device)
        if log_uniform:
            uniform = rng.random((n, 1))
            horizon_fraction = np.exp(
                math.log(float(lower_horizon))
                + uniform * (math.log(float(upper_horizon)) - math.log(float(lower_horizon)))
            )
        else:
            horizon_fraction = rng.uniform(float(lower_horizon), float(upper_horizon), (n, 1))
        available = np.maximum(1.0 - 2.0 * float(time_epsilon) - horizon_fraction, 1e-8)
        tau = float(time_epsilon) + rng.random((n, 1)) * available
        start = torch.as_tensor(
            left + tau * duration,
            dtype=torch.float32,
            device=model.mean.device,
        )
        end = torch.as_tensor(
            left + (tau + horizon_fraction) * duration,
            dtype=torch.float32,
            device=model.mean.device,
        )
        noise = torch.as_tensor(
            rng.normal(size=(n, 2)),
            dtype=torch.float32,
            device=model.mean.device,
        )
        statistics = diagonal_bridge_statistics(
            model.normalize(source),
            model.normalize(target),
            start,
            left,
            right,
            model.prior,
            model.scale,
        )
        standard_deviation = statistics.variance.sqrt().clamp_min(1e-8)
        state = statistics.mean + standard_deviation * noise
        score_target = -noise / standard_deviation
        instantaneous_variance = (
            instantaneous_variance_torch(start, model.prior) / model.scale**2
        )
        drift_target = (
            statistics.mean_derivative
            + 0.5 * statistics.variance_derivative / standard_deviation * noise
            + 0.5 * instantaneous_variance * score_target
        )
        interval_ids = torch.full(
            (n,), interval, dtype=torch.long, device=model.mean.device
        )
        chunks.append(
            (state, start, end, drift_target, interval_ids)
        )

    combined = tuple(
        torch.cat([chunk[field] for chunk in chunks], dim=0)
        for field in range(5)
    )
    # PairBank returns interval-contiguous chunks. Randomize them before the
    # off-diagonal prefix is reused for the lineage term, otherwise that loss
    # can silently see only the first (often unconstrained root) interval.
    order = torch.as_tensor(
        rng.permutation(count), dtype=torch.long, device=model.mean.device
    )
    return tuple(value[order] for value in combined)


def _endpoint_mse(prediction, target, start, end, horizon_power=1.0):
    horizon = (end - start).squeeze(-1).clamp_min(1e-6)
    denominator = horizon ** float(horizon_power)
    return ((prediction - target.detach()).square().mean(dim=-1) / denominator).mean()


def _update_full_ema(ema, model, decay):
    """Update every teacher parameter; unlike the paper port, copy none directly."""
    with torch.no_grad():
        targets = dict(ema.named_parameters())
        for name, source in model.named_parameters():
            targets[name].mul_(float(decay)).add_(source, alpha=1.0 - float(decay))


def _real_clift_objective(model, teacher, bank, classifier, settings, step, rng, generator):
    """Maizels-style local/semigroup/lineage objective on fixed OT pairs."""
    batch_size = int(settings["batch_size"])
    local_fraction = float(settings["local_fraction"])
    n_local = max(1, min(batch_size - 1, int(batch_size * local_fraction)))
    n_off_diagonal = batch_size - n_local
    local_step = float(settings["local_step_fraction"])
    epsilon = float(settings["time_eps_fraction"])

    local_state, local_start, local_end, local_drift, _ = _sample_real_clift_bridge_states(
        model,
        bank,
        n_local,
        local_step / 2.0,
        local_step,
        False,
        epsilon,
        rng,
    )
    local_atoms, local_variances = draw_atoms(
        model, local_start, local_end, generator
    )
    local_target = (
        local_state
        + (local_end - local_start) * local_drift
        + local_atoms.sum(dim=1)
    )
    local_prediction = model(
        local_state,
        local_start,
        local_end,
        local_atoms,
        local_variances,
    )
    local_loss = _endpoint_mse(
        local_prediction,
        local_target,
        local_start,
        local_end,
        horizon_power=float(settings["local_loss_horizon_power"]),
    )

    horizon_max = real_clift_horizon_max(settings, step)
    off_state, off_start, off_end, _, off_interval = _sample_real_clift_bridge_states(
        model,
        bank,
        n_off_diagonal,
        local_step,
        horizon_max,
        bool(settings["log_uniform_horizon"]),
        epsilon,
        rng,
    )
    fine_atoms, fine_variances = draw_atoms(
        model,
        off_start,
        off_end,
        generator,
        bins=2 * model.bins,
    )
    direct_atoms, direct_variances = coarsen(
        fine_atoms, fine_variances, model.bins
    )
    prediction = model(
        off_state,
        off_start,
        off_end,
        direct_atoms,
        direct_variances,
    )
    midpoint = 0.5 * (off_start + off_end)
    with torch.no_grad():
        left = teacher(
            off_state,
            off_start,
            midpoint,
            fine_atoms[:, : model.bins],
            fine_variances[:, : model.bins],
        )
        two_step_target = teacher(
            left,
            midpoint,
            off_end,
            fine_atoms[:, model.bins :],
            fine_variances[:, model.bins :],
        )
    semigroup_loss = _endpoint_mse(
        prediction,
        two_step_target,
        off_start,
        off_end,
        horizon_power=float(settings["semigroup_loss_horizon_power"]),
    )

    constraint_size = min(
        int(settings["constraint_batch_size"]), n_off_diagonal
    )
    lineage_loss = endpoint_lineage_nll(
        model,
        classifier,
        off_state[:constraint_size],
        prediction[:constraint_size],
    )
    ssfm_loss = (
        local_fraction * local_loss
        + (1.0 - local_fraction) * semigroup_loss
    )
    total = ssfm_loss + float(settings["constraint_weight"]) * lineage_loss
    metrics = {
        "loss": total.detach(),
        "local": local_loss.detach(),
        "semigroup": semigroup_loss.detach(),
        "lineage": lineage_loss.detach(),
        "lineage_weighted": (
            float(settings["constraint_weight"]) * lineage_loss
        ).detach(),
        "constraint_final_interval_fraction": (
            off_interval[:constraint_size] == (len(bank.times) - 2)
        ).float().mean(),
        "horizon_max_fraction": torch.as_tensor(horizon_max),
    }
    return total, metrics


def _real_clift_learning_rate(settings, step):
    maximum = float(settings["learning_rate"])
    minimum = float(settings["minimum_learning_rate"])
    total = int(settings["steps"])
    warmup = int(settings["warmup_steps"])
    if warmup and step <= warmup:
        return maximum * float(step) / float(warmup)
    progress = min(max((step - warmup) / max(1, total - warmup), 0.0), 1.0)
    return minimum + 0.5 * (maximum - minimum) * (1.0 + math.cos(math.pi * progress))


def train_sf2m_base(root, seed, prior_name, config, device):
    output = root / "models" / f"seed_{seed}" / prior_name / "_sf2m_base"
    output.mkdir(parents=True, exist_ok=True)
    settings = config["sf2m"]
    prior = resolve_prior(config["priors"][prior_name], config["dataset"])
    banks, data = load_pair_banks(
        root / "inputs" / "learning_data.npz", root / "couplings" / prior_name / "couplings.npz"
    ), np.load(root / "inputs" / "learning_data.npz", allow_pickle=False)
    mean, scale = _model_statistics(data)
    torch.manual_seed(int(seed))
    model = Fields(
        mean,
        scale,
        prior,
        config["network"],
        config["dataset"]["end_time"],
    ).to(device)
    rng = np.random.default_rng(int(seed) + 1501)
    validation = banks["validation"].bridge(
        model, int(settings["validation_batch_size"]), np.random.default_rng(int(seed) + 1502)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, int(settings["base_steps"]), eta_min=3e-5
    )
    history, best, selected, state = [], float("inf"), 0, None
    started = time.perf_counter()
    for step in range(1, int(settings["base_steps"]) + 1):
        optimizer.zero_grad(set_to_none=True)
        velocity, score = bridge_losses(model, banks["train"].bridge(model, int(settings["batch_size"]), rng))
        loss = velocity + score
        loss.backward()
        _clip(model, settings["gradient_clip"])
        optimizer.step()
        scheduler.step()
        if step == 1 or step % int(settings["eval_every"]) == 0 or step == int(settings["base_steps"]):
            with torch.no_grad():
                val_velocity, val_score = [float(value) for value in bridge_losses(model, validation)]
            value = val_velocity + val_score
            if value < best:
                best, selected, state = value, step, copy.deepcopy(model.state_dict())
            history.append({"step": step, "velocity": val_velocity, "score": val_score, "selected_step": selected})
            print(f"seed={seed} {prior_name} SF2M base {step}: validation={value:.6g}", flush=True)
    if state is None:
        raise RuntimeError("SF2M base training selected no checkpoint.")
    model.load_state_dict(state)
    metadata = {"seed": seed, "prior": prior_name, "selected_step": selected, "phase": "base"}
    save_fields(model, output / "model.pt", metadata)
    _write(output / "history.json", history)
    _write(
        output / "manifest.json",
        {
            "status": "completed",
            "termination": "fixed_budget",
            "updates": int(settings["base_steps"]),
            "selected_step": selected,
            "training_seconds": time.perf_counter() - started,
            "checkpoint_sha256": sha256(output / "model.pt"),
            "settings": settings,
        },
    )
    return output / "model.pt"


def continue_sf2m(root, seed, prior_name, config, device):
    base_path = root / "models" / f"seed_{seed}" / prior_name / "_sf2m_base" / "model.pt"
    output = root / "models" / f"seed_{seed}" / prior_name / "sf2m"
    output.mkdir(parents=True, exist_ok=True)
    settings = config["sf2m"]
    model = load_fields(base_path, device)
    banks = load_pair_banks(
        root / "inputs" / "learning_data.npz", root / "couplings" / prior_name / "couplings.npz"
    )
    validation = banks["validation"].bridge(
        model, int(settings["validation_batch_size"]), np.random.default_rng(int(seed) + 1502)
    )
    with torch.no_grad():
        initial = torch.as_tensor([float(value) for value in bridge_losses(model, validation)], device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["continuation_learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    rng = np.random.default_rng(int(seed) + 4400)
    best, selected, state = 2.0, 0, copy.deepcopy(model.state_dict())
    history = [{"step": 0, "base_ratio_sum": 2.0, "selected_step": 0}]
    started = time.perf_counter()
    for step in range(1, int(settings["continuation_steps"]) + 1):
        optimizer.zero_grad(set_to_none=True)
        velocity, score = bridge_losses(model, banks["train"].bridge(model, int(settings["batch_size"]), rng))
        loss = velocity / initial[0] + score / initial[1]
        loss.backward()
        _clip(model, settings["gradient_clip"])
        optimizer.step()
        if step % int(settings["continuation_eval_every"]) == 0 or step == int(settings["continuation_steps"]):
            with torch.no_grad():
                values = torch.as_tensor([float(value) for value in bridge_losses(model, validation)])
            ratio = float((values / initial.cpu()).sum())
            eligible = ratio <= 2.0 * (1.0 + float(settings["maximum_base_increase"]))
            if eligible and ratio < best:
                best, selected, state = ratio, step, copy.deepcopy(model.state_dict())
            history.append({"step": step, "base_ratio_sum": ratio, "eligible": eligible, "selected_step": selected})
            print(f"seed={seed} {prior_name} SF2M control {step}: ratio={ratio:.6g}", flush=True)
    model.load_state_dict(state)
    metadata = {"seed": seed, "prior": prior_name, "selected_step": selected, "phase": "zero_fate_control"}
    save_fields(model, output / "model.pt", metadata)
    _write(output / "history.json", history)
    _write(
        output / "manifest.json",
        {
            "status": "completed",
            "termination": "fixed_budget",
            "updates": int(settings["continuation_steps"]),
            "selected_step": selected,
            "training_seconds": time.perf_counter() - started,
            "checkpoint_sha256": sha256(output / "model.pt"),
            "base_checkpoint_sha256": sha256(base_path),
            "settings": settings,
        },
    )


def _field_parameters(model):
    return list(model.velocity.parameters()) + list(model.score.parameters())


def _map_validation(model, bridge, composition, classifier=None, fate=None):
    with torch.no_grad():
        velocity, score = bridge_losses(model, bridge)
        result = {
            "velocity": float(velocity),
            "score": float(score),
            "consistency": float(consistency_loss(model, model, composition)),
        }
        if fate is not None:
            result["fate_nll"] = float(map_fate_loss(model, classifier, fate))
    return result


def train_map_base(root, seed, prior_name, config, device):
    output = root / "models" / f"seed_{seed}" / prior_name / "_map_base"
    output.mkdir(parents=True, exist_ok=True)
    settings = config["strong_map"]
    prior = resolve_prior(config["priors"][prior_name], config["dataset"])
    banks = load_pair_banks(
        root / "inputs" / "learning_data.npz", root / "couplings" / prior_name / "couplings.npz"
    )
    data = np.load(root / "inputs" / "learning_data.npz", allow_pickle=False)
    mean, scale = _model_statistics(data)
    torch.manual_seed(int(seed))
    model = DirectStrongMap(
        mean,
        scale,
        prior,
        config["network"],
        config["dataset"]["end_time"],
        settings["noise_bins"],
        settings["anchor_step"],
    ).to(device)
    ema = copy.deepcopy(model).requires_grad_(False)
    rng = np.random.default_rng(int(seed) + 1501)
    composition_rng = np.random.default_rng(int(seed) + 2001)
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + 2002)
    validation = banks["validation"].bridge(
        model, int(settings["validation_bridge_batch_size"]), np.random.default_rng(int(seed) + 1502)
    )
    validation_composition = composition_batch(
        model,
        banks["validation"],
        int(settings["validation_composition_batch_size"]),
        settings["horizons"],
        np.random.default_rng(int(seed) + 2011),
        torch.Generator(device="cpu").manual_seed(int(seed) + 2012),
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": _field_parameters(model), "lr": float(settings["field_learning_rate"])},
            {"params": model.finite.parameters(), "lr": float(settings["finite_learning_rate"])},
        ],
        weight_decay=float(settings["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, int(settings["base_steps"]), eta_min=3e-5
    )
    best, selected, state, history = float("inf"), 0, None, []
    started = time.perf_counter()
    for step in range(1, int(settings["base_steps"]) + 1):
        optimizer.zero_grad(set_to_none=True)
        velocity, score = bridge_losses(model, banks["train"].bridge(model, int(settings["batch_size"]), rng))
        batch = composition_batch(
            model,
            banks["train"],
            int(settings["composition_batch_size"]),
            settings["horizons"],
            composition_rng,
            generator,
        )
        consistency = consistency_loss(model, ema, batch)
        loss = velocity + score + float(settings["consistency_weight"]) * consistency
        loss.backward()
        _clip_map(model, settings["gradient_clip"])
        optimizer.step()
        scheduler.step()
        update_ema(ema, model, float(settings["ema_decay"]))
        if step == 1 or step % int(settings["eval_every"]) == 0 or step == int(settings["base_steps"]):
            values = _map_validation(model, validation, validation_composition)
            score_value = values["velocity"] + values["score"]
            if score_value < best:
                best, selected, state = score_value, step, copy.deepcopy(model.state_dict())
            history.append({"step": step, **values, "selected_step": selected})
            print(f"seed={seed} {prior_name} map base {step}: validation={score_value:.6g}", flush=True)
    if state is None:
        raise RuntimeError("Strong-map base training selected no checkpoint.")
    model.load_state_dict(state)
    metadata = {"seed": seed, "prior": prior_name, "selected_step": selected, "phase": "base"}
    save_map(model, output / "model.pt", metadata)
    _write(output / "history.json", history)
    _write(
        output / "manifest.json",
        {
            "status": "completed",
            "termination": "fixed_budget",
            "updates": int(settings["base_steps"]),
            "selected_step": selected,
            "training_seconds": time.perf_counter() - started,
            "checkpoint_sha256": sha256(output / "model.pt"),
            "settings": settings,
        },
    )


def continue_map(root, seed, prior_name, method, config, device):
    guided = method == "clift"
    base_path = root / "models" / f"seed_{seed}" / prior_name / "_map_base" / "model.pt"
    output = root / "models" / f"seed_{seed}" / prior_name / method
    output.mkdir(parents=True, exist_ok=True)
    settings = config["strong_map"]
    model = load_map(base_path, device)
    ema = copy.deepcopy(model).requires_grad_(False)
    banks = load_pair_banks(
        root / "inputs" / "learning_data.npz", root / "couplings" / prior_name / "couplings.npz"
    )
    classifier = load_classifier(root / "classifier" / "classifier.pt", device)
    rng = np.random.default_rng(int(seed) + 4400)
    composition_rng = np.random.default_rng(int(seed) + 4401)
    fate_rng = np.random.default_rng(int(seed) + 4500)
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + 4402)
    fate_generator = torch.Generator(device="cpu").manual_seed(int(seed) + 4502)
    validation = banks["validation"].bridge(
        model, int(settings["validation_bridge_batch_size"]), np.random.default_rng(int(seed) + 1502)
    )
    validation_composition = composition_batch(
        model,
        banks["validation"],
        int(settings["validation_composition_batch_size"]),
        settings["horizons"],
        np.random.default_rng(int(seed) + 2011),
        torch.Generator(device="cpu").manual_seed(int(seed) + 2012),
    )
    validation_fate = map_fate_batch(
        model,
        banks["validation"],
        int(settings["validation_fate_batch_size"]),
        int(settings["fate_map_steps"]),
        np.random.default_rng(int(seed) + 4501),
        torch.Generator(device="cpu").manual_seed(int(seed) + 4503),
        settings,
    )
    initial = _map_validation(model, validation, validation_composition, classifier, validation_fate)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["continuation_learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    best = initial["fate_nll"] if guided else 2.0 + float(settings["consistency_weight"]) * initial["consistency"]
    selected, state = 0, copy.deepcopy(model.state_dict())
    history = [{"step": 0, **initial, "selected_step": 0, "eligible": True}]
    started = time.perf_counter()
    for step in range(1, int(settings["continuation_steps"]) + 1):
        optimizer.zero_grad(set_to_none=True)
        velocity, score = bridge_losses(model, banks["train"].bridge(model, int(settings["batch_size"]), rng))
        composition = composition_batch(
            model,
            banks["train"],
            int(settings["composition_batch_size"]),
            settings["horizons"],
            composition_rng,
            generator,
        )
        consistency = consistency_loss(model, ema, composition)
        loss = (
            velocity / initial["velocity"]
            + score / initial["score"]
            + float(settings["consistency_weight"]) * consistency
        )
        if guided:
            fate = map_fate_loss(
                model,
                classifier,
                map_fate_batch(
                    model,
                    banks["train"],
                    int(settings["rollout_batch_size"]),
                    int(settings["fate_map_steps"]),
                    fate_rng,
                    fate_generator,
                    settings,
                ),
            )
            loss = loss + float(settings["fate_weight"]) * fate
        loss.backward()
        _clip_map(model, settings["gradient_clip"])
        optimizer.step()
        update_ema(ema, model, float(settings["ema_decay"]))
        if step % int(settings["continuation_eval_every"]) == 0 or step == int(settings["continuation_steps"]):
            values = _map_validation(model, validation, validation_composition, classifier, validation_fate)
            ratio = values["velocity"] / initial["velocity"] + values["score"] / initial["score"]
            consistency_guard = max(
                initial["consistency"] * (1.0 + float(settings["maximum_consistency_increase"])),
                float(settings["consistency_guard_floor"]),
            )
            eligible = (
                ratio <= 2.0 * (1.0 + float(settings["maximum_base_increase"]))
                and values["consistency"] <= consistency_guard
            )
            value = values["fate_nll"] if guided else ratio + float(settings["consistency_weight"]) * values["consistency"]
            if eligible and value < best:
                best, selected, state = value, step, copy.deepcopy(model.state_dict())
            history.append(
                {"step": step, **values, "base_ratio_sum": ratio, "eligible": eligible, "selected_step": selected}
            )
            print(f"seed={seed} {prior_name} {method} {step}: objective={value:.6g}", flush=True)
    model.load_state_dict(state)
    metadata = {
        "seed": seed,
        "prior": prior_name,
        "selected_step": selected,
        "phase": "guided" if guided else "zero_fate_control",
    }
    save_map(model, output / "model.pt", metadata)
    _write(output / "history.json", history)
    _write(
        output / "manifest.json",
        {
            "status": "completed",
            "termination": "fixed_budget",
            "updates": int(settings["continuation_steps"]),
            "selected_step": selected,
            "training_seconds": time.perf_counter() - started,
            "checkpoint_sha256": sha256(output / "model.pt"),
            "base_checkpoint_sha256": sha256(base_path),
            "guided": guided,
            "settings": settings,
        },
    )


def train_real_clift(root, seed, prior_name, config, device):
    """Train the Maizels-style CLIFT objective end-to-end.

    The EMA is used only as the stop-gradient two-half-step teacher.  The
    checkpoint and all downstream evaluation use the final instantaneous
    student parameters.
    """
    output = root / "models" / f"seed_{seed}" / prior_name / "real_clift"
    output.mkdir(parents=True, exist_ok=True)
    settings = config["real_clift"]
    prior = resolve_prior(config["priors"][prior_name], config["dataset"])
    banks = load_pair_banks(
        root / "inputs" / "learning_data.npz",
        root / "couplings" / prior_name / "couplings.npz",
    )
    data = np.load(root / "inputs" / "learning_data.npz", allow_pickle=False)
    mean, scale = _model_statistics(data)
    torch.manual_seed(int(seed))
    model = DirectStrongMap(
        mean,
        scale,
        prior,
        config["network"],
        config["dataset"]["end_time"],
        settings["noise_bins"],
        settings["anchor_step"],
    ).to(device)
    teacher = copy.deepcopy(model).requires_grad_(False).eval()
    classifier = load_classifier(root / "classifier" / "classifier.pt", device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        betas=(float(settings["adam_beta1"]), float(settings["adam_beta2"])),
        weight_decay=float(settings["weight_decay"]),
    )
    rng = np.random.default_rng(int(seed) + 6101)
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + 6102)
    history = []
    total_steps = int(settings["steps"])
    started = time.perf_counter()
    model.train()
    for step in range(1, total_steps + 1):
        learning_rate = _real_clift_learning_rate(settings, step)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        loss, train_metrics = _real_clift_objective(
            model,
            teacher,
            banks["train"],
            classifier,
            settings,
            step - 1,
            rng,
            generator,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite real_clift loss at optimizer step {step}."
            )
        loss.backward()
        _clip_map(model, settings["gradient_clip"])
        optimizer.step()
        _update_full_ema(teacher, model, float(settings["ema_decay"]))

        if step == 1 or step % int(settings["eval_every"]) == 0 or step == total_steps:
            validation_rng = np.random.default_rng(int(seed) + 6201)
            validation_generator = torch.Generator(device="cpu").manual_seed(
                int(seed) + 6202
            )
            validation_settings = dict(settings)
            validation_settings["batch_size"] = int(settings["validation_batch_size"])
            with torch.no_grad():
                _, validation_metrics = _real_clift_objective(
                    model,
                    teacher,
                    banks["validation"],
                    classifier,
                    validation_settings,
                    step - 1,
                    validation_rng,
                    validation_generator,
                )
            record = {
                "step": step,
                "learning_rate": learning_rate,
                **{
                    f"train_{name}": float(value)
                    for name, value in train_metrics.items()
                },
                **{
                    f"validation_{name}": float(value)
                    for name, value in validation_metrics.items()
                },
            }
            history.append(record)
            print(
                f"seed={seed} {prior_name} real_clift {step}: "
                f"loss={record['train_loss']:.6g}, "
                f"validation={record['validation_loss']:.6g}, "
                f"h_max={record['train_horizon_max_fraction']:.4g}",
                flush=True,
            )

    model.eval()
    metadata = {
        "seed": seed,
        "prior": prior_name,
        "selected_step": total_steps,
        "phase": "end_to_end_real_clift",
        "evaluation_parameters": "instantaneous_student",
    }
    save_map(model, output / "model.pt", metadata)
    _write(output / "history.json", history)
    _write(
        output / "manifest.json",
        {
            "status": "completed",
            "termination": "fixed_budget",
            "updates": total_steps,
            "selected_step": total_steps,
            "training_seconds": time.perf_counter() - started,
            "checkpoint_sha256": sha256(output / "model.pt"),
            "teacher": "full_parameter_ema_stop_gradient",
            "teacher_ema_decay": float(settings["ema_decay"]),
            "lineage_source": "detached_classifier_distribution_at_map_input",
            "evaluation_parameters": "instantaneous_student",
            "fixed_precomputed_ot": True,
            "settings": settings,
        },
    )


def _complete(path):
    manifest = path / "manifest.json"
    checkpoint = path / "model.pt"
    if not manifest.exists() or not checkpoint.exists():
        return False
    values = json.loads(manifest.read_text())
    return values.get("status") == "completed" and values.get("checkpoint_sha256") == sha256(checkpoint)


def prepare_couplings(root, config):
    for prior_name, prior_settings in config["priors"].items():
        output = root / "couplings" / prior_name
        if (output / "couplings.npz").exists():
            continue
        output.mkdir(parents=True, exist_ok=True)
        prior = resolve_prior(prior_settings, config["dataset"])
        build_pair_banks(root / "inputs" / "learning_data.npz", prior, output)


def train_all(root, config, methods=METHODS, seeds=None, priors=None, device="auto"):
    root = Path(root)
    methods = tuple(methods)
    if not methods:
        raise ValueError("At least one method must be selected.")
    unknown = set(methods) - set(METHODS)
    if unknown:
        raise ValueError(f"Unknown methods: {sorted(unknown)}")
    seeds = [int(value) for value in (config["seeds"] if seeds is None else seeds)]
    unknown_seeds = set(seeds) - set(config["seeds"])
    if unknown_seeds:
        raise ValueError(f"Training seeds are not registered in the config: {sorted(unknown_seeds)}")
    priors = list(config["priors"] if priors is None else priors)
    if not priors:
        raise ValueError("At least one prior must be selected.")
    unknown_priors = set(priors) - set(config["priors"])
    if unknown_priors:
        raise ValueError(f"Unknown priors: {sorted(unknown_priors)}")
    torch.set_num_threads(1)
    selected_device = select_device(device)
    prepare_couplings(root, config)
    for seed in seeds:
        for prior_name in priors:
            parent = root / "models" / f"seed_{seed}" / prior_name
            parent.mkdir(parents=True, exist_ok=True)
            if "sf2m" in methods:
                base = parent / "_sf2m_base"
                if not _complete(base):
                    train_sf2m_base(root, seed, prior_name, config, selected_device)
                if not _complete(parent / "sf2m"):
                    continue_sf2m(root, seed, prior_name, config, selected_device)
            if set(methods) & {"ssfm", "clift"}:
                base = parent / "_map_base"
                if not _complete(base):
                    train_map_base(root, seed, prior_name, config, selected_device)
                for method in ("ssfm", "clift"):
                    if method in methods and not _complete(parent / method):
                        continue_map(root, seed, prior_name, method, config, selected_device)
            if "real_clift" in methods and not _complete(parent / "real_clift"):
                train_real_clift(root, seed, prior_name, config, selected_device)
