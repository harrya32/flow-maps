"""Profile existing Maizels components without running or changing experiments.

Run native first with the flowmaps environment, then otpfm with mfm_env.
Results use real retained-day cells and the current model presets on CPU.
"""

import argparse
import json
import math
import platform
import statistics
import sys
from collections import OrderedDict
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / "py"))


def measure(fn, repeats=9, block=lambda value: value):
    start = perf_counter()
    value = block(fn())
    first = perf_counter() - start
    elapsed = []
    for _ in range(repeats):
        start = perf_counter()
        value = block(fn())
        elapsed.append(perf_counter() - start)
    return {
        "first_call_s": first,
        "median_s": statistics.median(elapsed),
        "min_s": min(elapsed),
        "max_s": max(elapsed),
        "repeats": repeats,
    }, value


def emit(label, value):
    print(json.dumps({label: value}, default=str), flush=True)


def native():
    import jax
    from common import interpolant, logging, losses, maizels, state_utils, updates
    from configs.maizels_pca50 import get_config

    cfg = get_config(
        6,
        dataset_location=str(ROOT / "celltype_classification_pca50_dataset.csv.gz"),
        classifier_path=str(OUT / "selected_days_classifier.npz"),
        maizels_schedule="d3_d3p8_d8",
        maizels_ot_coupling="minibatch_ot",
        seed=0,
    )
    cfg.problem.ot_progress_enabled = False
    cfg.problem.ot_verbose = False
    cfg.problem.ot_cache_enabled = False
    result = {
        "backend": "native",
        "platform": platform.platform(),
        "jax_version": jax.__version__,
        "devices": [str(device) for device in jax.devices()],
        "numpy_version": np.__version__,
        "config": cfg.to_dict(),
        "components": {},
    }
    start = perf_counter()
    pools, _ = maizels.make_minibatch_ot_training_pools(cfg)
    result["data_load_s"] = perf_counter() - start
    splits = maizels.timepoint_pool_splits(cfg)
    retained = list(cfg.problem.retained_timepoints)
    arrays = {}
    counts = {}
    for index, day in enumerate(retained):
        split = splits[day]
        arrays[f"train_x{index}"] = split["train_x"]
        arrays[f"validation_x{index}"] = split["holdout_x"]
        arrays[f"train_types{index}"] = maizels._cell_type_ids(split["train_types"])
        counts[day] = {
            "train": len(split["train_x"]),
            "validation": len(split["holdout_x"]),
        }
    np.savez_compressed(OUT / "retained_cells.npz", **arrays)
    result["cells"] = counts
    result["full_support_counts"] = []
    reachable = maizels.build_transition_reachable("descendant")
    class_mask = np.array(
        [[maizels.endpoint_valid(a, b, reachable) for b in maizels.CLASS_NAMES]
         for a in maizels.CLASS_NAMES], dtype=bool
    )
    for index, (left, right) in enumerate(zip(retained[:-1], retained[1:])):
        source_counts = np.bincount(arrays[f"train_types{index}"], minlength=9)
        target_counts = np.bincount(arrays[f"train_types{index + 1}"], minlength=9)
        allowed = int(np.sum(source_counts[:, None] * target_counts[None, :] * class_mask))
        result["full_support_counts"].append({
            "interval": f"{left}->{right}",
            "all_edges": len(arrays[f"train_x{index}"]) * len(arrays[f"train_x{index + 1}"]),
            "endpoint_allowed_edges": allowed,
            "classifier_point_checks_before_interpolant_filter": allowed * cfg.problem.n_interpolant_check_times,
        })
    emit("real_data", {"cells": counts, "full_support": result["full_support_counts"]})

    seed = [720]
    pair = None
    for mode in ("ot_plain", "ot_endpoint", "ot_endpoint_interpolant"):
        def couple():
            seed[0] += 1
            return maizels.couple_minibatch_ot_timepoint_pools(
                cfg, pools, cfg.optimization.bs, seed=seed[0], pair_mode=mode
            )
        timing, (paired, stats) = measure(couple)
        timing["last_stats"] = stats
        result["components"][f"minibatch_pairing_{mode}"] = timing
        emit(f"minibatch_pairing_{mode}", timing)
        if mode == "ot_endpoint_interpolant":
            pair = paired

    # The legacy global path checks every endpoint-valid candidate interpolant.
    # Cap this diagnostic; do not construct the full multimillion-variable LP.
    index = 0
    x0 = arrays["train_x0"][:64]
    x1 = arrays["train_x1"][:64]
    ids0 = arrays["train_types0"][:64]
    ids1 = arrays["train_types1"][:64]
    names = np.asarray(maizels.CLASS_NAMES)
    for mode in ("ot_plain", "ot_endpoint_interpolant"):
        def collect():
            return maizels._collect_exact_ot_edges(
                cfg, x0, names[ids0], ids0, x1, names[ids1], ids1,
                pair_mode=mode, endpoint_reachable=reachable,
                lineage_transition_mode="descendant",
            )
        timing, (_, _, _, stats) = measure(collect, repeats=3)
        timing.update({"source_cells": 64, "target_cells": 64, "last_stats": stats})
        result["components"][f"global_edge_construction_{mode}_64x64"] = timing
        emit(f"global_edge_construction_{mode}_64x64", timing)

    weighted_mean = weighted_second = total_weight = 0.0
    for interval in pools["intervals"]:
        for role in ("source_time", "target_time"):
            x = pools["timepoints"][interval[role]]["x"].astype(np.float64)
            weight = float(interval["nominal_pairs"]) * x.shape[1]
            weighted_mean += weight * np.mean(x)
            weighted_second += weight * np.mean(x * x)
            total_weight += weight
    cfg.network.rescale = float(np.sqrt(weighted_second / total_weight - (weighted_mean / total_weight) ** 2))
    result["native_rescale"] = cfg.network.rescale
    state, net, schedule, _ = state_utils.setup_training_state(
        cfg, np.ones((50,), dtype=np.float32), jax.random.PRNGKey(1)
    )
    result["trainable_parameters"] = sum(np.size(v) for v in jax.tree_util.tree_leaves(state.params))
    interp = interpolant.setup_interpolant(cfg)
    bs = len(pair["x0"])
    rng = np.random.default_rng(712)
    local = np.sort(rng.random((bs, 2)), axis=1).astype(np.float32)
    diag_bs = int(bs * cfg.optimization.diag_fraction)
    local[:diag_bs, 0] = local[:diag_bs, 1]
    bounds = pair["label"][:, 2:4]
    times = bounds[:, :1] + local * (bounds[:, 1:2] - bounds[:, :1])
    batch_args = tuple(jax.numpy.asarray(value) if value is not None else None for value in (
        pair["x0"], pair["x1"], pair["label"], times[:, 0], times[:, 1],
        None, None, jax.random.split(jax.random.PRNGKey(3), bs),
        np.ones(bs, dtype=np.float32), np.ones(bs, dtype=np.float32),
    ))
    train_step = updates.setup_train_step(cfg)
    for enabled in (False, True):
        run_cfg = type(cfg)(cfg.to_dict())
        run_cfg.constraints.enabled = enabled
        loss = losses.setup_loss(run_cfg, net, interp)
        carry = [state]
        def update():
            current = carry[0]
            new, value, grads = train_step(current, loss, (current.params, *batch_args))
            carry[0] = new
            return new, value, grads
        timing, (trained_state, value, grads) = measure(update, repeats=15, block=jax.block_until_ready)
        timing["last_loss"] = float(value)
        if not np.isfinite(timing["last_loss"]):
            raise RuntimeError("Native profiling loss became non-finite")
        name = "native_gradient_with_lineage" if enabled else "native_gradient_without_lineage"
        result["components"][name] = timing
        emit(name, timing)
        if enabled:
            loss_with_constraint = loss
            state_with_constraint = trained_state
            grads_with_constraint = grads

    statics = SimpleNamespace(interp=interp, net=net, loss=loss_with_constraint)
    def diagnostics():
        return logging.compute_constraint_metrics(
            cfg, state_with_constraint, (state_with_constraint.params, *batch_args), statics=statics
        )
    timing, _ = measure(diagnostics, block=jax.block_until_ready)
    result["components"]["native_lineage_diagnostics_current"] = timing
    emit("native_lineage_diagnostics_current", timing)
    fused_diagnostics = jax.jit(lambda params, args: logging.compute_constraint_metrics(
        cfg, state_with_constraint.replace(params=params), (params, *args), statics=statics
    ))
    timing, _ = measure(lambda: fused_diagnostics(state_with_constraint.params, batch_args), block=jax.block_until_ready)
    result["components"]["native_lineage_diagnostics_jitted_comparison"] = timing
    emit("native_lineage_diagnostics_jitted_comparison", timing)
    timing, _ = measure(lambda: logging.compute_grad_norm(grads_with_constraint), block=jax.block_until_ready)
    result["components"]["native_gradient_norm_logging"] = timing

    validation_args = tuple(
        None if value is None else jax.numpy.tile(value, (8,) + (1,) * (value.ndim - 1))
        for value in batch_args
    )
    def validation():
        return loss_with_constraint(state_with_constraint.params, state_with_constraint.params, *validation_args)
    timing, _ = measure(validation, repeats=5, block=jax.block_until_ready)
    result["components"]["native_validation_1024_current"] = timing
    emit("native_validation_1024_current", timing)
    fused_validation = jax.jit(loss_with_constraint)
    timing, _ = measure(lambda: fused_validation(state_with_constraint.params, state_with_constraint.params, *validation_args), repeats=5, block=jax.block_until_ready)
    result["components"]["native_validation_1024_jitted_comparison"] = timing
    emit("native_validation_1024_jitted_comparison", timing)
    ema = updates.setup_ema_update(cfg)
    timing, _ = measure(lambda: ema(state_with_constraint), block=jax.block_until_ready)
    result["components"]["native_ema_two_copies"] = timing
    result["config"] = cfg.to_dict()
    (OUT / "native_profile.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    emit("saved", str(OUT / "native_profile.json"))


def otpfm():
    import torch
    import ot
    sys.path.insert(0, str(ROOT / "OTP-FM" / "src"))
    from otpfm import OTPFM
    from otpfm.potentials import W2InfPotential, W2Potential

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    cfg = json.loads((ROOT / "OTP-FM/configs/citeseq/defaults.json").read_text())
    cfg.update(json.loads((ROOT / "OTP-FM/configs/citeseq/W2.json").read_text()))
    cfg.update(solver_type="direct", pca_dim=50, tks=[0.16])
    result = {
        "backend": "otpfm", "device": "cpu", "torch_version": torch.__version__,
        "numpy_version": np.__version__, "pot_version": ot.__version__,
        "torch_threads": torch.get_num_threads(), "config": cfg,
        "components": {},
    }
    with np.load(OUT / "retained_cells.npz") as raw:
        train = [raw[f"train_x{i}"] for i in range(3)]
        validation = [raw[f"validation_x{i}"] for i in range(3)]
    all_train = np.concatenate(train)
    center = all_train.mean(axis=0, keepdims=True)
    scale = float(all_train.std(axis=0).max())
    train = [((x - center) / scale).astype(np.float32) for x in train]
    validation = [((x - center) / scale).astype(np.float32) for x in validation]
    mappings = []
    result["startup_alignments"] = []
    for split_name, populations in (("train", train), ("validation", validation)):
        for index in range(2):
            start = perf_counter()
            cost = ot.dist(populations[index], populations[index + 1], metric="sqeuclidean")
            cost_time = perf_counter() - start
            start = perf_counter()
            plan = ot.emd(ot.unif(len(populations[index])), ot.unif(len(populations[index + 1])), cost)
            solve_time = perf_counter() - start
            start = perf_counter()
            mapping = np.argmax(plan, axis=1)
            extraction_time = perf_counter() - start
            if split_name == "train":
                mappings.append(mapping)
            record = {"split": split_name, "interval_index": index, "shape": list(cost.shape), "cost_build_s": cost_time, "solve_s": solve_time, "argmax_s": extraction_time}
            result["startup_alignments"].append(record)
            emit("startup_alignment", record)
            del plan, cost
    result["training_budget"] = {
        "tuples_per_epoch": len(train[0]),
        "batch_size": cfg["batch_size"],
        "updates_per_epoch": math.ceil(len(train[0]) / cfg["batch_size"]),
        "epochs": cfg["epochs"],
        "total_updates": cfg["epochs"] * math.ceil(len(train[0]) / cfg["batch_size"]),
        "total_tuples": cfg["epochs"] * len(train[0]),
    }
    potential = W2InfPotential(tk=0.16, strength=cfg["strength"], lambda_type=cfg["lambda_type"], width=cfg["lambda_width"])
    model = OTPFM(
        d=50, tks=[0.16], potentials=OrderedDict([(0.16, potential)]),
        flownet_args={
            "x_emb_dim": cfg["x_emb_dim"], "t_emb_dim": cfg["t_emb_dim"],
            "hidden_dim": cfg["hidden_dim"], "num_hidden_layers": cfg["num_hidden_layers"],
            "x_hidden_layers": cfg["x_hidden_layers"], "t_hidden_layers": cfg["t_hidden_layers"],
            "activation_fn": cfg["activation"], "layernorm": cfg["layernorm"],
            "dropout": cfg["dropout"], "residual_every": cfg["residual_every"],
        }, consistency_loss=cfg["consistency_loss"], lossfn=cfg["lossfn"],
        time_sampler=cfg["time_sampler"], diag_prob=cfg["diag_prob"],
        euler_steps=cfg["euler_steps"], ema_decay=cfg["ema_decay"],
        ema_ot=cfg["ema_ot"], jvp_clamp=cfg["jvp_clamp"], solver_type="direct",
        picard_steps=cfg["picard_steps"],
    )
    model.train()
    result["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rng = np.random.default_rng(721)
    index0 = rng.choice(len(train[0]), cfg["batch_size"], replace=True)
    index1 = mappings[0][index0]
    index2 = mappings[1][index1]
    batch = torch.as_tensor(np.stack([train[0][index0], train[1][index1], train[2][index2]], axis=1))
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    for use_potential in (False, True):
        def step():
            optimizer.zero_grad()
            loss = model.forward_with_loss(batch, otp_alpha=1.0, do_otp=use_potential)
            loss.backward()
            optimizer.step()
            model.update_ema()
            return float(loss.detach())
        timing, value = measure(step, repeats=15)
        timing["last_loss"] = value
        if not np.isfinite(value):
            raise RuntimeError("OTP-FM profiling loss became non-finite")
        name = "otpfm_step_with_w2inf" if use_potential else "otpfm_step_without_potential"
        result["components"][name] = timing
        emit(name, timing)
    stages = {key: [] for key in ("forward", "backward", "optimizer", "ema")}
    for _ in range(9):
        optimizer.zero_grad()
        start = perf_counter()
        loss = model.forward_with_loss(batch, otp_alpha=1.0, do_otp=True)
        stages["forward"].append(perf_counter() - start)
        start = perf_counter()
        loss.backward()
        stages["backward"].append(perf_counter() - start)
        start = perf_counter()
        optimizer.step()
        stages["optimizer"].append(perf_counter() - start)
        start = perf_counter()
        model.update_ema()
        stages["ema"].append(perf_counter() - start)
    result["stage_medians_s"] = {key: statistics.median(value) for key, value in stages.items()}
    emit("stage_medians_s", result["stage_medians_s"])
    with torch.no_grad():
        timing, _ = measure(lambda: model.solve_X_tks(batch[:, 0], batch[:, -1] - batch[:, 0], batch[:, 1].unsqueeze(0), [potential], [0.16], 1.0))
        result["components"]["otpfm_w2inf_direct_solver"] = timing
        exact = W2Potential(tk=0.16, strength=cfg["strength"], lambda_type="delta", width=0.0)
        timing, _ = measure(lambda: exact.grad_gk(batch[:, 1], 0.84 * batch[:, 0] + 0.16 * batch[:, -1]))
        result["components"]["otpfm_actual_w2_gradient_only"] = timing
    model.eval()
    original_lossfn = model.lossfn
    model.lossfn = "mse"
    with torch.no_grad():
        timing, _ = measure(lambda: model.forward_with_loss(batch, otp_alpha=1.0, do_otp=True), repeats=7)
    model.lossfn = original_lossfn
    result["components"]["otpfm_validation_forward_256"] = timing
    (OUT / "otpfm_profile.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    emit("saved", str(OUT / "otpfm_profile.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("backend", choices=("native", "otpfm"))
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        (native if args.backend == "native" else otpfm)()
