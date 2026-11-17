#!/usr/bin/env python3
"""Validation-selected expressive classifiers on the existing Calderon PCA100.

Search writes checkpoints and validation predictions only. The separately
frozen selection stage precedes test evaluation. This keeps new architecture,
regularisation and ensemble choices independent of the reported test scores.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/calderon2022/matplotlib"))

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from scripts.train_calderon_lineage_classifiers import load_data, scores
from scripts import train_cite_multi_celltype_classifiers as baseline_helpers

TARGETS = ("branch_id", "broad_tissue")
VARIANTS = ("random", "heldout_8_10")
RECIPES = {
    "wide": dict(family="wide", dropout=0.1, weight_power=0.25,
                 smoothing=0.0, mixup=0.0, noise=0.0, ema=0.0),
    "wide_balanced": dict(family="wide", dropout=0.1, weight_power=0.5,
                          smoothing=0.0, mixup=0.0, noise=0.0, ema=0.0),
    "residual": dict(family="residual", dropout=0.1, weight_power=0.25,
                     smoothing=0.0, mixup=0.0, noise=0.0, ema=0.0),
    "residual_mixup_ema": dict(family="residual", dropout=0.1, weight_power=0.25,
                              smoothing=0.02, mixup=0.2, noise=0.03, ema=0.995),
    "joint": dict(family="wide", dropout=0.1, weight_power=0.25,
                  smoothing=0.0, mixup=0.0, noise=0.0, ema=0.0,
                  tissue_loss_weight=0.5, consistency_weight=0.1),
}


def log(message):
    print(message, flush=True)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


class ResidualBlock(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(256)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 256)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.dropout(F.gelu(self.fc1(self.norm(x))))
        return x + 0.5 * self.dropout(self.fc2(h))


class ExpressiveClassifier(nn.Module):
    def __init__(self, family, dropout, class_sizes):
        super().__init__()
        if family == "wide":
            modules = []
            previous = 100
            for width in (512, 256, 128):
                modules.extend((nn.Linear(previous, width), nn.BatchNorm1d(width),
                                nn.GELU(), nn.Dropout(dropout)))
                previous = width
            self.trunk = nn.Sequential(*modules)
            output_width = 128
        elif family == "residual":
            self.trunk = nn.Sequential(nn.Linear(100, 256), nn.GELU(),
                                       ResidualBlock(dropout), ResidualBlock(dropout),
                                       ResidualBlock(dropout), nn.LayerNorm(256))
            output_width = 256
        else:
            raise ValueError(f"Unknown network family: {family}")
        self.heads = nn.ModuleDict({name: nn.Linear(output_width, size)
                                    for name, size in class_sizes.items()})

    def forward(self, x):
        h = self.trunk(x)
        return {name: head(h) for name, head in self.heads.items()}


def build_model(metadata):
    return ExpressiveClassifier(metadata["recipe_config"]["family"],
                                metadata["recipe_config"]["dropout"],
                                metadata["class_sizes"])


def load_checkpoint_model(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(checkpoint["metadata"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint["metadata"]


def predict_logits(model, x, device, batch_size=4096):
    model.eval()
    outputs = {name: [] for name in model.heads}
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = x[start:start + batch_size].to(device)
            for name, logits in model(batch).items():
                outputs[name].append(logits.cpu().numpy())
    result = {name: np.concatenate(values) for name, values in outputs.items()}
    if not all(np.isfinite(value).all() for value in result.values()):
        raise ValueError("Nonfinite validation logits")
    return result


def quick_metrics(truth, logits):
    predicted = logits.argmax(axis=1)
    n_classes = logits.shape[1]
    matrix = np.bincount(truth * n_classes + predicted,
                         minlength=n_classes ** 2).reshape(n_classes, n_classes)
    support = matrix.sum(axis=1)
    denominator = support + matrix.sum(axis=0)
    f1 = np.divide(2 * matrix.diagonal(), denominator,
                   out=np.zeros(n_classes), where=denominator > 0)
    # Stable unweighted, unsmoothed cross entropy for comparable validation loss.
    from scipy.special import logsumexp
    loss = (logsumexp(logits.astype(np.float64), axis=1)
            - logits[np.arange(len(truth)), truth]).mean()
    return float((predicted == truth).mean()), float(f1[support > 0].mean()), float(loss)


def hierarchy_js(branch_logits, tissue_logits, mapping):
    branch_tissue = F.softmax(branch_logits, dim=1) @ mapping
    tissue = F.softmax(tissue_logits, dim=1)
    midpoint = (branch_tissue + tissue) * 0.5
    log_midpoint = midpoint.clamp_min(1e-8).log()
    value = (branch_tissue * (branch_tissue.clamp_min(1e-8).log() - log_midpoint)
             + tissue * (tissue.clamp_min(1e-8).log() - log_midpoint)).sum(dim=1)
    return value.mean() * 0.5


def fit_recipe(args, data, variant, recipe_name, targets):
    x, obs, features, indices = data
    recipe = RECIPES[recipe_name]
    run_id = f"{variant}_{'+'.join(targets)}_{recipe_name}_seed{args.seed}"
    run_dir = args.output_dir / "search" / run_id
    complete_path = run_dir / "complete.json"
    if complete_path.exists() and not args.overwrite:
        completed = json.loads(complete_path.read_text())
        settings = {"max_epochs": args.max_epochs, "patience": args.patience,
                    "batch_size": args.batch_size, "learning_rate": args.learning_rate,
                    "recipe_config": recipe}
        if any(completed[key] != value for key, value in settings.items()):
            raise ValueError(f"Search settings changed for {run_id}; pass --overwrite")
        log(f"Reusing completed search run: {run_id}")
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    classes = {name: sorted(obs[name].astype(str).unique().tolist()) for name in targets}
    encoded = {}
    counts = {}
    for name in targets:
        ids = {label: i for i, label in enumerate(classes[name])}
        encoded[name] = obs[name].astype(str).map(ids).to_numpy(dtype=np.int64)
        counts[name] = np.bincount(encoded[name][indices["train"]], minlength=len(ids))
        if np.any(counts[name] == 0):
            raise ValueError("Training data do not cover the fixed label vocabulary")
    scaler = StandardScaler().fit(x[indices["train"]])
    train_x = scaler.transform(x[indices["train"]]).astype(np.float32)
    val_x = scaler.transform(x[indices["validation"]]).astype(np.float32)
    baseline_helpers.seed_everything(np, torch, args.seed)
    device = torch.device(args.device)
    if device.type == "mps":
        torch.mps.manual_seed(args.seed)
    class_sizes = {name: len(values) for name, values in classes.items()}
    model = ExpressiveClassifier(recipe["family"], recipe["dropout"], class_sizes).to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    train_tensor = torch.from_numpy(train_x).to(device)
    val_tensor = torch.from_numpy(val_x)
    train_y = {name: torch.from_numpy(encoded[name][indices["train"]]).to(device)
               for name in targets}
    val_y = {name: encoded[name][indices["validation"]] for name in targets}
    criteria = {}
    for name in targets:
        weights = (len(train_x) / (len(classes[name]) * counts[name])) ** recipe["weight_power"]
        criteria[name] = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device),
                                             label_smoothing=recipe["smoothing"])
    mapping = None
    if len(targets) == 2:
        branch_to_tissue = obs.groupby("branch_id", observed=True).broad_tissue.first().astype(str).to_dict()
        tissue_ids = {label: i for i, label in enumerate(classes["broad_tissue"])}
        matrix = np.zeros((len(classes["branch_id"]), len(tissue_ids)), dtype=np.float32)
        for row, branch in enumerate(classes["branch_id"]):
            matrix[row, tissue_ids[branch_to_tissue[branch]]] = 1
        mapping = torch.from_numpy(matrix).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    ema = None
    if recipe["ema"]:
        from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
        ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(recipe["ema"]))
    generator = torch.Generator().manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    best = {name: dict(score=-np.inf, loss=np.inf, epoch=0, state=None) for name in targets}
    stopping_score = -np.inf
    bad_epochs = 0
    rows = []
    started = time.monotonic()
    log(f"SEARCH {run_id}: {parameter_count:,} parameters, {len(train_x):,} training cells, device={device}")
    for epoch in range(1, args.max_epochs + 1):
        if epoch <= 5:
            learning_rate = args.learning_rate * epoch / 5
        else:
            progress = (epoch - 5) / max(args.max_epochs - 5, 1)
            learning_rate = args.learning_rate * (0.02 + 0.98 * (1 + math.cos(math.pi * progress)) / 2)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        model.train()
        order = torch.randperm(len(train_x), generator=generator).to(device)
        total_loss = torch.zeros((), device=device)
        total_correct = {name: torch.zeros((), device=device, dtype=torch.int64) for name in targets}
        seen = 0
        for start in range(0, len(train_x), args.batch_size):
            selected = order[start:start + args.batch_size]
            if len(selected) == 1:
                continue
            batch_x = train_tensor[selected]
            batch_y = {name: labels[selected] for name, labels in train_y.items()}
            if recipe["noise"]:
                batch_x = batch_x + torch.randn_like(batch_x) * recipe["noise"]
            mixing = 1.0
            permutation = None
            if recipe["mixup"]:
                mixing = max(value := float(rng.beta(recipe["mixup"], recipe["mixup"])), 1 - value)
                permutation = torch.randperm(len(selected), generator=generator).to(device)
                batch_x = mixing * batch_x + (1 - mixing) * batch_x[permutation]
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            losses = {}
            for name in targets:
                value = criteria[name](logits[name], batch_y[name])
                if permutation is not None:
                    value = mixing * value + (1 - mixing) * criteria[name](logits[name], batch_y[name][permutation])
                losses[name] = value
            if len(targets) == 2:
                loss = (losses["branch_id"] + recipe["tissue_loss_weight"] * losses["broad_tissue"]
                        + recipe["consistency_weight"] * hierarchy_js(logits["branch_id"], logits["broad_tissue"], mapping))
            else:
                loss = losses[targets[0]]
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            if ema is not None:
                ema.update_parameters(model)
            total_loss += loss.detach() * len(selected)
            for name in targets:
                total_correct[name] += (logits[name].argmax(dim=1) == batch_y[name]).sum().detach()
            seen += len(selected)
        evaluated_model = ema.module if ema is not None else model
        val_logits = predict_logits(evaluated_model, val_tensor, device)
        row = dict(epoch=epoch, train_loss=float(total_loss.cpu()) / seen,
                   learning_rate=learning_rate, elapsed_seconds=time.monotonic() - started)
        current_scores = []
        for name in targets:
            accuracy, macro_f1, val_loss = quick_metrics(val_y[name], val_logits[name])
            if not np.isfinite(val_loss) or not np.isfinite(row["train_loss"]):
                raise ValueError(f"Nonfinite training/validation loss for {run_id}")
            row[f"{name}_train_accuracy"] = float(total_correct[name].cpu()) / seen
            row[f"{name}_validation_accuracy"] = accuracy
            row[f"{name}_validation_macro_f1"] = macro_f1
            row[f"{name}_validation_loss"] = val_loss
            current_scores.append(macro_f1)
            previous = best[name]
            if macro_f1 > previous["score"] or (macro_f1 == previous["score"] and val_loss < previous["loss"]):
                best[name] = dict(score=macro_f1, loss=val_loss, epoch=epoch,
                                  state={key: value.detach().cpu().clone()
                                         for key, value in evaluated_model.state_dict().items()})
        aggregate_score = float(np.mean(current_scores))
        if aggregate_score > stopping_score + 0.0005:
            stopping_score = aggregate_score
            bad_epochs = 0
        else:
            bad_epochs += 1
        rows.append(row)
        pd.DataFrame(rows).to_csv(run_dir / "history.csv", index=False)
        if epoch == 1 or epoch % 5 == 0:
            text = "; ".join(f"{name}: val acc={row[f'{name}_validation_accuracy']:.3f}, macro-F1={row[f'{name}_validation_macro_f1']:.3f}" for name in targets)
            log(f"{run_id}: epoch {epoch}, {text}, train loss={row['train_loss']:.3f}, elapsed={row['elapsed_seconds']:.0f}s")
        if bad_epochs >= args.patience:
            log(f"{run_id}: early stopping at epoch {epoch}")
            break
    digest = hashlib.sha256()
    digest.update(obs.index.to_numpy(dtype=str).tobytes())
    digest.update(obs.classifier_split.astype(str).to_numpy(dtype=str).tobytes())
    common_metadata = dict(recipe=recipe_name, recipe_config=recipe, variant=variant,
                           seed=args.seed, preprocessing_seed=0, n_pcs=100,
                           source_h5ad=str(args.data_dir / f"lineage_assigned_hvg3000_pca100_{variant}.h5ad"),
                           class_sizes=class_sizes, all_class_names=classes,
                           selected_genes=features, parameter_count=parameter_count,
                           scaler_mean=scaler.mean_.tolist(), scaler_scale=scaler.scale_.tolist(),
                           cell_and_split_sha256=digest.hexdigest(),
                           split_sizes={name: len(value) for name, value in indices.items()},
                           max_epochs=args.max_epochs, epochs_run=len(rows), batch_size=args.batch_size,
                           learning_rate=args.learning_rate, weight_decay=1e-4,
                           patience=args.patience, gradient_clip=5.0,
                           scheduler="5-epoch linear warmup, cosine decay to 2% of initial LR",
                           selection_metric="validation macro-F1; ties use lower unweighted validation cross entropy",
                           device=str(device), elapsed_seconds=time.monotonic() - started,
                           classifier_inputs=getattr(args, "classifier_inputs", "Existing training-fitted HVG3000/PCA100 only"))
    from scipy.special import softmax
    for name in targets:
        metadata = dict(common_metadata, target=name, class_names=classes[name],
                        best_epoch=best[name]["epoch"], best_validation_macro_f1_during_fit=best[name]["score"])
        cpu_model = build_model(metadata)
        cpu_model.load_state_dict(best[name]["state"])
        cpu_model.eval()
        logits = predict_logits(cpu_model, val_tensor, torch.device("cpu"))[name]
        metadata["validation"] = scores(val_y[name], softmax(logits, axis=1), counts[name])
        checkpoint = run_dir / f"{name}.pt"
        torch.save(dict(model_state_dict=best[name]["state"], metadata=metadata), checkpoint)
        np.savez_compressed(run_dir / f"{name}_validation_logits.npz", logits=logits,
                            class_names=np.asarray(classes[name]),
                            cell_and_split_sha256=np.asarray(digest.hexdigest()))
        write_json(run_dir / f"{name}_validation.json", metadata)
        reloaded, _ = load_checkpoint_model(checkpoint)
        with torch.no_grad():
            check = reloaded(val_tensor[:256])[name].numpy()
        np.testing.assert_allclose(check, logits[:256], rtol=1e-5, atol=1e-5)
        log(f"VALIDATION COMPLETE {run_id}, {name}: acc={metadata['validation']['accuracy']:.4f}, macro-F1={metadata['validation']['macro_f1']:.4f}, best epoch={metadata['best_epoch']}")
    write_json(complete_path, common_metadata)
    if device.type == "mps":
        del model, train_tensor, train_y, optimizer, ema
        torch.mps.empty_cache()


def search(args):
    write_json(args.output_dir / "search_plan.json", dict(
        recipes=RECIPES, variants=args.variants, targets=list(TARGETS), seed=args.seed,
        max_epochs=args.max_epochs, patience=args.patience, batch_size=args.batch_size,
        learning_rate=args.learning_rate, data_directory=str(args.data_dir),
        selection="Validation macro-F1 only; no new test scores during search",
        ensemble="Validation-selected greedy probability averaging, at most three members; temperatures fit on validation only"))
    for variant in args.variants:
        path = args.data_dir / f"lineage_assigned_hvg3000_pca100_{variant}.h5ad"
        data = load_data(path, expected_seed=0, expected_variant=variant)
        for recipe in args.recipes:
            if recipe == "joint":
                fit_recipe(args, data, variant, recipe, TARGETS)
            else:
                for target in TARGETS:
                    fit_recipe(args, data, variant, recipe, (target,))
    log("Search completed. Only validation scores have been evaluated.")


def fit_temperature(logits, truth):
    from scipy.optimize import minimize_scalar
    from scipy.special import logsumexp
    logits = logits.astype(np.float64)

    def objective(temperature):
        scaled = logits / temperature
        return float((logsumexp(scaled, axis=1)
                      - scaled[np.arange(len(truth)), truth]).mean())

    fitted = minimize_scalar(objective, bounds=(0.35, 3.0), method="bounded",
                             options=dict(xatol=1e-4))
    return float(fitted.x)


def scaled_probabilities(logits, temperature):
    from scipy.special import softmax
    return softmax(logits.astype(np.float64) / temperature, axis=1).astype(np.float32)


def greedy_ensemble(probabilities, truth, counts, max_members=3):
    individual = {name: scores(truth, values, counts)
                  for name, values in probabilities.items()}
    best_name = max(individual, key=lambda name: (individual[name]["macro_f1"],
                                                 -individual[name]["log_loss"]))
    members = [best_name]
    current = probabilities[best_name]
    current_metrics = individual[best_name]
    while len(members) < max_members:
        options = []
        for name, values in probabilities.items():
            if name in members:
                continue
            proposed = (current * len(members) + values) / (len(members) + 1)
            measured = scores(truth, proposed, counts)
            options.append((measured["macro_f1"], -measured["log_loss"], name, proposed, measured))
        if not options:
            break
        winner = max(options, key=lambda value: (value[0], value[1]))
        if winner[0] < current_metrics["macro_f1"] + 0.0005:
            break
        members.append(winner[2])
        current = winner[3]
        current_metrics = winner[4]
    return members, current, current_metrics, individual


def baseline_model(path, metadata):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = baseline_helpers.make_model(nn, 100, len(metadata["class_names"]),
                                        metadata["dropout"], (128, 64))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def baseline_logits(model, x, batch_size=4096):
    with torch.no_grad():
        return np.concatenate([model(torch.from_numpy(x[start:start + batch_size])).numpy()
                               for start in range(0, len(x), batch_size)])


def freeze_selection(args):
    plan = json.loads((args.output_dir / "search_plan.json").read_text())
    if plan["recipes"] != RECIPES or plan["seed"] != args.seed:
        raise ValueError("Search configuration differs from the requested selection")
    groups = []
    for variant in args.variants:
        path = args.data_dir / f"lineage_assigned_hvg3000_pca100_{variant}.h5ad"
        x, obs, _, indices = load_data(path, expected_seed=0, expected_variant=variant)
        scaler = StandardScaler().fit(x[indices["train"]])
        val_x = scaler.transform(x[indices["validation"]]).astype(np.float32)
        for target in TARGETS:
            baseline_stem = f"calderon_{target}_hvg3000_pca100_{variant}_mlp"
            metadata_path = ROOT / "calderon-classifiers/reports" / f"{baseline_stem}_metrics.json"
            original = json.loads(metadata_path.read_text())
            checkpoint = ROOT / "calderon-classifiers" / f"{baseline_stem}.pt"
            classes = original["class_names"]
            ids = {label: i for i, label in enumerate(classes)}
            encoded = obs[target].astype(str).map(ids).to_numpy(dtype=np.int64)
            truth = encoded[indices["validation"]]
            counts = np.bincount(encoded[indices["train"]], minlength=len(classes))
            np.testing.assert_allclose(scaler.mean_, original["scaler_mean"], rtol=1e-8, atol=1e-8)
            candidates = [dict(id="baseline", kind="baseline", checkpoint=str(checkpoint),
                               metadata=original, logits=baseline_logits(baseline_model(checkpoint, original), val_x))]
            for recipe_name in args.recipes:
                run_targets = TARGETS if recipe_name == "joint" else (target,)
                run_id = f"{variant}_{'+'.join(run_targets)}_{recipe_name}_seed{args.seed}"
                folder = args.output_dir / "search" / run_id
                if not (folder / "complete.json").exists():
                    raise ValueError(f"Search incomplete: {run_id}")
                metadata = json.loads((folder / f"{target}_validation.json").read_text())
                if metadata["class_names"] != classes or metadata["cell_and_split_sha256"] != original["cell_and_split_sha256"]:
                    raise ValueError("Candidate and baseline classes or held-out cells differ")
                with np.load(folder / f"{target}_validation_logits.npz", allow_pickle=False) as stored:
                    logits = stored["logits"].copy()
                    assert np.array_equal(stored["class_names"], classes)
                candidates.append(dict(id=recipe_name, kind="expressive", checkpoint=str(folder / f"{target}.pt"),
                                       metadata=metadata, logits=logits))
            calibrated = {}
            descriptions = []
            for candidate in candidates:
                temperature = fit_temperature(candidate["logits"], truth)
                probabilities = scaled_probabilities(candidate["logits"], temperature)
                calibrated[candidate["id"]] = probabilities
                measured = scores(truth, probabilities, counts)
                descriptions.append(dict(id=candidate["id"], kind=candidate["kind"],
                                         checkpoint=candidate["checkpoint"], temperature=temperature,
                                         parameter_count=(candidate["metadata"].get("parameter_count")
                                                          or sum(p.numel() for p in baseline_model(Path(candidate["checkpoint"]), candidate["metadata"]).parameters())),
                                         validation=measured,
                                         original_validation=candidate["metadata"].get("validation") or candidate["metadata"]["metrics"]["validation"]))
            members, average, measured, individual = greedy_ensemble(calibrated, truth, counts)
            ensemble_temperature = fit_temperature(np.log(np.maximum(average, 1e-30)), truth)
            ensemble_prob = scaled_probabilities(np.log(np.maximum(average, 1e-30)), ensemble_temperature)
            ensemble_metrics = scores(truth, ensemble_prob, counts)
            best_expressive = max((item for item in descriptions if item["kind"] == "expressive"),
                                  key=lambda item: (item["validation"]["macro_f1"], -item["validation"]["log_loss"]))["id"]
            single_winner = max(individual, key=lambda name: (individual[name]["macro_f1"], -individual[name]["log_loss"]))
            selected = "ensemble" if len(members) > 1 else members[0]
            group = dict(variant=variant, target=target, class_names=classes,
                         cell_and_split_sha256=original["cell_and_split_sha256"],
                         split_sizes={name: len(value) for name, value in indices.items()},
                         candidates=descriptions, best_expressive=best_expressive,
                         single_winner=single_winner, ensemble_members=members,
                         ensemble_temperature=ensemble_temperature, selected=selected,
                         ensemble_validation=ensemble_metrics,
                         selection_metric="Validation macro-F1; CE used only for ties",
                         minimum_ensemble_gain=0.0005, test_used_for_selection=False)
            groups.append(group)
            log(f"FROZEN {variant}/{target}: best new NN={best_expressive}, selected={selected}, members={members}, val macro-F1={ensemble_metrics['macro_f1']:.4f}")
    result = dict(schema_version=1, groups=groups, recipes=RECIPES, seed=args.seed,
                  selection_frozen_before_new_test_evaluation=True,
                  calibration="Scalar temperatures fit on validation negative log likelihood only")
    write_json(args.output_dir / "selection.json", result)
    log("All model and ensemble choices frozen; test evaluation may now begin.")


def predict_selected(manifest_path, x_pca, batch_size=4096):
    """Load a selected network/ensemble and predict in its original PCA space."""
    manifest_path = Path(manifest_path)
    spec = json.loads(manifest_path.read_text())
    x = np.asarray(x_pca, dtype=np.float32).copy()
    x -= np.asarray(spec["scaler_mean"], dtype=np.float64)
    x /= np.asarray(spec["scaler_scale"], dtype=np.float64)
    probabilities = []
    for member in spec["members"]:
        checkpoint = (manifest_path.parent / member["checkpoint"]).resolve()
        if member["kind"] == "baseline":
            stored = torch.load(checkpoint, map_location="cpu", weights_only=False)
            logits = baseline_logits(baseline_model(checkpoint, stored["metadata"]), x, batch_size)
        else:
            model, metadata = load_checkpoint_model(checkpoint)
            if metadata["class_names"] != spec["class_names"]:
                raise ValueError("Selected model and manifest label vocabularies differ")
            logits = predict_logits(model, torch.from_numpy(x), torch.device("cpu"), batch_size)[spec["target"]]
        probabilities.append(scaled_probabilities(logits, member["temperature"]))
    average = np.mean(probabilities, axis=0)
    return scaled_probabilities(np.log(np.maximum(average, 1e-30)), spec["temperature"])


def save_test_details(args, group, method, probabilities, truth, obs, indices, counts,
                      validation, selection_hash):
    from sklearn.metrics import classification_report, confusion_matrix
    prefix = args.output_dir / "reports" / f"{group['variant']}_{group['target']}_{method}"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    classes = group["class_names"]
    predicted = probabilities.argmax(axis=1)
    measured = scores(truth, probabilities, counts)
    record = dict(method=method, target=group["target"], variant=group["variant"],
                  class_names=classes, validation=validation, test=measured,
                  selected=method == group["selected"],
                  best_expressive=method == group["best_expressive"],
                  selection_sha256=selection_hash,
                  cell_and_split_sha256=group["cell_and_split_sha256"],
                  selection_metric=group["selection_metric"], test_used_for_selection=False)
    if method == "ensemble":
        record["members"] = group["ensemble_members"]
        record["temperature"] = group["ensemble_temperature"]
    else:
        candidate = next(value for value in group["candidates"] if value["id"] == method)
        record["parameter_count"] = candidate["parameter_count"]
        record["temperature"] = candidate["temperature"]
        record["checkpoint"] = candidate["checkpoint"]
    report = classification_report(truth, predicted, labels=np.arange(len(classes)),
                                   target_names=classes, output_dict=True, zero_division=0)
    per_class = pd.DataFrame({label: report[label] for label in classes}).T
    per_class["train_cells"] = counts
    per_class.to_csv(prefix.with_name(prefix.name + "_per_class.csv"), index_label="class")
    confusion = confusion_matrix(truth, predicted, labels=np.arange(len(classes)))
    pd.DataFrame(confusion, index=classes, columns=classes).to_csv(prefix.with_name(prefix.name + "_confusion.csv"))
    test_obs = obs.iloc[indices["test"]]
    windows = test_obs.day.astype(str).to_numpy()
    window_rows = []
    for window in sorted(set(windows), key=lambda value: int(value.split("-")[0])):
        selected = windows == window
        subset = scores(truth[selected], probabilities[selected], counts)
        window_rows.append(dict(window=window, **{key: subset[key] for key in
                            ("n_cells", "n_present_classes", "accuracy", "balanced_accuracy", "macro_f1")}))
    pd.DataFrame(window_rows).to_csv(prefix.with_name(prefix.name + "_by_window.csv"), index=False)
    destination = args.data_dir / "predictions/expressive"
    destination.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dict(cell_id=test_obs.index, assigned_label=np.asarray(classes)[truth],
                      predicted_label=np.asarray(classes)[predicted], confidence=probabilities.max(axis=1),
                      inferred_window=windows)).to_csv(destination / f"{prefix.name}.csv.gz", index=False)
    write_json(prefix.with_name(prefix.name + "_metrics.json"), record)
    log(f"TEST {group['variant']}/{group['target']}/{method}: acc={measured['accuracy']:.4f}, macro-F1={measured['macro_f1']:.4f}")
    return record


def evaluate_selected(args):
    selection_path = args.output_dir / "selection.json"
    selection = json.loads(selection_path.read_text())
    if not selection["selection_frozen_before_new_test_evaluation"]:
        raise ValueError("Model selection has not been frozen")
    selection_hash = hashlib.sha256(selection_path.read_bytes()).hexdigest()
    records = []
    current_variant = None
    for group in selection["groups"]:
        variant, target = group["variant"], group["target"]
        if variant != current_variant:
            path = args.data_dir / f"lineage_assigned_hvg3000_pca100_{variant}.h5ad"
            x, obs, _, indices = load_data(path, expected_seed=0, expected_variant=variant)
            scaler = StandardScaler().fit(x[indices["train"]])
            test_x = scaler.transform(x[indices["test"]]).astype(np.float32)
            current_variant = variant
        ids = {label: i for i, label in enumerate(group["class_names"])}
        encoded = obs[target].astype(str).map(ids).to_numpy(dtype=np.int64)
        truth = encoded[indices["test"]]
        counts = np.bincount(encoded[indices["train"]], minlength=len(ids))
        probabilities = {}
        for candidate in group["candidates"]:
            checkpoint = Path(candidate["checkpoint"])
            if candidate["kind"] == "baseline":
                stored = torch.load(checkpoint, map_location="cpu", weights_only=False)
                metadata = stored["metadata"]
                logits = baseline_logits(baseline_model(checkpoint, metadata), test_x)
            else:
                model, metadata = load_checkpoint_model(checkpoint)
                logits = predict_logits(model, torch.from_numpy(test_x), torch.device("cpu"))[target]
            np.testing.assert_allclose(scaler.mean_, metadata["scaler_mean"], rtol=1e-8, atol=1e-8)
            probabilities[candidate["id"]] = scaled_probabilities(logits, candidate["temperature"])
            record = save_test_details(args, group, candidate["id"], probabilities[candidate["id"]],
                                       truth, obs, indices, counts, candidate["validation"], selection_hash)
            if candidate["kind"] == "baseline":
                for metric in ("accuracy", "macro_f1", "balanced_accuracy"):
                    np.testing.assert_allclose(record["test"][metric], metadata["metrics"]["test"][metric], rtol=0, atol=1e-12)
            records.append(record)
        if len(group["ensemble_members"]) > 1:
            average = np.mean([probabilities[name] for name in group["ensemble_members"]], axis=0)
            probabilities["ensemble"] = scaled_probabilities(np.log(np.maximum(average, 1e-30)), group["ensemble_temperature"])
            records.append(save_test_details(args, group, "ensemble", probabilities["ensemble"], truth,
                                              obs, indices, counts, group["ensemble_validation"], selection_hash))
        chosen = group["ensemble_members"] if group["selected"] == "ensemble" else [group["selected"]]
        manifest_path = args.output_dir / "models" / f"{variant}_{target}_selected.json"
        members = []
        for name in chosen:
            candidate = next(value for value in group["candidates"] if value["id"] == name)
            members.append(dict(checkpoint=os.path.relpath(candidate["checkpoint"], manifest_path.parent),
                                kind=candidate["kind"], temperature=candidate["temperature"], method=name))
        manifest = dict(target=target, variant=variant, class_names=group["class_names"], members=members,
                        temperature=group["ensemble_temperature"] if group["selected"] == "ensemble" else 1.0,
                        scaler_mean=scaler.mean_.tolist(), scaler_scale=scaler.scale_.tolist(), n_pcs=100,
                        selection_sha256=selection_hash, selected=group["selected"],
                        source_h5ad=str(args.data_dir / f"lineage_assigned_hvg3000_pca100_{variant}.h5ad"))
        write_json(manifest_path, manifest)
        reloaded = predict_selected(manifest_path, x[indices["test"][:256]])
        np.testing.assert_allclose(reloaded, probabilities[group["selected"]][:256], rtol=1e-5, atol=1e-5)
        baseline_correct = probabilities["baseline"].argmax(axis=1) == truth
        selected_correct = probabilities[group["selected"]].argmax(axis=1) == truth
        improved = int((selected_correct & ~baseline_correct).sum())
        worsened = int((baseline_correct & ~selected_correct).sum())
        write_json(manifest_path.with_name(manifest_path.stem + "_validation.json"),
                   dict(probabilities_reproduced_after_manifest_reload=True,
                        max_abs_probability_difference=float(np.max(abs(reloaded - probabilities[group["selected"]][:256]))),
                        paired_cells_improved_over_baseline=improved,
                        paired_cells_worsened_from_baseline=worsened,
                        paired_net_accuracy_change=(improved - worsened) / len(truth),
                        selection_sha256=selection_hash))
    frame = pd.DataFrame([dict(target=record["target"], variant=record["variant"], method=record["method"],
                              selected=record["selected"], best_expressive=record["best_expressive"],
                              validation_macro_f1=record["validation"]["macro_f1"],
                              accuracy=record["test"]["accuracy"], balanced_accuracy=record["test"]["balanced_accuracy"],
                              macro_f1=record["test"]["macro_f1"], test_cells=record["test"]["n_cells"],
                              test_classes=record["test"]["n_present_classes"], classes=len(record["class_names"]))
                          for record in records])
    frame.to_csv(args.output_dir / "reports/results.csv", index=False)
    write_json(args.output_dir / "evaluation_complete.json", dict(selection_sha256=selection_hash,
               methods_evaluated=len(records), selection_used_validation_only=True, selected_manifests_verified=True))
    log("All new held-out evaluations completed; selected model manifests verified.")


METHOD_LABELS = {
    "baseline": "Original 128 / 64 MLP",
    "wide": "Wide 512 / 256 / 128",
    "wide_balanced": "Wide + stronger class weights",
    "residual": "Residual 256, three blocks",
    "residual_mixup_ema": "Residual + mixup + EMA",
    "joint": "Joint branch / tissue heads",
    "ensemble": "Validation-selected ensemble",
}


def comparison_figures(args, frame, selection):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({"svg.fonttype": "none", "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(2, 2, figsize=(17, 11))
    for column, variant in enumerate(VARIANTS):
        for row, target in enumerate(TARGETS):
            ax = axes[row, column]
            shown = frame[(frame.variant == variant) & (frame.target == target)].set_index("method")
            methods = [name for name in METHOD_LABELS if name in shown.index]
            y = np.arange(len(methods))
            ax.barh(y - .17, shown.loc[methods].accuracy, height=.32, color="#507F9D", label="Accuracy")
            ax.barh(y + .17, shown.loc[methods].macro_f1, height=.32, color="#82A88E", label="Macro-F1")
            names = [METHOD_LABELS[name] + (" *" if bool(shown.loc[name, "selected"]) else "") for name in methods]
            ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9)
            for location, name in enumerate(methods):
                for offset, metric in ((-.17, "accuracy"), (.17, "macro_f1")):
                    value = shown.loc[name, metric]
                    ax.text(value + .008, location + offset, f"{value:.1%}", va="center", fontsize=8)
            target_title = "Branch IDs" if target == "branch_id" else "Broad tissues"
            protocol = "Random cell split" if variant == "random" else "Held-out 8-10 h window"
            ax.set(title=f"{target_title}: {protocol}", xlim=(0, 1), xlabel="Held-out test score")
            ax.xaxis.set_major_formatter(PercentFormatter(1)); ax.invert_yaxis(); ax.grid(axis="x", alpha=.15)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .945), ncol=2)
    fig.suptitle("Calderon PCA100: expressive classifiers and training recipes\n* Selected using validation macro-F1", fontsize=15, y=.995)
    fig.tight_layout(rect=(0, 0, 1, .90))
    fig.savefig(args.output_dir / "reports/expressive_comparison.png", dpi=180)
    fig.savefig(args.output_dir / "reports/expressive_comparison.svg"); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for column, variant in enumerate(VARIANTS):
        for row, target in enumerate(TARGETS):
            ax = axes[row, column]
            group = next(value for value in selection["groups"] if value["variant"] == variant and value["target"] == target)
            for candidate in group["candidates"]:
                if candidate["id"] == "baseline":
                    ax.axhline(candidate["validation"]["macro_f1"], color="#666666", linestyle="--", label="Original MLP")
                    continue
                history = pd.read_csv(Path(candidate["checkpoint"]).parent / "history.csv")
                ax.plot(history.epoch, history[f"{target}_validation_macro_f1"], label=METHOD_LABELS[candidate["id"]])
            protocol = "Random cell split" if variant == "random" else "Held-out 8-10 h window"
            ax.set(title=f"{'Branch IDs' if target == 'branch_id' else 'Broad tissues'}: {protocol}",
                   xlabel="Epoch", ylabel="Validation macro-F1", ylim=(.3, .8))
            ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.grid(alpha=.15)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("Validation macro-F1 during classifier training", y=.995)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .95), ncol=3, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, .87))
    fig.savefig(args.output_dir / "reports/validation_curves.png", dpi=180); plt.close(fig)


def selected_detail_figures(args, selection):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    for group in selection["groups"]:
        prefix = args.output_dir / "reports" / f"{group['variant']}_{group['target']}_{group['selected']}"
        per_class = pd.read_csv(prefix.with_name(prefix.name + "_per_class.csv"), index_col=0)
        matrix = pd.read_csv(prefix.with_name(prefix.name + "_confusion.csv"), index_col=0).to_numpy()
        absent = matrix.sum(axis=1) == 0
        normalised = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
        shown = np.ma.masked_array(normalised, mask=np.broadcast_to(absent[:, None], matrix.shape))
        cmap = plt.get_cmap("Blues").copy(); cmap.set_bad("#ECEEEF")
        protocol = "Random cell split" if group["variant"] == "random" else "Held-out 8-10 h window"
        title = f"{'Branch IDs' if group['target'] == 'branch_id' else 'Broad tissues'}: {protocol}\n{METHOD_LABELS[group['selected']]}"
        size = 14 if len(per_class) > 30 else 10
        fig, ax = plt.subplots(figsize=(size, size), constrained_layout=True)
        picture = ax.imshow(shown, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks(np.arange(len(per_class))); ax.set_yticks(np.arange(len(per_class)))
        fontsize = 7 if len(per_class) > 30 else 9
        ax.set_xticklabels(per_class.index, rotation=90, fontsize=fontsize)
        ax.set_yticklabels(per_class.index, fontsize=fontsize)
        for row, column in zip(*np.where(normalised >= .15)):
            ax.text(column, row, f"{normalised[row, column]:.0%}", ha="center", va="center", fontsize=6.5,
                    color="white" if normalised[row, column] > .5 else "#1F3444")
        if absent.any():
            title += "\nGrey rows: no cells of this class in the test split"
        ax.set(title=title, xlabel="Predicted label", ylabel="Assigned label")
        fig.colorbar(picture, ax=ax, shrink=.7, label="Fraction within assigned class")
        fig.savefig(prefix.with_name(prefix.name + "_confusion.png"), dpi=180)
        fig.savefig(prefix.with_name(prefix.name + "_confusion.svg")); plt.close(fig)
        shown = per_class[per_class.support > 0].sort_values("recall")
        fig, ax = plt.subplots(figsize=(11, max(6, len(shown) * .23 + 1.5)), constrained_layout=True)
        ax.barh(np.arange(len(shown)), shown.recall, color="#507F9D")
        ax.set_yticks(np.arange(len(shown))); ax.set_yticklabels(shown.index, fontsize=8)
        for row, (_, values) in enumerate(shown.iterrows()):
            ax.text(1.02, row, f"n={int(values.support)}", va="center", fontsize=8)
        ax.set(title=title, xlim=(0, 1.2), xlabel="Recall on held-out cells")
        ax.set_xticks(np.linspace(0, 1, 6)); ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis="x", alpha=.15)
        fig.savefig(prefix.with_name(prefix.name + "_recall.png"), dpi=180)
        fig.savefig(prefix.with_name(prefix.name + "_recall.svg")); plt.close(fig)


def write_report(args):
    selection_path = args.output_dir / "selection.json"
    selection = json.loads(selection_path.read_text())
    selection_hash = hashlib.sha256(selection_path.read_bytes()).hexdigest()
    completed = json.loads((args.output_dir / "evaluation_complete.json").read_text())
    if completed["selection_sha256"] != selection_hash:
        raise ValueError("Evaluation does not correspond to the current frozen selection")
    frame = pd.read_csv(args.output_dir / "reports/results.csv")
    records = [json.loads(path.read_text()) for path in (args.output_dir / "reports").glob("*_metrics.json")]
    records = [record for record in records if record["selection_sha256"] == selection_hash]
    lookup = {(record["variant"], record["target"], record["method"]): record for record in records}
    comparison_figures(args, frame, selection)
    selected_detail_figures(args, selection)
    completed_runs = list((args.output_dir / "search").glob("*/complete.json"))
    lines = ["# Calderon PCA100: expressive classifiers and training recipes", "",
             f"This follow-up trained **{len(completed_runs)} new networks** on the same 314,387 assigned RNA profiles, "
             "the same training-fitted HVG3000/PCA100 spaces and the same random-cell and held-out 8-10 h splits as the "
             "[original classifier pilot](classification.md). Joint models provide separately evaluated branch and tissue heads.", "",
             "Network checkpoints and recipes were selected using validation macro-F1. Probability temperatures and ensembles "
             "were fitted/selected on validation cells only; [selection.json](../../calderon-classifiers/expressive/selection.json) "
             "was written before any new test evaluation. The test cells are the same ones used in the previous pilot.", "",
             "## Original versus validation-selected results", "",
             "| Protocol | Target | Selected method | Original accuracy | New accuracy | Original macro-F1 | New macro-F1 |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
    summary = []
    for group in selection["groups"]:
        base = lookup[group["variant"], group["target"], "baseline"]["test"]
        chosen = lookup[group["variant"], group["target"], group["selected"]]["test"]
        protocol = "Random cells" if group["variant"] == "random" else "Entire 8-10 h window"
        target = "Branch IDs" if group["target"] == "branch_id" else "Broad tissues"
        lines.append(f"| {protocol} | {target} | {METHOD_LABELS[group['selected']]} | {base['accuracy']:.2%} | "
                     f"{chosen['accuracy']:.2%} | {base['macro_f1']:.2%} | {chosen['macro_f1']:.2%} |")
        summary.append(dict(protocol=protocol, target=target, accuracy_gain=100 * (chosen["accuracy"] - base["accuracy"]),
                            macro_f1_gain=100 * (chosen["macro_f1"] - base["macro_f1"])))
    lines.extend(["", "The selected method is the validation winner, not the largest score observed on the test set. "
                  "Accuracy and macro-F1 gains in percentage points are:", ""])
    for item in summary:
        lines.append(f"- {item['target']}, {item['protocol'].lower()}: accuracy {item['accuracy_gain']:+.2f} points; "
                     f"macro-F1 {item['macro_f1_gain']:+.2f} points.")
    random_gains = [item["accuracy_gain"] for item in summary if item["protocol"] == "Random cells"]
    temporal_gains = [item["accuracy_gain"] for item in summary if item["protocol"] != "Random cells"]
    lines.extend(["", f"The selected predictors gain {min(random_gains):.2f}–{max(random_gains):.2f} accuracy points "
                  f"on random held-out cells, and {min(temporal_gains):.2f}–{max(temporal_gains):.2f} points on the held-out developmental window. "
                  "The detailed recipe table shows that extra capacity alone can reduce performance on the held-out window. "
                  "Within this search, regularisation and probability averaging provide modest improvements but leave the cross-stage "
                  "classification problem unresolved. Coarser labels remain easier to recover; the remaining errors still require "
                  "classifier uncertainty to be included in a downstream flow experiment."])
    lines.extend(["", "![All recipes on held-out cells](../../calderon-classifiers/expressive/reports/expressive_comparison.png)", "",
                  "## Recipes and training", "",
                  "| Recipe | Network | Changes from the original training |",
                  "| --- | --- | --- |",
                  "| Wide | 100 -> 512 -> 256 -> 128 -> output, BatchNorm and GELU | Dropout 0.1; no label smoothing; class-weight power 0.25 |",
                  "| Wide balanced | Same wide network | Increase class-weight power to 0.5 |",
                  "| Residual | Width 256; three pre-LayerNorm residual blocks, 128-unit bottlenecks; GELU | Dropout 0.1; no label smoothing; class-weight power 0.25 |",
                  "| Residual + mixup + EMA | Same residual network | Mixup alpha 0.2; Gaussian PC noise SD 0.03; label smoothing 0.02; EMA decay 0.995 |",
                  "| Joint | Wide shared trunk, separate 53-branch and 24-tissue heads | Branch CE + 0.5 tissue CE + 0.1 Jensen-Shannon consistency between tissue probabilities and aggregated branch probabilities |", "",
                  "The new single-head networks have about 221k–239k parameters, compared with about 23k–25k in the original MLP. "
                  "All new recipes use AdamW, initial learning rate 0.001, weight decay 0.0001, gradient clipping at 5, "
                  "a five-epoch warmup and cosine decay to 2% of the initial rate. They are allowed up to 160 epochs, "
                  "with early stopping after 30 epochs without a validation macro-F1 gain of at least 0.0005. "
                  "Each joint head keeps its own best validation checkpoint; the two checkpoints can come from different epochs.", "",
                  "The PCA representation and PC scaler are unchanged and fitted on training cells only. Neither inferred time "
                  "nor annotation metadata enters the network inputs. Joint training uses both available training labels, "
                  "whose branch-to-tissue relationship is deterministic in the authors' decoder.", "",
                  "![Validation learning curves](../../calderon-classifiers/expressive/reports/validation_curves.png)", ""])
    lines.extend(["## All individual recipes and ensembles", "",
                  "| Protocol | Target | Method | Validation macro-F1 | Test accuracy | Test mean recall | Test macro-F1 | Selected |",
                  "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |"])
    for group in selection["groups"]:
        for method in METHOD_LABELS:
            key = (group["variant"], group["target"], method)
            if key not in lookup:
                continue
            record = lookup[key]
            val, test = record["validation"], record["test"]
            protocol = "Random" if group["variant"] == "random" else "8-10 h held out"
            target = "Branch" if group["target"] == "branch_id" else "Tissue"
            lines.append(f"| {protocol} | {target} | {METHOD_LABELS[method]} | {val['macro_f1']:.2%} | "
                         f"{test['accuracy']:.2%} | {test['balanced_accuracy']:.2%} | {test['macro_f1']:.2%} | "
                         f"{'Yes' if record['selected'] else ''} |")
    lines.extend(["", "## Ensembles and probability calibration", "",
                  "Each model receives a scalar temperature fitted to validation negative log likelihood over the range 0.35–3.0. "
                  "This changes confidence but preserves an individual model's predicted class. Greedy probability averaging "
                  "starts with the highest validation macro-F1 model and adds at most two other members; an addition requires "
                  "a validation macro-F1 gain of at least 0.0005. A final temperature is fitted to the averaged validation probabilities. "
                  "These choices are frozen before test evaluation.", "",
                  "| Protocol | Target | Ensemble members | Final temperature |",
                  "| --- | --- | --- | ---: |"])
    for group in selection["groups"]:
        lines.append(f"| {group['variant']} | {group['target']} | {', '.join(group['ensemble_members'])} | {group['ensemble_temperature']:.3f} |")
    lines.extend(["", "The following confidence statistics are for the validation-selected predictors, after calibration. "
                  "Calibration is fitted on the other atlas bins in the temporal protocol and can still be inaccurate on the held-out bin.", "",
                  "| Protocol | Target | Coverage at probability >= 0.9 | Accuracy within subset | Test log loss | Test ECE |",
                  "| --- | --- | ---: | ---: | ---: | ---: |"])
    for group in selection["groups"]:
        test = lookup[group["variant"], group["target"], group["selected"]]["test"]
        subset = test["confidence_subsets"]["0.9"]
        subset_accuracy = f"{subset['accuracy']:.2%}" if subset["accuracy"] is not None else "n/a"
        lines.append(f"| {group['variant']} | {group['target']} | {subset['coverage']:.2%} | {subset_accuracy} | "
                     f"{test['log_loss']:.3f} | {test['expected_calibration_error_15bins']:.3f} |")
    temporal = {group["target"]: lookup[group["variant"], group["target"], group["selected"]]["test"]
                for group in selection["groups"] if group["variant"] == "heldout_8_10"}
    high_accuracy = {}
    for target, values in temporal.items():
        value = values["confidence_subsets"]["0.9"]["accuracy"]
        high_accuracy[target] = f"{value:.2%}" if value is not None else "n/a"
    lines.extend(["", f"On the held-out window, a 0.9 probability cutoff gives {high_accuracy['branch_id']} branch accuracy "
                  f"and {high_accuracy['broad_tissue']} tissue accuracy within the retained subsets. The corresponding ECE values "
                  f"are {temporal['branch_id']['expected_calibration_error_15bins']:.3f} and "
                  f"{temporal['broad_tissue']['expected_calibration_error_15bins']:.3f}. "
                  "Higher random-cell accuracy does not establish reliable probabilities for other developmental stages."])
    lines.extend(["", "## Remaining weak classes", "",
                  "The five lowest-recall classes present in each selected predictor's test split are listed below; "
                  "complete per-class CSV files and selected-model confusion/recall plots are in the report directory.", "",
                  "| Protocol | Target | Assigned label | Test cells | Recall |",
                  "| --- | --- | --- | ---: | ---: |"])
    for group in selection["groups"]:
        stem = f"{group['variant']}_{group['target']}_{group['selected']}"
        per_class = pd.read_csv(args.output_dir / "reports" / f"{stem}_per_class.csv")
        for _, row in per_class[per_class.support > 0].sort_values("recall").head(5).iterrows():
            lines.append(f"| {group['variant']} | {group['target']} | `{row['class']}` | {int(row.support):,} | {row.recall:.2%} |")
    lines.extend(["", "## Data and evaluation scope", "",
                  "The random protocol has 251,509 training, 31,439 validation and 31,439 test cells. "
                  "The temporal protocol has 230,744 training and 25,639 validation cells outside 8-10 h, and 58,004 test cells in 8-10 h. "
                  "The latter test set contains 27 of the 53 branch IDs and 20 of the 24 broad labels; macro-F1 and mean recall "
                  "average classes present in each evaluated split. Every class remains represented in training.", "",
                  "This is a single-seed search over five declared recipes in PCA100. The labels and time coordinate "
                  "are the authors' atlas-derived annotations; broad labels are the branch decoder, including `Unknown`. "
                  "The random protocol shares collection groups across splits. These results measure recovery of assigned labels, "
                  "and the temporal test measures transfer to an expression-inferred age window, rather than independent-embryo or lineage-tracing accuracy.", "",
                  "## Reproduce and use the selected classifiers", "", "```sh",
                  "python scripts/train_calderon_expressive_classifiers.py --stage search --device mps",
                  "python scripts/train_calderon_expressive_classifiers.py --stage freeze",
                  "python scripts/train_calderon_expressive_classifiers.py --stage evaluate",
                  "python scripts/train_calderon_expressive_classifiers.py --stage report", "```", "",
                  "Use `--device cpu` for search when MPS is unavailable. Test inference and exported-model verification use CPU. "
                  "The original HVG/PCA preparation does not need to be rerun.", "",
                  "Checkpoints and validation histories are under `calderon-classifiers/expressive/search/`. "
                  "The four selected predictor manifests are under `calderon-classifiers/expressive/models/`, "
                  "with relative checkpoint paths, scalers, label vocabularies and fitted temperatures. "
                  "These larger architectures require the accompanying loader rather than the original fixed-width MLP loader:", "",
                  "```python", "from scripts.train_calderon_expressive_classifiers import predict_selected",
                  "probabilities = predict_selected(",
                  "    'calderon-classifiers/expressive/models/heldout_8_10_broad_tissue_selected.json',",
                  "    pca100_cells,", ")", "```", "",
                  "Pass cells projected into the corresponding stored PCA100 space. "
                  "`reports/results.csv` contains all scores; each method has per-class, per-window and confusion CSV files. "
                  "Per-cell predictions are in the ignored `datasets/calderon2022/predictions/expressive/` directory. "
                  "Selected predictor manifests are checked by reloading and reproducing 256 test-cell probability vectors.", ""])
    report = ROOT / "docs/calderon2022/expressive_classification.md"
    report.write_text("\n".join(lines) + "\n")
    log(f"Expressive classifier report written: {report}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "datasets/calderon2022")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "calderon-classifiers/expressive")
    parser.add_argument("--stage", choices=("search", "freeze", "evaluate", "report", "all"), default="all")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--recipes", nargs="+", choices=tuple(RECIPES), default=list(RECIPES))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=160)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if min(args.max_epochs, args.patience, args.batch_size, args.threads) <= 0:
        parser.error("Epoch, patience, batch-size and thread settings must be positive")
    args.data_dir = args.data_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    return args


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    with threadpool_limits(limits=args.threads):
        if args.stage in ("search", "all"):
            search(args)
        if args.stage in ("freeze", "all"):
            freeze_selection(args)
        if args.stage in ("evaluate", "all"):
            evaluate_selected(args)
        if args.stage in ("report", "all"):
            write_report(args)


if __name__ == "__main__":
    main()
