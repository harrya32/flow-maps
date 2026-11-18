"""Fused JAX interpolant classification and hard tissue-transition filtering.

Only endpoint coordinates enter the device; full probabilities remain there.
The original decoder handles numerically ambiguous paths so a near-tie does
not silently change which pairs enter the masked OT refinement loop.
"""

from __future__ import annotations

import numpy as np


class JAXInterpolantFilter:
    def __init__(self, classifier, allowed, n_times, prob_threshold=0.,
                 margin_threshold=0., decision_tolerance=1e-4):
        import jax
        import jax.numpy as jnp

        self.classifier = classifier
        self.allowed = np.asarray(allowed, dtype=bool)
        self.n_times = int(n_times)
        self.prob_threshold = float(prob_threshold)
        self.margin_threshold = float(margin_threshold)
        self.decision_tolerance = float(decision_tolerance)
        n_classes = len(classifier.class_names)
        if self.allowed.shape != (n_classes, n_classes) or n_classes < 2:
            raise ValueError("GPU interpolant filter requires a matching tissue relation.")
        if self.n_times < 1:
            raise ValueError("Interpolant check count must be positive.")
        if not (0 <= self.prob_threshold <= 1 and 0 <= self.margin_threshold <= 1
                and 0 < self.decision_tolerance <= 1):
            raise ValueError("Invalid GPU filter confidence or decision tolerance.")
        # Bound classifier memory and reuse a small set of compiled shapes.
        # Padding whole paths also avoids splitting a path across device calls.
        max_pairs = max(1, int(classifier.batch_size) // self.n_times)
        self.pair_batch_size = 1 << (max_pairs.bit_length() - 1)
        self.taus = np.linspace(0., 1., self.n_times + 2, dtype=np.float32)[1:-1]
        taus = jnp.asarray(self.taus)[None, :, None]
        mean = jnp.asarray(classifier.mean, dtype=jnp.float32)
        scale = jnp.asarray(classifier.scale, dtype=jnp.float32)
        temperatures = jnp.asarray([m["temperature"] for m in classifier.spec["members"]], dtype=jnp.float32)
        ensemble_temperature = jnp.asarray(classifier.spec["temperature"], dtype=jnp.float32)
        relation = jnp.asarray(self.allowed)
        raw_labels = self.prob_threshold == 0 and self.margin_threshold == 0

        def filter_batch(source, source_ids, target, target_ids):
            points = source[:, None] * (1 - taus) + target[:, None] * taus
            normalized = (points.reshape(-1, 100) - mean) / scale
            logits = classifier._logits(normalized)
            member_probs = jax.nn.softmax(logits / temperatures[:, None, None], axis=-1)
            average = jnp.mean(member_probs, axis=0)
            if raw_labels:
                # Positive ensemble-temperature calibration is monotone and
                # cannot change the argmax. Its extra log/softmax is unused by
                # a raw-label filter; reference fallback covers near-ties.
                probabilities = average.reshape(source.shape[0], self.n_times, n_classes)
            else:
                probabilities = jax.nn.softmax(
                    jnp.log(jnp.maximum(average, 1e-30)) / ensemble_temperature, axis=-1,
                ).reshape(source.shape[0], self.n_times, n_classes)
            predicted = jnp.argmax(probabilities, axis=-1)
            confidence = jnp.max(probabilities, axis=-1)
            second = jnp.max(jnp.where(jnp.arange(n_classes) == predicted[..., None],
                                      -jnp.inf, probabilities), axis=-1)
            margin = confidence - second
            confident = (jnp.ones(predicted.shape, dtype=bool) if raw_labels else
                         ((confidence >= self.prob_threshold) & (margin >= self.margin_threshold)))

            if raw_labels:
                # With every point assessed, successive labels suffice; no
                # stateful scan or early exit is necessary for the first fault.
                previous = jnp.concatenate([source_ids[:, None], predicted[:, :-1]], axis=1)
                rejected_at_checkpoint = jnp.any(~relation[previous, predicted], axis=1)
                rejected_at_final = (~rejected_at_checkpoint
                                     & ~relation[predicted[:, -1], target_ids])
            else:
                def assess(carry, item):
                    current, rejected = carry
                    label, assessed = item
                    active = assessed & ~rejected
                    bad = active & ~relation[current, label]
                    return (jnp.where(active & ~bad, label, current), rejected | bad), None

                (current, rejected_at_checkpoint), _ = jax.lax.scan(
                    assess, (source_ids, jnp.zeros(source.shape[0], dtype=bool)),
                    (predicted.T, confident.T),
                )
                # The observed final label is appended as a confident one-hot
                # prediction by the original reference checker.
                rejected_at_final = ~rejected_at_checkpoint & ~relation[current, target_ids]

            ambiguous = jnp.any(margin <= self.decision_tolerance, axis=1)
            if self.prob_threshold > 0:
                ambiguous |= jnp.any(jnp.abs(confidence - self.prob_threshold)
                                     <= self.decision_tolerance, axis=1)
            if self.margin_threshold > 0:
                ambiguous |= jnp.any(jnp.abs(margin - self.margin_threshold)
                                     <= self.decision_tolerance, axis=1)
            ambiguous |= ~jnp.all(jnp.isfinite(probabilities), axis=(1, 2))
            return {
                "valid": ~(rejected_at_checkpoint | rejected_at_final),
                "confident": confident,
                "rejected_at_checkpoint": rejected_at_checkpoint,
                "rejected_at_final": rejected_at_final,
                "ambiguous": ambiguous,
            }

        self._filter_batch = jax.jit(filter_batch)

    def __call__(self, source_x, source_type_ids, target_x, target_type_ids):
        import jax
        import jax.numpy as jnp
        from .calderon import interpolant_validity

        source = np.asarray(source_x, dtype=np.float32)
        target = np.asarray(target_x, dtype=np.float32)
        source_ids = np.asarray(source_type_ids, dtype=np.int32)
        target_ids = np.asarray(target_type_ids, dtype=np.int32)
        n_pairs = len(source)
        if (source.shape != (n_pairs, 100) or target.shape != source.shape or n_pairs == 0
                or source_ids.shape != (n_pairs,) or target_ids.shape != (n_pairs,)
                or not np.isfinite(source).all() or not np.isfinite(target).all()):
            raise ValueError("GPU filter requires nonempty finite PCA100 endpoint pairs.")
        if (source_ids.min() < 0 or target_ids.min() < 0
                or source_ids.max() >= len(self.allowed) or target_ids.max() >= len(self.allowed)):
            raise ValueError("GPU filter endpoint tissue IDs are outside its vocabulary.")
        parts = []
        for start in range(0, n_pairs, self.pair_batch_size):
            end = min(start + self.pair_batch_size, n_pairs)
            count = end - start
            padded = 1 << (count - 1).bit_length()
            arrays = [source[start:end], source_ids[start:end], target[start:end], target_ids[start:end]]
            if count < padded:
                arrays = [np.pad(value, ((0, padded - count),) + ((0, 0),) * (value.ndim - 1), mode="edge")
                          for value in arrays]
            checked = jax.device_get(self._filter_batch(*(jnp.asarray(value) for value in arrays)))
            parts.append({name: np.asarray(value)[:count] for name, value in checked.items()})
        result = {name: np.concatenate([part[name] for part in parts]) for name in parts[0]}
        ambiguous = np.flatnonzero(result.pop("ambiguous"))
        if len(ambiguous):
            times = self.taus[None, :, None]
            paths = source[ambiguous, None] * (1 - times) + target[ambiguous, None] * times
            probabilities = self.classifier.predict_proba(paths.reshape(-1, 100)).reshape(
                len(ambiguous), self.n_times, len(self.allowed))
            reference = interpolant_validity(source_ids[ambiguous], target_ids[ambiguous], probabilities,
                                             self.allowed, self.prob_threshold, self.margin_threshold)
            for name in result:
                result[name][ambiguous] = reference[name]
        result.update({"n_confident": int(result["confident"].sum()), "n_points": n_pairs * self.n_times,
                       "reference_fallback_pairs": len(ambiguous),
                       "reference_fallback_points": len(ambiguous) * self.n_times})
        result["reference_fallback_mask"] = np.zeros(n_pairs, dtype=bool)
        result["reference_fallback_mask"][ambiguous] = True
        return result


def get_jax_interpolant_filter(classifier, allowed, n_times, prob_threshold=0.,
                               margin_threshold=0., decision_tolerance=1e-4):
    """Share compiled kernels across intervals using the same frozen decoder."""
    key = (int(n_times), float(prob_threshold), float(margin_threshold), float(decision_tolerance),
           np.asarray(allowed, dtype=bool).tobytes())
    cache = getattr(classifier, "_jax_interpolant_filters", None)
    if cache is None:
        cache = classifier._jax_interpolant_filters = {}
    if key not in cache:
        cache[key] = JAXInterpolantFilter(classifier, allowed, n_times, prob_threshold,
                                          margin_threshold, decision_tolerance)
    return cache[key]


def sample_intervals_with_batched_filter(datasets, counts, keys):
    """Run independent CPU OT loops, combining their pending GPU checks.

    Each worker retains its own key, RNG, assignment/refinement state and
    partial-OT fallback. GPU inference runs only in the calling thread.
    """
    from concurrent.futures import Future, ThreadPoolExecutor
    import queue
    import threading
    import time

    original_filters = [dataset.jax_interpolant_filter for dataset in datasets]
    classifier_filter = original_filters[0]
    if classifier_filter is None or any(checker is not classifier_filter for checker in original_filters):
        raise ValueError("Batched interval checks require one shared JAX interpolant filter.")
    events = queue.Queue()
    lock = threading.Lock()
    request_futures = []
    failure = [None]
    dispatch_stats = {"dispatches": 0, "seconds": 0., "pairs": 0}

    def enqueue(index, source_x, source_type_ids, target_x, target_type_ids):
        with lock:
            if failure[0] is not None:
                raise RuntimeError("Batched GPU filtering cancelled.") from failure[0]
            future = Future()
            request_futures.append(future)
            events.put(("check", index, ((source_x, source_type_ids, target_x, target_type_ids), future)))
        return future.result()

    def sample(index):
        try:
            return datasets[index].sample_device_batch(counts[index], keys[index])
        finally:
            events.put(("done", index, None))

    for index, dataset in enumerate(datasets):
        dataset.jax_interpolant_filter = lambda *inputs, _index=index: enqueue(_index, *inputs)
    try:
        with ThreadPoolExecutor(max_workers=len(datasets), thread_name_prefix="calderon-ot") as executor:
            workers = [executor.submit(sample, index) for index in range(len(datasets))]
            active, pending = set(range(len(datasets))), {}
            try:
                while active:
                    # Every active OT loop either has a pending check or will
                    # finish. Synchronizing here amortizes a GPU launch without
                    # changing when that loop accepts or rejects its own edges.
                    while not active.issubset(pending):
                        kind, index, payload = events.get()
                        if kind == "done":
                            active.remove(index)
                            workers[index].result()
                        else:
                            pending[index] = payload
                    if not pending:
                        continue
                    indices = sorted(pending)
                    inputs = [pending[index][0] for index in indices]
                    combined = [np.concatenate([item[column] for item in inputs], axis=0) for column in range(4)]
                    started = time.monotonic()
                    checked = classifier_filter(*combined)
                    dispatch_stats["seconds"] += time.monotonic() - started
                    dispatch_stats["dispatches"] += 1
                    dispatch_stats["pairs"] += len(combined[0])
                    offset = 0
                    for index, item in zip(indices, inputs):
                        count = len(item[0])
                        end = offset + count
                        part = {name: checked[name][offset:end] for name in (
                            "valid", "confident", "rejected_at_checkpoint", "rejected_at_final", "reference_fallback_mask")}
                        fallback_pairs = int(part["reference_fallback_mask"].sum())
                        part.update({"n_confident": int(part["confident"].sum()),
                                     "n_points": count * classifier_filter.n_times,
                                     "reference_fallback_pairs": fallback_pairs,
                                     "reference_fallback_points": fallback_pairs * classifier_filter.n_times})
                        pending[index][1].set_result(part)
                        offset = end
                    pending.clear()
                parts = [worker.result() for worker in workers]
            except BaseException as error:
                # Release all blocked workers before the executor joins them,
                # including requests queued while the main thread was checking.
                with lock:
                    failure[0] = error
                    for future in request_futures:
                        if not future.done():
                            future.set_exception(RuntimeError("Batched GPU filtering cancelled."))
                raise
    finally:
        for dataset, checker in zip(datasets, original_filters):
            dataset.jax_interpolant_filter = checker
    return parts, dispatch_stats
