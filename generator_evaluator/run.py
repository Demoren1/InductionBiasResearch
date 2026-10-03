"""Run a short pattern integration check or the frozen-protocol experiment.

No experiment starts on import. Test data are materialized only after saving
the selected common mask, generator, evaluator and solver in frozen.pt.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import ExitStack
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Callable

import torch

from .adapters import (FunctionalBank, build_pattern_fixture, load_deepsets_bank,
                       make_deepsets_tasks, make_pattern_test_tasks,
                       materialize_deepsets_test_tasks, measure_mask)
from .artifacts import paired_comparison, save_json, save_torch, write_plots
from .data import InnerProtocol, RealReplay, TaskData, tensor_hash, topology_id
from .measurements import MeasurementStore as _SharedMeasurementStore
from .runtime import DenseProtocolSelector, RunSession, cpu_state, random_exact_k
from .models import QualityEnsemble, TransformerMaskGenerator
from .mask_priors import SlidingWindowMaskPrior
from .progress import progress
from .training import (direct_generator_update, generator_update, propose_candidates,
                       select_acquisition, train_evaluators)


@dataclass(frozen=True)
class SearchConfig:
    seed: int = 4100
    k: int = 32
    generator_epochs: int = 10
    updates_per_epoch: int = 10
    refresh_every: int = 5
    acquisition_budget: int = 6
    candidates: int = 24
    initial_random: int = 8
    evaluator_epochs: int = 50
    evaluator_batch_size: int = 32
    generator_lr: float = 0.001
    evaluator_lr: float = 0.001
    width: int = 64
    layers: int = 2
    heads: int = 4
    ensemble_members: int = 3
    noise_dim: int = 16
    permutation_weight: float = 1.0
    uncertainty_weight: float = 0.0
    gap_threshold: float = 0.1
    direct_control: bool = True
    smoke: bool = False

    def __post_init__(self):
        if min(self.k, self.generator_epochs, self.updates_per_epoch, self.refresh_every,
               self.acquisition_budget, self.candidates, self.initial_random,
               self.evaluator_epochs, self.evaluator_batch_size, self.ensemble_members) < 1:
            raise ValueError("search budgets must be positive")
        if self.candidates < self.acquisition_budget:
            raise ValueError("candidate pool must cover acquisition_budget")
        if self.gap_threshold <= 0:
            raise ValueError("gap_threshold must be positive")


def _cpu_state(module):
    """Compatibility seam for old callers; canonical implementation is runtime.cpu_state."""
    return cpu_state(module)


def _random_masks(count: int, features: int, hidden: int, k: int, rng) -> torch.Tensor:
    return random_exact_k(count, features, hidden, k, rng)


class MeasurementStore(_SharedMeasurementStore):
    """Legacy alias with the runner's fitter injection preserved for tests."""

    def __init__(self, out: Path, replay: RealReplay, device: str):
        super().__init__(out, replay, device, measure_fn=measure_mask)


def _dense_tune(out, tasks, mask, protocol, device, learning_rates, *,
                measurement_devices=None, measurement_batch_size=8):
    """Choose solver on meta-validation via the shared runtime selector."""
    measure_many_fn = None
    if measurement_devices:
        from .parallel_measurements import ParallelMeasurementStore
        def measure_many_fn(mask, validation, setting, device):
            replay = RealReplay(setting)
            with ParallelMeasurementStore(out / "dense_tuning" / "measurements", replay, device,
                    devices=measurement_devices, batch_size=measurement_batch_size) as tuning_store:
                return [result for _, result in tuning_store.measure_many(
                    [(mask, task, "dense_tuning") for task in validation], desc="Dense tuning")]
    return DenseProtocolSelector(out, measure_fn=measure_mask, save_torch_fn=save_torch,
                                 save_json_fn=save_json, measure_many_fn=measure_many_fn).select(
        tasks, mask, protocol, device, learning_rates, selection_label="meta-validation")


class LegacySearchController:
    """Owns immutable inputs and validation for the legacy single-policy search.

    The legacy/direct-control workflow remains separate from the cooperative
    controller; both expose a stable function wrapper for existing callers.
    """

    def __init__(self, bank, tasks, test_factory, out, protocol, config, *, device, resume,
                 dense_learning_rates):
        self.bank, self.tasks, self.test_factory = bank, tasks, test_factory
        self.out, self.protocol, self.config = Path(out).resolve(), protocol, config
        self.device, self.resume = device, resume
        self.dense_learning_rates = dense_learning_rates

    def validate_inputs(self):
        bank, tasks, config = self.bank, self.tasks, self.config
        if not 0 < config.k < bank.masks.shape[1] * bank.masks.shape[2]:
            raise ValueError("search requires a sparse exact-K budget")
        train_tasks = [task for task in tasks if task.split == "train"]
        validation_tasks = [task for task in tasks if task.split == "validation"]
        if not train_tasks or not validation_tasks or any(task.split == "test" for task in tasks):
            raise ValueError("initial tasks must contain train and meta-validation, without test")
        if len({task.task_id for task in tasks}) != len(tasks):
            raise ValueError("train/validation task IDs must be distinct")
        features, hidden = bank.masks.shape[1:]
        if bank.baseline_mask.shape != (features, hidden) or int(bank.baseline_mask.sum()) != config.k:
            raise ValueError("functional baseline must match the generated exact-K budget")
        return train_tasks, validation_tasks, features, hidden

    def source_files(self):
        project = Path(__file__).resolve().parents[1]
        files = list(Path(__file__).parent.glob("*.py")) + [
            project / "deepsets_vaae" / name for name in
            ("core.py", "utility_graph_child.py", "followup_batched_eval.py",
             "permutation_bank_encoder.py", "permutation_utility_loss.py")]
        files += [project / "pattern/task_quality/meta.py", project / "meta_pattern/data.py"]
        return project, files


def run_experiment(bank: FunctionalBank, tasks: list[TaskData],
                   test_factory: Callable[[], list[TaskData]], out: str | Path,
                   protocol: InnerProtocol, config: SearchConfig, *, device="cpu",
                   resume=False, dense_learning_rates=None,
                   measurement_devices=None, measurement_batch_size=8) -> dict:
    # Release GPU workers on success, a fitting failure, or user interruption.
    with ExitStack() as cleanup:
        return _run_experiment(bank, tasks, test_factory, out, protocol, config,
            device=device, resume=resume, dense_learning_rates=dense_learning_rates,
            measurement_devices=measurement_devices, measurement_batch_size=measurement_batch_size,
            measurement_cleanup=cleanup)


def _run_experiment(bank, tasks, test_factory, out, protocol, config, *, device,
                    resume, dense_learning_rates, measurement_devices,
                    measurement_batch_size, measurement_cleanup):
    controller = LegacySearchController(bank, tasks, test_factory, out, protocol, config,
                                        device=device, resume=resume,
                                        dense_learning_rates=dense_learning_rates)
    train_tasks, validation_tasks, features, hidden = controller.validate_inputs()
    out = controller.out
    out.mkdir(parents=True, exist_ok=True)
    requested_protocol = asdict(protocol)
    project, source_files = controller.source_files()
    source_hashes = {str(path.relative_to(project)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in source_files}
    run_spec = dict(config=asdict(config), requested_inner_protocol=requested_protocol,
                    source_hashes=source_hashes,
                    tasks={task.task_id: task.fingerprint for task in tasks},
                    bank_token_hash=tensor_hash(bank.tokens), bank_provenance=bank.provenance,
                    bank_mask_hash=tensor_hash(bank.masks), baseline_hash=tensor_hash(bank.baseline_mask),
                    quality_hash=None if bank.quality is None else tensor_hash(bank.quality),
                    dense_learning_rates=dense_learning_rates, device=str(device))
    if measurement_devices:
        run_spec["measurement_execution"] = dict(devices=list(measurement_devices),
                                                 batch_size=measurement_batch_size)
    session = RunSession(out, run_spec, source_files, project=project, resume=resume,
                         save_torch_fn=save_torch, save_json_fn=save_json)
    try:
        completed = session.prepare(inputs=dict(bank=bank, tasks=tasks))
    except ValueError as error:
        # Preserve the historical public error contract for legacy scripts.
        if "different code/inputs/settings" in str(error):
            raise ValueError("existing output uses different inputs/protocol; choose a new output directory") from error
        raise
    if completed is not None:
        return completed
    torch.manual_seed(config.seed)
    rng = torch.Generator(device=device).manual_seed(config.seed + 101)
    cpu_rng = torch.Generator().manual_seed(config.seed + 102)
    dense = torch.ones(features, hidden)
    if (out / "protocol.json").exists():
        protocol = InnerProtocol(**json.loads((out / "protocol.json").read_text())["inner_protocol"])
    else:
        rates = dense_learning_rates or [protocol.lr / 3, protocol.lr, protocol.lr * 3]
        protocol = _dense_tune(out, tasks, dense, protocol, device, rates,
                              measurement_devices=measurement_devices,
                              measurement_batch_size=measurement_batch_size)
        save_json(out / "protocol.json", dict(inner_protocol=asdict(protocol),
                  protocol_id=protocol.fingerprint, search=asdict(config),
                  epoch_definition=f"{config.updates_per_epoch} generator updates",
                  quality="mean terminal fresh-weight query error over independent initializations",
                  objective="worst task predicted error minus measured dense error",
                  smoke_only=config.smoke))
    replay = (RealReplay.load(out / "replay.pt") if (out / "replay.pt").exists()
              else RealReplay(protocol, split_seed=config.seed))
    if replay.protocol.fingerprint != protocol.fingerprint:
        raise ValueError("saved replay protocol does not match frozen solver")
    def make_store(folder, current_replay):
        if measurement_devices:
            from .parallel_measurements import ParallelMeasurementStore
            return measurement_cleanup.enter_context(ParallelMeasurementStore(
                folder, current_replay, device, devices=measurement_devices,
                batch_size=measurement_batch_size))
        return MeasurementStore(folder, current_replay, device)
    store = make_store(out, replay)

    def measure_entries(entries, desc):
        entries = list(entries)
        if measurement_devices:
            return store.measure_many(entries, desc=desc)
        return [store.measure(mask, task, origin)
                for mask, task, origin in progress(entries, desc=desc, unit="fit")]
    tokens = bank.tokens.to(device)
    quality = None if bank.quality is None else bank.quality.to(device)
    contexts = torch.stack([task.context for task in train_tasks]).to(device)
    generator = TransformerMaskGenerator(tokens.shape[-1], features, hidden, config.width,
                                        config.heads, config.layers, config.noise_dim).to(device)
    ensemble = QualityEnsemble(features, contexts.shape[1], num_members=config.ensemble_members,
                               width=config.width, heads=config.heads, layers=config.layers).to(device)
    optimizer = torch.optim.Adam(generator.parameters(), lr=config.generator_lr)
    initial_generator = _cpu_state(generator)
    dense_results = measure_entries([(dense, task, "dense") for task in tasks], "Dense controls")
    dense_rows = {task.task_id: row for task, (row, _) in zip(tasks, dense_results)}
    dense_quality = torch.tensor([dense_rows[t.task_id]["quality"] for t in train_tasks], device=device)
    history, evaluator_history, calibration = [], [], []
    best_mask, best_cost, best_epoch = bank.baseline_mask.clone(), float("inf"), 0
    best_generator, best_evaluator = initial_generator, _cpu_state(ensemble)
    start_epoch, cadence = 0, config.refresh_every
    checkpoint = out / "checkpoint.pt"
    if checkpoint.exists() and resume:
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        replay = saved["replay"]
        replay.validate()
        store.replay = replay
        generator.load_state_dict(saved["generator"])
        ensemble.load_state_dict(saved["ensemble"])
        if saved.get("evaluator_training_state") is not None:
            ensemble.training_state = saved["evaluator_training_state"]
        optimizer.load_state_dict(saved["optimizer"])
        history, evaluator_history, calibration = saved["history"], saved["evaluator_history"], saved["calibration"]
        best_mask, best_cost, best_epoch = saved["best_mask"].cpu(), saved["best_cost"], saved["best_epoch"]
        best_generator, best_evaluator = saved["best_generator"], saved["best_evaluator"]
        initial_generator = saved["initial_generator"]
        start_epoch, cadence = saved["epoch"], saved["cadence"]
        rng.set_state(saved["rng_state"].cpu())
        cpu_rng.set_state(saved["cpu_rng_state"].cpu())
        torch.set_rng_state(saved["torch_rng_state"].cpu())
        if str(device).startswith("cuda") and saved.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state(saved["cuda_rng_state"].cpu(), device)
    else:
        toeplitz = SlidingWindowMaskPrior().mask() if config.k == 32 else None
        initial = [] if toeplitz is None else [(toeplitz, "toeplitz")]
        initial += [(mask, "bank") for mask in bank.masks if int(mask.sum()) == config.k][:config.initial_random]
        initial.append((bank.baseline_mask, "functional"))
        initial.extend((mask, "random") for mask in _random_masks(config.initial_random, features, hidden, config.k, cpu_rng))
        # Reserve genuine topology holdouts even in tiny fixtures.
        for partition in ("train", "holdout"):
            for _ in range(1000):
                if sum(replay.mask_split(mask) == partition for mask, _ in initial) >= 2:
                    break
                candidate = _random_masks(1, features, hidden, config.k, cpu_rng)[0]
                if replay.mask_split(candidate) == partition:
                    initial.append((candidate, "random_partition_reserve"))
            else:
                raise RuntimeError("unable to create distinct topology train/holdout samples")
        seen = set()
        measurements = []
        for mask, origin in initial:
            identity = topology_id(mask)
            if identity in seen:
                continue
            seen.add(identity)
            for task in tasks:
                measurements.append((mask, origin, task))
        measure_entries([(mask, task, origin) for mask, origin, task in measurements],
                        "Initial real labels")
        replay.save(out / "replay.pt")
        evaluator_history.append(train_evaluators(ensemble, replay, epochs=config.evaluator_epochs,
             batch_size=config.evaluator_batch_size, lr=config.evaluator_lr, seed=config.seed, device=device))

    def validation_cost(mask, origin):
        results = measure_entries([(mask, task, origin) for task in validation_tasks],
                                  "Validate mask")
        deltas = [row["quality"] - dense_rows[task.task_id]["quality"]
                  for task, (row, _) in zip(validation_tasks, results)]
        return max(deltas)

    def save_checkpoint(epoch):
        # One atomic payload is the source of truth on resume. replay.pt is
        # an inspectable export, and may be newer if an export was interrupted.
        session.save_checkpoint(dict(generator=_cpu_state(generator), ensemble=_cpu_state(ensemble),
                   evaluator_training_state=getattr(ensemble, "training_state", None), replay=replay,
                   optimizer=optimizer.state_dict(), initial_generator=initial_generator,
                   epoch=epoch, cadence=cadence, history=history, evaluator_history=evaluator_history,
                   calibration=calibration, best_mask=best_mask, best_cost=best_cost, best_epoch=best_epoch,
                   best_generator=best_generator, best_evaluator=best_evaluator,
                   rng_state=rng.get_state(), cpu_rng_state=cpu_rng.get_state(),
                   torch_rng_state=torch.get_rng_state(),
                   cuda_rng_state=torch.cuda.get_rng_state(device) if str(device).startswith("cuda") else None),
                   name=checkpoint.name)

    if not checkpoint.exists():
        save_checkpoint(0)

    for epoch in range(start_epoch + 1, config.generator_epochs + 1):
        updates_bar = progress(range(config.updates_per_epoch),
                               desc=f"Generator epoch {epoch}/{config.generator_epochs}", unit="update")
        for update in updates_bar:
            logs = generator_update(generator, ensemble, tokens, quality, contexts, dense_quality,
                                    optimizer, config.k, rng,
                                    permutation_weight=config.permutation_weight,
                                    uncertainty_weight=config.uncertainty_weight)
            history.append(dict(epoch=epoch, update=update, **logs))
            updates_bar.set_postfix(predicted_delta=f"{logs['predicted_cost']:.5f}", refresh=False)
        if epoch % cadence and epoch != config.generator_epochs:
            continue
        proposals = propose_candidates(generator, tokens, quality, config.k, config.candidates, rng).cpu()
        fresh_random = _random_masks(max(config.acquisition_budget, config.candidates // 3), features, hidden, config.k, cpu_rng)
        pool = torch.cat((proposals, fresh_random)).to(device)
        measured_topologies = {row["topology_id"] for row in replay.records if row["task_split"] == "train"}
        pool = torch.stack([mask for mask in pool if topology_id(mask) not in measured_topologies])
        acquired, origins = select_acquisition(pool, ensemble, contexts, dense_quality,
                                               config.acquisition_budget, rng)
        gaps = []
        # Fit the whole acquisition together before inspecting cached per-mask results.
        measure_entries([(mask, task, f"acquisition_{origin}")
                         for mask, origin in zip(acquired.cpu(), origins) for task in train_tasks],
                        "Acquire real labels")
        for mask, origin in progress(list(zip(acquired.cpu(), origins)), desc="Acquire real labels", unit="mask"):
            for task in train_tasks:
                with torch.no_grad():
                    predicted, std = ensemble.predict(mask[None].to(device), task.context[None].to(device))
                row, _ = store.measure(mask, task, f"acquisition_{origin}")
                gap = abs(float(predicted) - row["quality"])
                gaps.append(gap)
                calibration.append(dict(epoch=epoch, task_id=task.task_id, topology_id=row["topology_id"],
                                        predicted=float(predicted), std=float(std), actual=row["quality"], absolute_gap=gap))
        validation_pool = [best_mask, bank.baseline_mask, *list(acquired.cpu())]
        measure_entries([(mask, task, "selection_meta_validation")
                         for mask in validation_pool for task in validation_tasks],
                        "Selection real labels")
        for mask in progress(validation_pool, desc="Select on validation", unit="mask"):
            cost = validation_cost(mask, "selection_meta_validation")
            if cost < best_cost:
                best_cost, best_mask, best_epoch = cost, mask.clone(), epoch
                best_generator, best_evaluator = _cpu_state(generator), _cpu_state(ensemble)
        mean_gap = sum(gaps) / max(len(gaps), 1)
        scale = max(float(dense_quality.abs().mean()), 1e-8)
        previous_cadence = cadence
        if mean_gap / scale > config.gap_threshold:
            cadence = max(1, cadence // 2)
        evaluator_history.append(train_evaluators(ensemble, replay, epochs=config.evaluator_epochs,
             batch_size=config.evaluator_batch_size, lr=config.evaluator_lr, seed=config.seed + epoch, device=device))
        replay.save(out / "replay.pt")
        save_checkpoint(epoch)
        save_json(out / "progress.json", dict(epoch=epoch, selected_meta_validation_delta=best_cost,
                  real_label_count=len(replay.records), acquisition_mean_absolute_gap=mean_gap,
                  previous_refresh_every=previous_cadence, refresh_every=cadence))
        print(f"epoch={epoch}, measured_labels={len(replay.records)}, validation_delta={best_cost:.6g}, refresh={cadence}", flush=True)

    direct_mask, direct_history = None, []
    # Both searches get the same count of real mask-task observations. Dense
    # anchors/tuning and final validation are itemized separately from that budget.
    surrogate_budget = sum(row["task_split"] == "train" and row["split"] != "control" for row in replay.records)
    direct_budget = 0
    if config.direct_control:
        direct = copy.deepcopy(generator)
        direct.load_state_dict(initial_generator)
        direct_optimizer = torch.optim.Adam(direct.parameters(), lr=config.generator_lr)
        direct_rng = torch.Generator(device=device).manual_seed(config.seed + 201)
        direct_replay = RealReplay(protocol, split_seed=config.seed)
        direct_store = make_store(out / "direct", direct_replay)
        def measure_cost(masks):
            nonlocal direct_budget
            costs = []
            for mask in masks.cpu():
                deltas = []
                for task in train_tasks:
                    before = len(direct_replay.records)
                    row, _ = direct_store.measure(mask, task, "direct_policy", fresh=True)
                    deltas.append(row["quality"] - dense_rows[task.task_id]["quality"])
                    direct_budget += len(direct_replay.records) - before
                costs.append(max(deltas))
            return torch.tensor(costs, device=device)
        direct_candidates = [bank.baseline_mask]
        direct_bar = progress(total=surrogate_budget, desc="Direct quality control", unit="fit")
        while surrogate_budget - direct_budget >= 2 * len(train_tasks):
            logs = direct_generator_update(direct, tokens, quality, direct_optimizer, config.k,
                                            direct_rng, measure_cost,
                                            permutation_weight=config.permutation_weight)
            direct_history.append(logs)
            direct_bar.update(direct_budget-direct_bar.n)
            if len(direct_history) > 5 * surrogate_budget:
                raise RuntimeError("direct policy repeatedly proposes already measured topologies")
        # Spend any residual budget on real task measurements (not pseudo-labels).
        residual_masks = propose_candidates(direct, tokens, quality, config.k, config.acquisition_budget, direct_rng).cpu()
        direct_candidates.extend(list(residual_masks))
        index = 0
        while direct_budget < surrogate_budget:
            task = train_tasks[index % len(train_tasks)]
            mask = propose_candidates(direct, tokens, quality, config.k, 1, direct_rng).cpu()[0]
            before = len(direct_replay.records)
            direct_store.measure(mask, task, "direct_budget_remainder", fresh=True)
            direct_budget += len(direct_replay.records) - before
            index += 1
            direct_bar.update(direct_budget-direct_bar.n)
            if index > 5 * surrogate_budget:
                raise RuntimeError("unable to fill direct control with fresh measurements")
        direct_bar.close()
        direct_mask = min(direct_candidates, key=lambda mask: validation_cost(mask, "direct_selection_meta_validation"))
        direct_replay.save(out / "direct" / "replay.pt")
        save_torch(out / "direct" / "checkpoint.pt", dict(generator=_cpu_state(direct),
                   optimizer=direct_optimizer.state_dict(), rng_state=direct_rng.get_state(),
                   selected_mask=direct_mask, history=direct_history, measurement_budget=direct_budget))
        if measurement_devices:
            direct_store.close()

    random_mask = _random_masks(1, features, hidden, config.k, cpu_rng)[0]
    masks = dict(generator=best_mask, functional=bank.baseline_mask, random=random_mask, dense=dense)
    toeplitz = SlidingWindowMaskPrior().mask() if config.k == 32 else None
    if toeplitz is not None:
        masks["toeplitz"] = toeplitz
    if direct_mask is not None:
        masks["direct_generator"] = direct_mask
    # Save the frozen choice before touching any final test task labels.
    frozen = dict(masks=masks, generator=best_generator, evaluator=best_evaluator,
                  inner_protocol=asdict(protocol), protocol_id=protocol.fingerprint,
                  selected_epoch=best_epoch, selected_meta_validation_delta=best_cost,
                  selection_split="meta-validation", test_used=False)
    if (out / "frozen.pt").exists():
        previous = torch.load(out / "frozen.pt", map_location="cpu", weights_only=False)
        if (previous["protocol_id"] != frozen["protocol_id"] or
            set(previous["masks"]) != set(masks) or
            any(not torch.equal(previous["masks"][name], mask) for name, mask in masks.items())):
            raise ValueError("existing frozen masks cannot change after test has been opened")
    else:
        save_torch(out / "frozen.pt", frozen)
    save_json(out / "frozen.json", {key: value for key, value in frozen.items() if key not in ("masks", "generator", "evaluator")})
    test_tasks = test_factory()
    if not test_tasks or any(task.split != "test" for task in test_tasks):
        raise ValueError("final evaluation requires held-out test tasks")
    if set(task.task_id for task in test_tasks) & set(task.task_id for task in tasks):
        raise ValueError("final test tasks overlap train/validation tasks")
    save_torch(out / "test_tasks.pt", test_tasks)
    measure_entries([(mask, task, f"final_{method}")
                     for task in test_tasks for method, mask in masks.items()], "Frozen final labels")
    comparisons, examples = {}, []
    for task in progress(test_tasks, desc="Frozen final test", unit="task"):
        dense_row, _ = store.measure(dense, task, "final_dense")
        task_results = {}
        for method, mask in masks.items():
            row, result = store.measure(mask, task, f"final_{method}")
            task_results[method] = dict(query_error=row["quality"], plateau_flags=row["plateau_flags"],
                **paired_comparison(row["replica_losses"], dense_row["replica_losses"]))
            if task is test_tasks[0]:
                examples.append(dict(method=method, mask=mask, effective_weights=result["effective_weights"],
                                     history=result["history"]))
        comparisons[task.task_id] = task_results
    replay.save(out / "replay.pt")
    save_json(out / "history.json", dict(generator=history, evaluator=evaluator_history,
              calibration=calibration, direct_generator=direct_history))
    summary = dict(smoke_only=config.smoke, protocol_id=protocol.fingerprint,
                   selected_epoch=best_epoch, selected_meta_validation_delta=best_cost,
                   test=comparisons, real_labels=len(replay.records),
                   surrogate_training_mask_task_budget=surrogate_budget,
                   direct_training_mask_task_budget=direct_budget,
                   budgets_match=not config.direct_control or direct_budget == surrogate_budget,
                   improves_every_test_task=all(row["generator"]["point_improvement"] for row in comparisons.values()),
                   interval_improvement_every_test_task=all(row["generator"]["interval_below_zero"] for row in comparisons.values()),
                   uncertainty_limitation="initialization intervals conditional on fixed task/data; not population/task guarantees",
                   topology_partition="hidden-column canonical hashes; holdout labels never update evaluator",
                   ensemble_uncertainty="diagnostic; inspect calibration against real measurements")
    if toeplitz is not None:
        summary["structural_prior"] = SlidingWindowMaskPrior().diagnostics(best_mask)
    tuning_budget = json.loads((out / "dense_tuning.json").read_text())["label_measurements"]
    summary["measurement_accounting"] = dict(
        primary_search_train=surrogate_budget, direct_search_train=direct_budget,
        dense_tuning=tuning_budget,
        primary_controls=sum(row["split"] == "control" for row in replay.records),
        primary_train_topology_holdout=sum(row["split"] == "mask_validation" for row in replay.records),
        meta_validation_and_selection=sum(row["task_split"] == "validation" and row["split"] != "control" for row in replay.records),
        final_test=sum(row["task_split"] == "test" and row["split"] != "control" for row in replay.records),
        total_real_mask_task_measurements=len(replay.records)+direct_budget+tuning_budget,
        total_new_child_fits=(len(replay.records)+direct_budget+tuning_budget)*protocol.replicas)
    save_json(out / "summary.json", summary)
    write_plots(out, history, examples, evaluator_history=evaluator_history, calibration=calibration)
    (out / "COMPLETE").write_text("Complete frozen-protocol run; consult summary.json for scientific conclusions.\n")
    if measurement_devices:
        store.close()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=("pattern", "deepsets"), default="pattern")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--measurement-devices", nargs="+",
                        help="child-fit devices, e.g. cuda:0 cuda:1; auto uses all visible GPUs")
    parser.add_argument("--measurement-batch-size", type=int, default=8,
                        help="candidate masks trained together per child-fit worker")
    parser.add_argument("--seed", type=int, default=4100)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--replicas", type=int, default=2)
    parser.add_argument("--lr", type=float, default=.01)
    parser.add_argument("--l2", type=float, default=0.)
    parser.add_argument("--dense-learning-rates", type=float, nargs="+")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--updates-per-epoch", type=int)
    parser.add_argument("--refresh-every", type=int)
    parser.add_argument("--evaluator-epochs", type=int)
    parser.add_argument("--acquisition-budget", type=int)
    for name in ("width", "heads", "layers", "noise-dim", "ensemble-members",
                 "candidates", "initial-random", "evaluator-batch-size"):
        parser.add_argument(f"--{name}", type=int)
    parser.add_argument("--support-count", type=int)
    parser.add_argument("--query-count", type=int)
    parser.add_argument("--teacher-count", type=int)
    parser.add_argument("--bank-steps", type=int)
    parser.add_argument("--bank-context", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("datasets/mnist8m"))
    parser.add_argument("--skip-direct-control", action="store_true")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--progress", action="store_true", help="show progress bars even when stderr is not a TTY")
    parser.add_argument("--no-progress", action="store_true", help="disable progress bars")
    args = parser.parse_args()
    if args.progress and args.no_progress:
        parser.error("--progress and --no-progress are mutually exclusive")
    if args.progress:
        os.environ["GENERATOR_EVALUATOR_PROGRESS"] = "1"
    if args.no_progress:
        os.environ["GENERATOR_EVALUATOR_PROGRESS"] = "0"
    torch.set_num_threads(args.threads)
    if args.measurement_batch_size < 1:
        parser.error("--measurement-batch-size must be positive")
    measurement_devices = args.measurement_devices
    if measurement_devices == ["auto"]:
        measurement_devices = [f"cuda:{index}" for index in range(torch.cuda.device_count())]
        if not measurement_devices:
            parser.error("--measurement-devices auto requires visible CUDA GPUs")
    if measurement_devices:
        if args.domain != "deepsets":
            parser.error("batched multi-device child fits currently require --domain deepsets")
        for child_device in measurement_devices:
            try:
                parsed_device = torch.device(child_device)
            except (ValueError, RuntimeError):
                parser.error(f"invalid measurement device: {child_device}")
            if parsed_device.type not in ("cpu", "cuda"):
                parser.error("measurement devices must be cpu or cuda")
            if parsed_device.type == "cuda" and (not torch.cuda.is_available() or
                    (parsed_device.index or 0) >= torch.cuda.device_count()):
                parser.error(f"measurement device is unavailable: {child_device}")
        print(f"Child-fit devices: {', '.join(measurement_devices)}; "
              f"batch={args.measurement_batch_size} masks × {args.replicas} replicas", flush=True)
    k = 32 if args.domain == "pattern" else 7526
    config = SearchConfig(seed=args.seed, k=k, smoke=args.smoke, direct_control=not args.skip_direct_control)
    if args.smoke:
        config = replace(config, generator_epochs=2, updates_per_epoch=2, refresh_every=1,
                         acquisition_budget=3, candidates=12, initial_random=4,
                         evaluator_epochs=3, width=16, layers=1, noise_dim=4, ensemble_members=2)
    for arg_name, config_name in (("epochs", "generator_epochs"), ("updates_per_epoch", "updates_per_epoch"),
                                  ("refresh_every", "refresh_every"), ("evaluator_epochs", "evaluator_epochs"),
                                  ("acquisition_budget", "acquisition_budget"),
                                  ("width", "width"), ("heads", "heads"), ("layers", "layers"),
                                  ("noise_dim", "noise_dim"), ("ensemble_members", "ensemble_members"),
                                  ("candidates", "candidates"), ("initial_random", "initial_random"),
                                  ("evaluator_batch_size", "evaluator_batch_size")):
        value = getattr(args, arg_name)
        if value is not None:
            config = replace(config, **{config_name: value})
    protocol = InnerProtocol(steps=args.steps or (8 if args.smoke else 2000), replicas=args.replicas,
                             lr=args.lr, l2=args.l2, seed=args.seed,
                             metric="bce" if args.domain == "pattern" else "nmse")
    inputs_path = args.out / "inputs.pt"
    spec_path = args.out / "test_spec.pt"
    if args.resume and inputs_path.exists() and spec_path.exists():
        inputs = torch.load(inputs_path, map_location="cpu", weights_only=False)
        bank, tasks = inputs["bank"], inputs["tasks"]
        test_spec = torch.load(spec_path, map_location="cpu", weights_only=False)
    elif args.domain == "pattern":
        bank, tasks, test_spec = build_pattern_fixture(seed=args.seed,
            bank_steps=args.bank_steps or (8 if args.smoke else 2000),
            teacher_count=args.teacher_count or (8 if args.smoke else 32),
            support_count=args.support_count or (32 if args.smoke else 128),
            query_count=args.query_count or (32 if args.smoke else 256), k=k, device=args.device)
    else:
        context = args.bank_context or Path(f"outputs/deepsets_vaae/20261001_rebuilt_functional_bank/seed_{args.seed}/functional_context.pt")
        bank = load_deepsets_bank(context, k=k, teacher_count=args.teacher_count or 32, seed=args.seed)
        tasks, test_spec = make_deepsets_tasks(args.data_root, seed=args.seed,
            support_count=args.support_count or (8 if args.smoke else 205),
            query_count=args.query_count or (8 if args.smoke else 51))
    materialize = make_pattern_test_tasks if args.domain == "pattern" else materialize_deepsets_test_tasks
    if args.smoke:
        tasks = ([task for task in tasks if task.split == "train"][:2]
                 + [task for task in tasks if task.split == "validation"][:1])
    save_torch(spec_path, test_spec)
    def test_factory():
        result = materialize(test_spec)
        return result[:1] if args.smoke else result
    summary = run_experiment(bank, tasks, test_factory, args.out, protocol, config,
                             device=args.device, resume=args.resume,
                             measurement_devices=measurement_devices,
                             measurement_batch_size=args.measurement_batch_size,
                             dense_learning_rates=args.dense_learning_rates or ([args.lr] if args.smoke else None))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
