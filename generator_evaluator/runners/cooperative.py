"""Cooperative generators, a shared critic, and real feedback.

No training runs on import. Ordered held-out pattern or DeepSets roles remain
sealed until generator training has completed and an immutable selected
artifact has been written. Joint training is the default; the historical
quality-then-cooperation schedule remains available with ``--training-mode staged``.
Checkpoints retain stage, live banks, optimizers, replay, and per-device random state.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import ExitStack
from collections.abc import Mapping
from dataclasses import asdict, replace
import hashlib
import io
import json
import os
from pathlib import Path
import shutil

import torch

from generator_evaluator.storage.artifacts import paired_comparison, save_json, save_torch, write_plots
from generator_evaluator.data.pattern import (append_feedback, bank_input_fingerprint,
                               build_cooperative_fixture, make_cooperative_test_tasks)
from generator_evaluator.search.policy import (DensityConditionedGenerator,
                                 propose_at_budget, propose_shared_pool,
                                 rank_shared_pool, select_common_elites)
from generator_evaluator.data.types import InnerProtocol, RealReplay, TaskData, support_context, tensor_hash, topology_id
from generator_evaluator.models.transformer import QualityEnsemble
from generator_evaluator.training.objectives import reconstruct_bank_masks
from generator_evaluator.training.joint import joint_generator_update
from generator_evaluator.storage.functional import write_functional_card
from generator_evaluator.search.quality import QUALITY_OBJECTIVES, quality_objective_cost
from generator_evaluator.search.priors import SlidingWindowMaskPrior
from generator_evaluator.storage.progress import progress
from generator_evaluator.runners.legacy import MeasurementStore, _cpu_state, _dense_tune, _random_masks
from generator_evaluator.storage.runtime import RunSession
from generator_evaluator.training.updates import train_evaluators
from generator_evaluator.storage.toeplitz import write_toeplitz_report
from generator_evaluator.storage.warm_start import evaluator_bank_fingerprint, load_cooperative_warm_start
from generator_evaluator.search.schedule import (STAGE_BOOTSTRAP_QUALITY, STAGE_COOPERATION, STAGE_JOINT,
                            STAGE_QUALITY, STAGE_TRAINING_COMPLETE,
                            joint_search_stages, search_stages)


from generator_evaluator.config import CooperativeConfig, deepsets_config, pattern_small_config, _config_metadata


def _real_candidates(replay, tasks, k=None):
    """Complete, actual all-task labels from topology TRAIN only.

    Hidden-column permutations share a topology identity. Keep one fully
    measured coordinate representative for each topology so archive entries
    are canonical unique proposals.
    """
    task_ids = [task.task_id for task in tasks]
    grouped = {}
    for row in replay.records:
        if row["split"] == "train" and row["task_id"] in task_ids:
            if k is None or row["active_edges"] == k:
                grouped.setdefault(row["topology_id"], {}).setdefault(row["mask_key"], {})[
                    row["task_id"]] = row
    complete = []
    for identity, variants in grouped.items():
        keys = sorted(key for key, rows in variants.items()
                      if all(task_id in rows for task_id in task_ids))
        if keys:
            complete.append((identity, keys[0], variants[keys[0]]))
    if not complete:
        shape = next(iter(replay.masks.values())).shape if replay.masks else (11, 8)
        return torch.empty(0, *shape), torch.empty(0, len(tasks)), []
    return (torch.stack([replay.masks[key] for _, key, _ in complete]),
            torch.tensor([[rows[task_id]["quality"] for task_id in task_ids]
                          for _, _, rows in complete]),
            [[rows[task_id] for task_id in task_ids] for _, _, rows in complete])


_EVALUATOR_POLICY = "initial_bank_only"


def _bank_topology_ids(banks):
    return tuple(sorted({topology_id(mask) for bank in banks.values() for mask in bank.masks}))


def _evaluator_row_id(row):
    fields = ("topology_id", "mask_key", "task_id", "task_split", "split",
              "protocol_id", "task_fingerprint", "label_source", "quality",
              "replica_losses", "seeds", "active_edges", "density")
    return {field: row[field] for field in fields}


def _evaluator_bank_snapshot(replay, topology_ids, task_ids, row_ids=None):
    """Create the immutable, bank-membership-filtered critic dataset view."""
    topology_ids = tuple(sorted(set(topology_ids)))
    task_ids = set(task_ids)
    eligible = [row for row in replay.records
                if row["topology_id"] in topology_ids and row["task_id"] in task_ids
                and row["task_split"] == "train"]
    if row_ids is not None:
        requested = list(row_ids)
        def row_key(row):
            return json.dumps(_evaluator_row_id(row), sort_keys=True, separators=(",", ":"))
        by_id = {}
        for row in eligible:
            key = row_key(row)
            by_id.setdefault(key, []).append(row)
        rows = []
        for descriptor in requested:
            if isinstance(descriptor, Mapping):
                key = row_key(descriptor)
            else:
                key = json.dumps(descriptor, sort_keys=True, separators=(",", ":"))
            matches = by_id.get(key, [])
            if not matches:
                raise ValueError("fixed evaluator bank rows are missing from replay")
            rows.append(matches.pop(0))
    else:
        rows = eligible
    rows.sort(key=lambda row: (row["task_id"], row["split"], row["topology_id"], row["mask_key"]))
    train_rows = [row for row in rows if row["split"] == "train"]
    if not train_rows:
        raise ValueError("initial functional bank has no real TRAIN rows for evaluator fitting")
    row_manifest = [_evaluator_row_id(row) for row in rows]
    fingerprint = evaluator_bank_fingerprint(topology_ids, row_manifest)
    replay_view = copy.copy(replay)
    replay_view.records = rows
    manifest = dict(
        evaluator_policy=_EVALUATOR_POLICY,
        fingerprint=fingerprint,
        topology_ids=list(topology_ids),
        row_ids=[_evaluator_row_id(row) for row in rows],
        rows=row_manifest,
        train_masks=torch.stack([replay.masks[row["mask_key"]] for row in train_rows]),
        train_contexts=torch.stack([replay.contexts[row["task_id"]] for row in train_rows]),
        train_targets=torch.tensor([row["quality"] for row in train_rows], dtype=torch.float32),
    )
    return replay_view, manifest


class CooperativeSearchController:
    """Mutable search state for the cooperative experiment.

    The runner deliberately keeps lifecycle concerns here instead of passing a
    growing collection of closures through the epoch loop.  This makes a
    future policy, scorer, or feedback strategy an isolated substitution.
    """

    def __init__(self, *, out, config, device, patterns, banks, train_tasks,
                 selection_tasks, replay, store, models, optimizers, ensemble,
                 dense_rows, dense_quality, contexts, rng, cpu_rng, own_rngs, checkpoint,
                 measure_many, session, trainer, generator_devices,
                 generator_pretraining=None, initial_bank_topology_ids=()):
        self.out, self.config, self.device, self.patterns = out, config, device, patterns
        self.banks, self.train_tasks, self.selection_tasks = banks, train_tasks, selection_tasks
        self.replay, self.store = replay, store
        self.models, self.optimizers, self.ensemble = models, optimizers, ensemble
        self.dense_rows, self.dense_quality, self.contexts = dense_rows, dense_quality, contexts
        self.rng, self.cpu_rng, self.own_rngs, self.checkpoint = rng, cpu_rng, own_rngs, checkpoint
        self.measure_many, self.session = measure_many, session
        self.trainer, self.generator_devices = trainer, tuple(generator_devices)
        self.initial_shared_latent = trainer.shared_latent.detach().cpu().clone()
        self.history, self.evaluator_history, self.calibration, self.refresh_history = [], [], [], []
        self.pretraining_history = []
        self.pretraining_epoch = 0
        self.generator_pretraining = copy.deepcopy(generator_pretraining)
        self.initialization_done = False
        self.stage, self.stage_epoch = None, 0
        self.global_epoch = 0
        self.actual_archive = []
        self._training_views = {}
        self.stage_last_refresh = {}
        self.best_mask = banks[patterns[0]].baseline_mask.clone()
        self.best_cost, self.best_epoch = float("inf"), 0
        self.best_mean_delta = self.best_worst_delta = float("inf")
        self.best_stage = "initial"
        self.best_models, self.best_evaluator, self.best_banks = {}, {}, {}
        self.last_refresh, self.cadence = 0, config.refresh_every
        self.generator_executor = None
        self.evaluator_policy = _EVALUATOR_POLICY
        self.evaluator_bank_topology_ids = tuple(sorted(set(initial_bank_topology_ids)))
        self.evaluator_bank_rows = []
        self.evaluator_bank_fingerprint = None

    def _adapt_refresh_cadence(self, relative_gap):
        """Reduce refresh frequency when critic calibration is poor, within its configured floor."""
        previous = self.cadence
        if relative_gap > self.config.gap_threshold:
            self.cadence = max(self.config.minimum_refresh_every, self.cadence // 2)
        return previous

    def restore(self, saved):
        version = saved.get("algorithm_version")
        saved_mode = saved.get("training_mode")
        if version != 7 or saved_mode != self.config.training_mode:
            raise ValueError("checkpoint predates the frozen initial-bank evaluator policy; "
                             "use --warm-start-from to initialize a fresh critic")
        if "trainer_state" not in saved:
            raise ValueError("checkpoint lacks resumable trainer state; use --warm-start-from")
        if tuple(saved.get("generator_devices", ())) != self.generator_devices:
            raise ValueError("checkpoint generator devices differ from the requested run")
        self.replay, self.banks = saved["replay"], saved["banks"]
        self.replay.validate()
        self.store.replay = self.replay
        for name in self.patterns:
            self.models[name].load_state_dict(saved["models"][name])
            self.models[name].set_budget(saved["budgets"][name])
            self.optimizers[name].load_state_dict(saved["optimizers"][name])
            self.own_rngs[name].set_state(saved["own_rngs"][name])
            self.trainer.set_bank(name, self.banks[name])
            self.trainer.invalidate_bank_cache(name)
        self.trainer.load_state_dict(saved["trainer_state"])
        self.ensemble.load_state_dict(saved["ensemble"])
        self.ensemble.training_state = saved["evaluator_training_state"]
        for name in ("history", "evaluator_history", "calibration", "refresh_history",
                     "best_mask", "best_cost", "best_epoch", "best_models", "best_evaluator",
                     "best_banks", "best_stage", "last_refresh", "cadence", "actual_archive",
                     "stage", "stage_epoch", "stage_last_refresh", "initial_shared_latent"):
            setattr(self, name, saved[name])
        self.evaluator_policy = saved["evaluator_policy"]
        if self.evaluator_policy != _EVALUATOR_POLICY:
            raise ValueError("checkpoint evaluator policy differs from this run")
        self.evaluator_bank_topology_ids = tuple(saved["evaluator_bank_topology_ids"])
        self.evaluator_bank_rows = list(saved["evaluator_bank_rows"])
        self.evaluator_bank_fingerprint = saved["evaluator_bank_fingerprint"]
        if not (self.out / "evaluator_bank.pt").is_file():
            raise ValueError("checkpoint lacks its immutable evaluator_bank.pt provenance")
        self.fit_evaluator(self.evaluator_bank_topology_ids,
                           row_ids=self.evaluator_bank_rows,
                           expected_fingerprint=self.evaluator_bank_fingerprint,
                           train=False)
        self.cadence = max(self.config.minimum_refresh_every, self.cadence)
        self.pretraining_history = saved["pretraining_history"]
        self.pretraining_epoch = saved["pretraining_epoch"]
        self.generator_pretraining = copy.deepcopy(saved.get("generator_pretraining"))
        self.best_mean_delta = saved.get("best_mean_delta", saved["best_cost"])
        self.best_worst_delta = saved.get("best_worst_delta", saved["best_cost"])
        self.initialization_done = saved["initialization_done"]
        self.global_epoch = saved["epoch"]
        self.rng.set_state(saved["rng_state"])
        self.cpu_rng.set_state(saved["cpu_rng_state"])
        torch.set_rng_state(saved["torch_rng_state"])
        for device, state in saved.get("cuda_rng_states", {}).items():
            torch.cuda.set_rng_state(state, torch.device(device))
        return saved["epoch"]

    def fit_evaluator(self, topology_ids, *, row_ids=None, expected_fingerprint=None,
                       train=True, seed=None):
        """Fit once from the immutable initial functional-bank membership."""
        replay_view, manifest = _evaluator_bank_snapshot(
            self.replay, topology_ids, [task.task_id for task in self.train_tasks], row_ids)
        if expected_fingerprint is not None and manifest["fingerprint"] != expected_fingerprint:
            raise ValueError("warm-start evaluator bank fingerprint differs from its saved critic")
        if train:
            self.evaluator_history.append(train_evaluators(
                self.ensemble, replay_view, epochs=self.config.evaluator_epochs,
                batch_size=self.config.evaluator_batch_size, lr=self.config.evaluator_lr,
                seed=self.config.seed if seed is None else seed, device=self.device,
                restore_best=self.config.domain == "deepsets",
                selection_active_edges=self.config.k))
        self.evaluator_policy = _EVALUATOR_POLICY
        self.evaluator_bank_topology_ids = tuple(manifest["topology_ids"])
        self.evaluator_bank_rows = list(manifest["row_ids"])
        self.evaluator_bank_fingerprint = manifest["fingerprint"]
        bank_path = self.out / "evaluator_bank.pt"
        if bank_path.exists():
            existing = torch.load(bank_path, map_location="cpu", weights_only=False)
            if (existing.get("fingerprint") != manifest["fingerprint"] or
                    existing.get("topology_ids") != manifest["topology_ids"]):
                raise ValueError("immutable evaluator bank artifact differs from the fixed policy")
        else:
            save_torch(bank_path, manifest)
        self.freeze_evaluator()

    def freeze_evaluator(self):
        self.ensemble.eval()
        self.ensemble.zero_grad(set_to_none=True)
        for parameter in self.ensemble.parameters():
            parameter.requires_grad_(False)

    def ensure_initial_bank_labels(self, source_masks=None):
        """Fill missing cross-task labels for every starting functional topology."""
        candidates = {}
        if source_masks is None:
            source = ((name, mask) for name, bank in self.banks.items()
                      for mask in bank.masks)
        else:
            source = (("source_input", mask) for mask in source_masks)
        for name, mask in source:
            identity = topology_id(mask)
            if identity in self.evaluator_bank_topology_ids:
                candidates.setdefault(identity, (name, mask))
        if set(candidates) != set(self.evaluator_bank_topology_ids):
            raise ValueError("original functional bank masks do not match the fixed evaluator topology set")
        known = {(row["topology_id"], row["task_id"]) for row in self.replay.records
                 if row.get("task_split") == "train"}
        entries = [(mask, task, f"bank:{name}")
                   for identity, (name, mask) in candidates.items()
                   for task in self.train_tasks if (identity, task.task_id) not in known]
        if entries:
            self.measure_many(entries)

    def common_elites(self):
        masks, real, _ = _real_candidates(self.replay, self.train_tasks, self.config.k)
        return select_common_elites(masks, real, self.dense_quality,
                                    margin=-self.config.elite_margin, limit=self.config.elite_limit,
                                    quality_objective=self.config.quality_objective)

    def distillation_targets(self):
        """Return the bounded archive of actually measured TRAIN targets."""
        if not self.actual_archive:
            return torch.empty(0, self.config.features, self.config.hidden)
        return torch.stack([row["mask"] for row in self.actual_archive])

    def export_functional_cards(self):
        """Persist one compact card per teacher, including measured feedback."""
        for name, bank in self.banks.items():
            for row, teacher in enumerate(bank.states):
                identity = teacher.get("row_hash") or tensor_hash(bank.tokens[0, row])
                source_card = teacher.get("source", {}).get("artifact_path")
                if source_card:
                    source_path = Path(source_card)
                    if (source_path.parent.name == "maps" and source_path.is_file()
                            and source_path.resolve().is_relative_to(self.out.resolve())):
                        teacher["source"]["functional_artifact_path"] = str(source_path.resolve())
                        continue
                path = self.out / "functional_maps" / name / f"{identity}.pt"
                if not path.exists():
                    metadata = {key: value for key, value in teacher.get("source", {}).items()
                                if key not in ("source_mask", "functional_artifact_path")}
                    write_functional_card(path, token=bank.tokens[0, row], mask=bank.masks[row],
                        state={key: value for key, value in teacher["state_dict"].items() if key != "masks"},
                        metadata=dict(metadata, bank_role=name,
                        row_hash=identity, probe_fingerprint=bank.provenance.get("probe_fingerprint")))
                teacher.setdefault("source", {})["functional_artifact_path"] = str(path.resolve())

    def pretrain_generators(self):
        """Run the one-time reconstruction phase for fresh main generators."""
        from generator_evaluator.training.device_executor import PerDeviceGeneratorExecutor
        epochs = progress(range(self.pretraining_epoch, self.config.generator_pretrain_epochs),
                          desc="Main generator reconstruction", unit="epoch")
        with PerDeviceGeneratorExecutor(self.patterns, self.generator_devices) as executor:
            for epoch in epochs:
                for update in range(self.config.pretrain_updates_per_epoch):
                    def reconstruct(name):
                        return reconstruct_bank_masks(self.models[name], self.banks[name],
                            self.optimizers[name], rng=self.own_rngs[name],
                            batch_size=self.config.reconstruction_batch_size,
                            weight=self.config.reconstruction_weight,
                            bank_consensus=(self.config.training_mode == "staged" and
                                            bool(update % 2)))
                    rows = executor.map(reconstruct)
                    for name, logs in rows.items():
                        self.pretraining_history.append(dict(epoch=epoch + 1, update=update,
                                                             pattern=name, **logs))
                epochs.set_postfix(loss=f"{logs['reconstruction_loss']:.4f}", refresh=False)
                self.pretraining_epoch = epoch + 1
                self.stage, self.stage_epoch = "reconstruction", self.pretraining_epoch
                self.save_checkpoint(0)
        self.generator_pretraining = dict(origin="current_run",
                                          updates=len(self.pretraining_history),
                                          optimizer_states_imported=False)
        save_torch(self.out / "generator_reconstruction.pt", dict(
            models={name: _cpu_state(model) for name, model in self.models.items()},
            bank_hashes={name: bank_input_fingerprint(bank) for name, bank in self.banks.items()},
            updates=len(self.pretraining_history),
            test_used=False,
            origin=self.generator_pretraining,
            optimizer_states_imported=False))

    def _training_bank(self, name, ordinal, shared_density=None):
        """Use full mixed source densities and a sampled single-density view."""
        bank = self.banks[name]
        if ordinal % 2 == 0:
            return bank, "mixed"
        edges = bank.masks.sum((1, 2)).long()
        densities = torch.unique(edges)
        density = (densities[int(torch.randint(len(densities), (), generator=self.cpu_rng))]
                   if shared_density is None else torch.tensor(shared_density))
        rows = (edges == density).nonzero().flatten()
        key = (name, id(bank), int(density))
        view = self._training_views.get(key)
        if view is None:
            from generator_evaluator.data.adapters import FunctionalBank
            view = FunctionalBank(
                tokens=bank.tokens[:, rows],
                quality=None if bank.quality is None else bank.quality[:, rows],
                masks=bank.masks[rows], baseline_mask=bank.baseline_mask,
                provenance=bank.provenance,
                states=[bank.states[int(row)] for row in rows] if bank.states else [],
                diagnostics=bank.diagnostics)
            self._training_views[key] = view
        return view, str(int(density))

    def stage_update(self, stage, epoch, ordinal, output_k, auxiliary_budgets):
        """Dispatch a bootstrap, staged, or interleaved joint update."""
        if stage == STAGE_BOOTSTRAP_QUALITY:
            stage_label = "bootstrap_quality"
        elif stage == STAGE_QUALITY:
            stage_label = "quality"
        elif stage == STAGE_COOPERATION:
            stage_label = "cooperation"
        elif stage == STAGE_JOINT:
            stage_label = STAGE_JOINT
        else:
            raise ValueError(f"unknown generator stage: {stage}")

        shared_density = None
        if stage == STAGE_COOPERATION and ordinal % 2:
            available = set.intersection(*[
                set(bank.masks.sum((1, 2)).long().tolist()) for bank in self.banks.values()
            ])
            if available:
                densities = sorted(available)
                shared_density = densities[int(torch.randint(len(densities), (), generator=self.cpu_rng))]
        views, full_banks = {}, {}
        for name in self.patterns:
            if stage == STAGE_JOINT:
                # Cache the full live bank on each generator device. Feedback
                # invalidates this copy after mutating the source.
                self.trainer.set_bank(name, self.banks[name])
                full_banks[name] = self.trainer._prepared_bank(name)
                views[name] = "full"
                continue
            # Agreement compares matched input-density conditions. When
            # feedback leaves no shared stratum, use every full bank.
            bank_ordinal = (0 if stage == STAGE_COOPERATION and ordinal % 2
                            and shared_density is None else ordinal)
            view_bank, view = self._training_bank(name,
                bank_ordinal, shared_density)
            views[name] = view
            if self.trainer.banks[name] is not view_bank:
                self.trainer.set_bank(name, view_bank)
        if stage == STAGE_JOINT:
            update_ordinal = (epoch - 1) * self.config.updates_per_epoch + ordinal
            targets = self.distillation_targets()
            returned = joint_generator_update(
                models=self.models, banks=full_banks, optimizers=self.optimizers,
                # Retain the compatibility parameter as the complete per-device bank.
                training_views=full_banks,
                ensemble=self.ensemble, contexts=self.contexts,
                dense_quality=self.dense_quality.to(self.device), targets=targets,
                output_k=output_k, own_rngs=self.own_rngs, agreement_rng=self.rng,
                update_ordinal=update_ordinal,
                updates_per_epoch=self.config.updates_per_epoch,
                agreement_weight=self.config.agreement_weight,
                agreement_ramp_epochs=self.config.agreement_ramp_epochs,
                elite_weight=self.config.elite_distillation_weight,
                elite_limit=self.config.elite_limit,
                reconstruction_weight=self.config.reconstruction_weight,
                reconstruction_batch_size=self.config.reconstruction_batch_size,
                permutation_weight=self.config.permutation_weight,
                executor=self.generator_executor)
            for name in self.patterns:
                logs = dict(returned[name])
                logs["common_train_elite_count"] = float(len(self.common_elites()))
                self.history.append(dict(stage=stage_label, epoch=epoch,
                    stage_epoch=epoch, update=ordinal,
                    pattern=name, input_density=views[name], **logs))
        elif stage == STAGE_COOPERATION:
            targets = self.distillation_targets()
            update_ordinal = (epoch - 1) * self.config.cooperation_updates + ordinal
            ramp = min(1.0, (update_ordinal + 1) /
                       (min(self.config.agreement_ramp_epochs, self.config.cooperation_rounds)
                        * self.config.cooperation_updates))
            returned = self.trainer.cooperation_update(
                output_k, auxiliary_budgets, update_ordinal,
                self.config.agreement_weight * ramp, targets,
                elite_weight=self.config.elite_distillation_weight)
            for name in self.patterns:
                logs = dict(returned.get(name, {}))
                logs.setdefault("distillation_target_count", float(len(targets)))
                logs.setdefault("output_k", float(output_k))
                logs["common_train_elite_count"] = float(len(self.common_elites()))
                self.history.append(dict(stage=stage_label, epoch=self._global_epoch(stage, epoch),
                    stage_epoch=epoch, update=ordinal,
                    pattern=name, input_density=views[name], **logs))
        else:
            update_ordinal = (epoch - 1) * self.config.updates_per_epoch + ordinal
            for name in self.patterns:
                logs = self.trainer.quality_update(name, output_k, auxiliary_budgets,
                                                   update_ordinal)
                logs.setdefault("output_k", float(output_k))
                self.history.append(dict(stage=stage_label, epoch=self._global_epoch(stage, epoch),
                    stage_epoch=epoch, update=ordinal,
                    pattern=name, input_density=views[name], **logs))

    def _global_epoch(self, stage, stage_epoch):
        return (self.config.generator_epochs + stage_epoch
                if stage == STAGE_COOPERATION else stage_epoch)

    def refresh(self, stage, stage_epoch, auxiliary_budgets):
        """Acquire real masks and archive measured quality under the fixed critic."""
        cooperative = stage in (STAGE_COOPERATION, STAGE_JOINT)
        label = ("cooperation" if stage == STAGE_COOPERATION else
                 STAGE_JOINT if stage == STAGE_JOINT else
                 "bootstrap_quality" if stage == STAGE_BOOTSTRAP_QUALITY else "quality")
        excluded = ([] if self.config.training_mode == "joint" else
                    sorted(self.replay.mask_splits))
        proposal_trace = {}
        archive_masks = self.distillation_targets()
        shared_noise = (self.trainer.proposal_noise((self.config.candidates + 1) // 2,
                          rng=self.cpu_rng) if stage == STAGE_COOPERATION else None)
        masks, sources = propose_shared_pool(
            self.models, self.banks, self.config.k, self.config.candidates, self.rng,
            random_count=self.config.acquisition_budget,
            mutation_count=self.config.acquisition_budget if len(archive_masks) else 0,
            elites=archive_masks, excluded_topologies=excluded,
            proposal_trace=proposal_trace, paired_proposals=True,
            shared_noise=shared_noise, own_rngs=self.own_rngs,
            executor=self.generator_executor)
        families = [set(proposal_trace["generators"][name]["topology_ids"])
                    for name in self.patterns]
        intersection = set.intersection(*families)
        union = set.union(*families)
        overlap = dict(intersection_count=len(intersection), union_count=len(union),
                       jaccard=len(intersection) / max(1, len(union)),
                       unique_counts={name: len(family) for name, family in zip(self.patterns, families)},
                       sampled_per_generator=self.config.candidates, exact_k=self.config.k,
                       identity="hidden-column-permutation canonical topology, before exclusions")
        if cooperative:
            paired = [proposal_trace["generators"][name]["paired_topology_ids"]
                      for name in self.patterns]
            overlap["paired_exact_agreement"] = (
                sum(len(set(group)) == 1 for group in zip(*paired)) / max(1, len(paired[0])))
        ranks = rank_shared_pool(masks, self.ensemble, self.contexts, self.dense_quality,
                                 quality_objective=self.config.quality_objective)
        # A canonical topology is scored once on every training task. Stable
        # topology ordering resolves equal costs consistently across resumes.
        ranked_indices = sorted(range(len(masks)), key=lambda index: (
            float(ranks["objective_cost"][index]), topology_id(masks[index])))
        top_indices = ranked_indices[:self.config.acquisition_budget]
        top_pool_sources = [sources[index] for index in top_indices]
        top_ids = [topology_id(masks[index]) for index in top_indices]

        known_rows = {(row["topology_id"], row["task_id"]): row
                      for row in self.replay.records
                      if row["task_split"] == "train" and
                      row["task_id"] in {task.task_id for task in self.train_tasks}}
        task_ids = [task.task_id for task in self.train_tasks]
        new_fit_indices = []
        for index in ranked_indices:
            identity = topology_id(masks[index])
            if any((identity, task_id) not in known_rows for task_id in task_ids):
                new_fit_indices.append(index)
                if len(new_fit_indices) == self.config.acquisition_budget:
                    break
        cached_top_count = sum(all((identity, task_id) in known_rows for task_id in task_ids)
                               for identity in top_ids)
        acquisition_types = ["top_quality"] * len(top_indices)
        trace = top_pool_sources
        # The Toeplitz diagnostic still accepts the original single-loop
        # ``epoch_`` naming; the payload keeps the explicit joint stage.
        proposal_stem = "epoch" if stage == STAGE_JOINT else label
        proposal_path = self.out / "proposals" / f"{proposal_stem}_{stage_epoch:04d}.pt"
        save_torch(proposal_path, dict(
            stage=label, stage_epoch=stage_epoch, epoch=self._global_epoch(stage, stage_epoch),
            masks=masks, sources=sources, ranked_pool_indices=ranked_indices,
            ranked_pool_sources=[sources[index] for index in ranked_indices],
            ranks={key: value.cpu() for key, value in ranks.items()},
            selected=torch.tensor(top_indices, dtype=torch.long), top_pool_indices=top_indices,
            top_pool_sources=top_pool_sources,
            acquired_new_indices=new_fit_indices, acquired_new_count=len(new_fit_indices),
            cached_top_count=cached_top_count,
            acquisition_types=acquisition_types, selected_sources=trace,
            proposal_trace=proposal_trace, overlap=overlap,
            bank_hashes={name: bank_input_fingerprint(bank) for name, bank in self.banks.items()}))

        gaps = []
        entries = []
        for index in new_fit_indices:
            identity = topology_id(masks[index])
            for task_index, task in enumerate(self.train_tasks):
                if (identity, task.task_id) not in known_rows:
                    entries.append((masks[index], task, f"acquisition:{sources[index]}"))
        new_measurements = self.measure_many(entries) if entries else []
        for (mask, task, _), (row, _) in zip(entries, new_measurements):
            known_rows[(topology_id(mask), task.task_id)] = row

        # Calibration includes literal top-pool cache hits and all candidates
        # that received new labels, so replay reuse costs no child fits.
        calibrated_indices = list(dict.fromkeys(top_indices + new_fit_indices))
        for index in calibrated_indices:
            identity = topology_id(masks[index])
            for task_index, task in enumerate(self.train_tasks):
                row = known_rows.get((identity, task.task_id))
                if row is None:
                    raise RuntimeError("selected candidate lacks a real training-task label")
                predicted = float(ranks["mean"][index, task_index])
                std = float(ranks["std"][index, task_index])
                gaps.append(abs(predicted - row["quality"]))
                self.calibration.append(dict(stage=label, epoch=self._global_epoch(stage, stage_epoch),
                    stage_epoch=stage_epoch,
                    pattern=self.patterns[task_index], predicted=predicted, actual=row["quality"],
                    std=std, source=sources[index], topology_id=row["topology_id"],
                    partition=row["split"]))

        if self.config.auxiliary_budget and auxiliary_budgets:
            aux_k = auxiliary_budgets[len(self.refresh_history) % len(auxiliary_budgets)]
            entries = []
            for name in self.patterns:
                auxiliary = propose_at_budget(self.models[name], self.banks[name], aux_k,
                                              self.config.auxiliary_budget, self.own_rngs[name])
                entries.extend((mask, task, f"auxiliary:{name}:K{aux_k}")
                               for mask in auxiliary for task in self.train_tasks)
            self.measure_many(entries)

        # Functional teachers may consume labels, but the evaluator stays fixed
        # on the initial bank-membership dataset for the entire generator run.
        if cooperative:
            self.feedback()
        self.update_actual_archive()
        global_epoch = self._global_epoch(stage, stage_epoch)
        acquired = [masks[index] for index in new_fit_indices]
        literal_top_pool = [masks[index] for index in top_indices]
        archive = [row["mask"] for row in self.actual_archive]
        self.select(literal_top_pool + acquired + archive + [self.best_mask] +
                    list(self.common_elites()),
                    global_epoch, stage=label)

        gap = sum(gaps) / max(1, len(gaps))
        relative_gap = gap / max(float(self.dense_quality.abs().mean()), 1e-8)
        old_cadence = self._adapt_refresh_cadence(relative_gap)
        self.last_refresh += 1
        entry = dict(stage=label, epoch=global_epoch, stage_epoch=stage_epoch,
            prediction_mae=gap,
            relative_gap=relative_gap, previous_cadence=old_cadence,
            next_cadence=self.cadence, common_elites=len(self.common_elites()),
            actual_archive_size=len(self.actual_archive), proposal_overlap=overlap,
            top_pool_indices=top_indices, top_pool_sources=top_pool_sources,
            acquired_new_count=len(new_fit_indices), cached_top_count=cached_top_count,
            acquired_sources=trace,
            bank_sizes={name: bank.tokens.shape[1] for name, bank in self.banks.items()})
        self.refresh_history.append(entry)
        self.stage_last_refresh[stage] = stage_epoch
        return entry

    def feedback(self):
        masks, real, rows = _real_candidates(self.replay, self.train_tasks)
        if not len(masks):
            return
        costs = quality_objective_cost(
            real - self.dense_quality, self.config.quality_objective, task_dim=1)
        order = sorted(range(len(masks)),
                       key=lambda index: (float(costs[index]), topology_id(masks[index])))
        selected, densities = [], set()
        for index in order:
            edges = int(masks[index].sum())
            if edges not in densities:
                selected.append(index)
                densities.add(edges)
        selected += [index for index in order if int(masks[index].sum()) == self.config.k
                     and index not in selected][:self.config.feedback_masks]
        for index in progress(selected, desc="Real functional-map feedback", unit="mask"):
            for name, row in zip(self.patterns, rows[index]):
                payload = torch.load(row["artifact_path"], map_location="cpu", weights_only=False)
                append = append_feedback
                if self.config.domain == "deepsets":
                    from generator_evaluator.data.deepsets import append_deepsets_feedback
                    append = append_deepsets_feedback
                self.banks[name] = append(
                    self.banks[name], masks[index], payload["result"],
                    self.banks[name].diagnostics["probe_x"], task_id=row["task_id"],
                    artifact_path=row["artifact_path"], max_teachers=self.config.bank_capacity,
                    eligible=row["split"] == "train")
                for teacher in self.banks[name].states:
                    source = teacher.get("source", {})
                    if source.get("kind") == "feedback" and source.get("artifact_path") == row["artifact_path"]:
                        source["query_error"] = float(payload["result"]["replica_losses"][source["replica"]])
        self.export_functional_cards()
        for name in self.patterns:
            self.trainer.set_bank(name, self.banks[name])
            self.trainer.invalidate_bank_cache(name)
        self._training_views.clear()

    def update_actual_archive(self):
        """Keep the strongest fully measured exact-K training candidates."""
        masks, quality, _ = _real_candidates(self.replay, self.train_tasks, self.config.k)
        if not len(masks):
            self.actual_archive = []
            return
        delta = quality - self.dense_quality
        worst_delta = delta.max(1).values
        costs = quality_objective_cost(delta, self.config.quality_objective, task_dim=1)
        order = sorted(range(len(masks)),
                       key=lambda index: (float(costs[index]), topology_id(masks[index])))[:
                           self.config.elite_limit]
        self.actual_archive = [dict(mask=masks[index].clone().cpu(),
                                    qualities=quality[index].clone().cpu(),
                                    worst_delta=float(worst_delta[index]),
                                    objective_cost=float(costs[index]))
                               for index in order]

    def select(self, masks, epoch, *, stage="initial"):
        seen, unique = set(), []
        for mask in masks:
            identity = topology_id(mask)
            if identity not in seen:
                seen.add(identity)
                unique.append(mask)
        labels = self.measure_many([(mask, task, "selection") for mask in unique
                                    for task in self.selection_tasks])
        for index, mask in enumerate(unique):
            delta = torch.tensor([
                labels[len(self.selection_tasks) * index + task_index][0]["quality"]
                - self.dense_rows[task.task_id]["quality"]
                for task_index, task in enumerate(self.selection_tasks)], dtype=torch.float64)
            cost = float(quality_objective_cost(delta, self.config.quality_objective))
            if cost < self.best_cost:
                self.best_mask, self.best_cost, self.best_epoch = mask.clone().cpu(), cost, epoch
                self.best_mean_delta, self.best_worst_delta = float(delta.mean()), float(delta.max())
                self.best_stage = stage
                self.best_models = {name: _cpu_state(model) for name, model in self.models.items()}
                self.best_evaluator, self.best_banks = _cpu_state(self.ensemble), copy.deepcopy(self.banks)

    def save_checkpoint(self, epoch, *, stage=None, stage_epoch=None):
        if stage is not None:
            self.stage = stage
        if stage_epoch is not None:
            self.stage_epoch = stage_epoch
        cuda_rng_states = {}
        if torch.cuda.is_available():
            for device in sorted({str(item) for item in self.generator_devices} |
                                 ({str(self.device)} if str(self.device).startswith("cuda") else set())):
                cuda_rng_states[device] = torch.cuda.get_rng_state(torch.device(device)).cpu()
        payload = dict(
            algorithm_version=7, training_mode=self.config.training_mode,
            shared_latent_used=(self.config.phase == "search" and
                                self.config.training_mode == "staged"),
            best_mean_delta=self.best_mean_delta, best_worst_delta=self.best_worst_delta,
            epoch=epoch, last_refresh=self.last_refresh, cadence=self.cadence,
            stage=self.stage, stage_epoch=self.stage_epoch,
            stage_last_refresh=self.stage_last_refresh,
            models={name: _cpu_state(model) for name, model in self.models.items()},
            budgets={name: model.target_k for name, model in self.models.items()},
            optimizers={name: optimizer.state_dict() for name, optimizer in self.optimizers.items()},
            trainer_state=self.trainer.state_dict(),
            initial_shared_latent=self.initial_shared_latent,
            generator_devices=self.generator_devices,
            ensemble=_cpu_state(self.ensemble),
            evaluator_training_state=getattr(self.ensemble, "training_state", None),
            evaluator_policy=self.evaluator_policy,
            evaluator_bank_topology_ids=self.evaluator_bank_topology_ids,
            evaluator_bank_rows=self.evaluator_bank_rows,
            evaluator_bank_fingerprint=self.evaluator_bank_fingerprint,
            banks=self.banks, replay=self.replay, history=self.history,
            pretraining_history=self.pretraining_history,
            pretraining_epoch=self.pretraining_epoch, initialization_done=self.initialization_done,
            generator_pretraining=self.generator_pretraining,
            evaluator_history=self.evaluator_history, calibration=self.calibration,
            refresh_history=self.refresh_history, best_mask=self.best_mask,
            best_cost=self.best_cost, best_epoch=self.best_epoch, best_models=self.best_models,
            best_stage=self.best_stage,
            best_evaluator=self.best_evaluator, best_banks=self.best_banks,
            actual_archive=self.actual_archive,
            own_rngs={name: own_rng.get_state().cpu() for name, own_rng in self.own_rngs.items()},
            rng_state=self.rng.get_state().cpu(), cpu_rng_state=self.cpu_rng.get_state(),
            torch_rng_state=torch.get_rng_state(),
            cuda_rng_states=cuda_rng_states)
        self.session.save_checkpoint(payload, replay=self.replay, history=dict(
            generator=self.history, generator_pretraining=self.pretraining_history,
            evaluator=self.evaluator_history,
            refreshes=self.refresh_history, calibration=self.calibration))


def _validate_pattern_inputs(banks, train_tasks, selection_tasks, test_spec, config, protocol):
    patterns = tuple(config.train_patterns)
    test_patterns = config.effective_test_patterns
    if (tuple(banks) != patterns or len(train_tasks) != len(patterns) or
            len(selection_tasks) != len(patterns) or len(patterns) < 2 or
            protocol.replicas < 2 or protocol.metric != "bce"):
        raise ValueError("requires ordered own-pattern banks/tasks and at least two BCE replicas")
    if (not isinstance(test_spec, dict) or test_spec.get("family") != "cooperative_pattern" or
            test_spec.get("test_pattern") != test_patterns[0] or
            tuple(test_spec.get("test_patterns", (test_spec.get("test_pattern"),))) != test_patterns or
            test_spec.get("materialized") is not False or
            tuple(test_spec.get("train_patterns", ())) != patterns):
        raise ValueError("held-out patterns must be a sealed test specification")

    child_specs = test_spec.get("test_specs")
    if child_specs is None:
        if len(test_patterns) != 1:
            raise ValueError("multi-pattern test specification lacks ordered child specifications")
        child_specs = [test_spec]
    if not isinstance(child_specs, (list, tuple)) or len(child_specs) != len(test_patterns):
        raise ValueError("sealed test child specifications do not match ordered test patterns")

    def ids_for(spec, name):
        try:
            values = torch.as_tensor(spec[name], dtype=torch.long)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("sealed test requires valid support/query observation pools") from error
        if values.ndim != 1 or len(torch.unique(values)) != len(values):
            raise ValueError("sealed test requires unique one-dimensional observation pools")
        return set(values.tolist())

    child_ids, child_support_ids, child_query_ids = set(), set(), set()
    common_support_count, common_query_count = test_spec.get("support_count"), test_spec.get("query_count")
    if (not isinstance(common_support_count, int) or common_support_count < 1 or
            not isinstance(common_query_count, int) or common_query_count < 1):
        raise ValueError("sealed test requires positive support/query counts")
    for pattern, child in zip(test_patterns, child_specs):
        if (not isinstance(child, dict) or child.get("family") != "cooperative_pattern" or
                child.get("test_pattern") != pattern or child.get("materialized") is not False or
                tuple(child.get("train_patterns", ())) != patterns or
                child.get("support_count") != common_support_count or
                child.get("query_count") != common_query_count or
                ("seed" in test_spec and child.get("seed") != test_spec["seed"])):
            raise ValueError("sealed test child role or metadata does not match its parent")
        ids = ids_for(child, "test_ids")
        support_ids = ids_for(child, "test_support_ids")
        query_ids = ids_for(child, "test_query_ids")
        if (not support_ids.isdisjoint(query_ids) or support_ids | query_ids != ids or
                common_support_count > len(support_ids) or common_query_count > len(query_ids)):
            raise ValueError("sealed test requires independent support/query pools of sufficient size")
        child_ids.update(ids)
        child_support_ids.update(support_ids)
        child_query_ids.update(query_ids)

    sealed_ids = ids_for(test_spec, "test_ids")
    sealed_support_ids = ids_for(test_spec, "test_support_ids")
    sealed_query_ids = ids_for(test_spec, "test_query_ids")
    if (sealed_ids != child_ids or sealed_support_ids != child_support_ids or
            sealed_query_ids != child_query_ids or
            not sealed_support_ids.isdisjoint(sealed_query_ids) or
            sealed_support_ids | sealed_query_ids != sealed_ids or
            common_support_count > len(sealed_support_ids) or
            common_query_count > len(sealed_query_ids)):
        raise ValueError("sealed test parent pools do not match its ordered child specifications")
    for pattern, task, selection in zip(patterns, train_tasks, selection_tasks):
        bank = banks[pattern]
        if (bank.provenance.get("pattern") != pattern or task.task_id != f"pattern:{pattern}" or
                task.split != "train" or selection.split != "validation" or
                selection.task_id != f"pattern:{pattern}:selection" or
                selection.provenance.get("pattern") != pattern or
                selection.provenance.get("role") != "selection" or
                bank.masks.shape[1:] != (11, 8) or int(bank.baseline_mask.sum()) != config.k):
            raise ValueError("bank/task pattern, role or dimensions mismatch")
        if (not torch.equal(task.support_ids, selection.support_ids) or
                not torch.equal(task.x_support, selection.x_support) or
                not torch.equal(task.y_support, selection.y_support) or
                not torch.equal(task.context, selection.context)):
            raise ValueError("selection must reuse its own train support/context")
        if not set(task.query_ids.tolist()).isdisjoint(selection.query_ids.tolist()):
            raise ValueError("training and selection query observations overlap")
        for observed in (task, selection):
            if sealed_ids.intersection(observed.support_ids.tolist() + observed.query_ids.tolist()):
                raise ValueError("test observation leakage")
    return sealed_ids, sealed_support_ids, sealed_query_ids


def run_cooperative_experiment(banks, train_tasks, selection_tasks, test_spec, out,
                               protocol, config, *, device="cpu", resume=False,
                               dense_learning_rates=None, test_factory=None, build_settings=None,
                               warm_start=None, measurement_devices=None, measurement_batch_size=8,
                               generator_devices=None, generator_pretrained_from=None):
    with ExitStack() as cleanup:
        return _run_cooperative_experiment(banks, train_tasks, selection_tasks, test_spec,
            out, protocol, config, device=device, resume=resume,
            dense_learning_rates=dense_learning_rates, test_factory=test_factory,
            build_settings=build_settings, warm_start=warm_start,
            measurement_devices=measurement_devices, measurement_batch_size=measurement_batch_size,
            generator_devices=generator_devices,
            generator_pretrained_from=generator_pretrained_from,
            cleanup=cleanup)


def _device_list(values, fallback, count, *, label):
    devices = tuple(str(value) for value in (values or (fallback,)))
    if not devices:
        raise ValueError(f"{label} must contain at least one device")
    for value in devices:
        try:
            parsed = torch.device(value)
        except (ValueError, RuntimeError) as error:
            raise ValueError(f"invalid {label} device: {value}") from error
        if parsed.type not in ("cpu", "cuda"):
            raise ValueError(f"unsupported {label} device: {value}")
        if parsed.type == "cuda" and (not torch.cuda.is_available() or
                (parsed.index is not None and parsed.index >= torch.cuda.device_count())):
            raise ValueError(f"unavailable {label} device: {value}")
    return tuple(devices[index % len(devices)] for index in range(count))


def _load_generator_reconstruction(path, banks, config):
    """Validate a heldout-safe reconstruction artifact for the current banks."""
    source = Path(path).expanduser().resolve()
    try:
        content = source.read_bytes()
    except OSError as error:
        raise ValueError(f"generator reconstruction artifact is unavailable: {source}") from error
    checksum = hashlib.sha256(content).hexdigest()
    try:
        artifact = torch.load(io.BytesIO(content), map_location="cpu", weights_only=False)
    except Exception as error:
        raise ValueError(f"cannot read generator reconstruction artifact: {source}") from error
    if not isinstance(artifact, Mapping):
        raise ValueError("generator reconstruction artifact must be a mapping")

    roles = tuple(config.train_patterns)
    states, bank_hashes = artifact.get("models"), artifact.get("bank_hashes")
    if not isinstance(states, Mapping) or set(states) != set(roles):
        raise ValueError("generator reconstruction roles must exactly match train bank roles")
    if not isinstance(bank_hashes, Mapping) or set(bank_hashes) != set(roles):
        raise ValueError("generator reconstruction bank roles must exactly match train bank roles")
    expected_hashes = {name: bank_input_fingerprint(banks[name]) for name in roles}
    if dict(bank_hashes) != expected_hashes:
        raise ValueError("generator reconstruction bank fingerprints do not match current banks")
    if artifact.get("test_used") is not False:
        raise ValueError("generator reconstruction artifact must record test_used=False")
    updates = artifact.get("updates")
    if isinstance(updates, bool) or not isinstance(updates, int) or updates < 0:
        raise ValueError("generator reconstruction update count must be a nonnegative integer")

    # Instantiate CPU templates in an isolated RNG scope so validation cannot
    # alter the current run's initialization stream.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        templates = {
            name: DensityConditionedGenerator(
                banks[name].tokens.shape[-1], config.features, config.hidden,
                config.width, config.heads, config.layers, config.noise_dim,
                target_k=config.k)
            for name in roles
        }
    checked_states = {}
    for name, template in templates.items():
        state = states[name]
        if not isinstance(state, Mapping):
            raise ValueError(f"generator reconstruction state for role {name} must be a mapping")
        expected = template.state_dict()
        if set(state) != set(expected):
            raise ValueError(f"generator reconstruction state keys do not match role {name}")
        for key, value in expected.items():
            saved = state[key]
            if not isinstance(saved, torch.Tensor) or saved.shape != value.shape:
                raise ValueError(f"generator reconstruction state shape mismatch for {name}.{key}")
        try:
            template.load_state_dict(state, strict=True)
        except (RuntimeError, TypeError) as error:
            raise ValueError(f"generator reconstruction state does not load strictly for {name}") from error
        checked_states[name] = state

    provenance = dict(path=str(source), sha256=checksum, updates=updates,
                      optimizer_states_imported=False,
                      optimizer_initialization="fresh_adam")
    return checked_states, provenance


def _run_cooperative_experiment(banks, train_tasks, selection_tasks, test_spec, out,
                                protocol, config, *, device, resume, dense_learning_rates,
                                test_factory, build_settings, warm_start, measurement_devices,
                                measurement_batch_size, generator_devices,
                                generator_pretrained_from, cleanup):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if generator_pretrained_from is not None and config.phase != "search":
        raise ValueError("generator-pretrained-from is only valid for search runs")
    generator_devices = _device_list(generator_devices, device, len(config.train_patterns),
                                    label="generator")
    if measurement_devices is not None:
        measurement_devices = tuple(map(str, measurement_devices))
        if not measurement_devices or measurement_batch_size < 1:
            raise ValueError("measurement devices and batch size must be nonempty and positive")
    patterns = tuple(config.train_patterns)
    if config.domain == "pattern":
        sealed_ids, sealed_support_ids, sealed_query_ids = _validate_pattern_inputs(
            banks, train_tasks, selection_tasks, test_spec, config, protocol)
    else:
        from generator_evaluator.data.deepsets import validate_cooperative_deepsets_inputs
        validate_cooperative_deepsets_inputs(banks, train_tasks, selection_tasks,
                                            test_spec, config, protocol)
        sealed_ids = sealed_support_ids = sealed_query_ids = set()
    initial_evaluator_topology_ids = (
        tuple(warm_start.evaluator_bank_topology_ids)
        if warm_start is not None and warm_start.reuse_evaluator
        else tuple(warm_start.source_bank_topology_ids)
        if warm_start is not None
        else _bank_topology_ids(banks))
    imported_generator_states = None
    generator_pretraining = None
    has_resume_checkpoint = resume and (out / "checkpoint.pt").is_file()
    if has_resume_checkpoint:
        resume_header = torch.load(out / "checkpoint.pt", map_location="cpu", weights_only=False)
        version = resume_header.get("algorithm_version")
        saved_mode = resume_header.get("training_mode")
        if version != 7 or saved_mode != config.training_mode:
            raise ValueError("checkpoint predates the frozen initial-bank evaluator policy; "
                             "use --warm-start-from to initialize a fresh critic")
    if generator_pretrained_from is not None:
        imported_generator_states, generator_pretraining = _load_generator_reconstruction(
            generator_pretrained_from, banks, config)
    project = Path(__file__).resolve().parents[2]
    source_files = sorted((path for path in Path(__file__).resolve().parents[1].rglob("*.py") if "tests" not in path.parts)) + [
        project / "pattern/task_quality/meta.py", project / "meta_pattern/data.py",
        project / "deepsets_vaae/permutation_utility_loss.py"]
    if config.domain == "deepsets":
        source_files += [project / "deepsets_vaae" / name for name in
                         ("core.py", "utility_graph_child.py", "followup_batched_eval.py")]
    spec = dict(config=_config_metadata(config), requested_protocol=asdict(protocol), device=str(device),
                generator_devices=list(generator_devices),
                measurement_execution=dict(devices=measurement_devices,
                                           batch_size=measurement_batch_size),
                banks={name: bank_input_fingerprint(bank) for name, bank in banks.items()},
                tasks={task.task_id: task.fingerprint for task in train_tasks + selection_tasks},
                test_spec=test_spec, dense_learning_rates=dense_learning_rates, build_settings=build_settings,
                warm_start=None if warm_start is None else warm_start.provenance,
                evaluator_policy=_EVALUATOR_POLICY,
                evaluator_bank_topology_ids=list(initial_evaluator_topology_ids),
                generator_pretrained_from=generator_pretraining,
                source_hashes={str(p.relative_to(project)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in source_files})
    if config.domain == "deepsets":
        spec["evaluator_task_ids"] = {task.task_id: task.provenance.get("evaluator_task_id")
                                      for task in train_tasks + selection_tasks}
    spec = json.loads(json.dumps(spec))
    session = RunSession(out, spec, source_files, project=project, resume=resume,
                         save_torch_fn=save_torch, save_json_fn=save_json)
    try:
        completed = session.prepare(inputs=dict(banks=banks, train_tasks=train_tasks,
                                                selection_tasks=selection_tasks, test_spec=test_spec))
    except ValueError as error:
        if "different" in str(error) and "settings" in str(error):
            raise ValueError("existing output has different code/inputs/settings; choose new --out") from error
        raise
    if completed is not None:
        return completed
    if warm_start is not None:
        protocol = warm_start.protocol
    banks = copy.deepcopy(banks)
    torch.manual_seed(config.seed)
    rng = torch.Generator().manual_seed(config.seed + 101)
    shared_noise_rng = torch.Generator().manual_seed(config.seed + 103)
    cpu_rng = torch.Generator().manual_seed(config.seed + 102)
    own_rngs = {name: torch.Generator(device=generator_devices[i]).manual_seed(config.seed + 1009 * (i + 1))
                for i, name in enumerate(patterns)}
    dense = torch.ones(config.features, config.hidden)
    if (out / "protocol.json").exists():
        protocol = InnerProtocol(**json.loads((out / "protocol.json").read_text())["inner_protocol"])
    else:
        if warm_start is not None:
            tuning = dict(settings=[], selected=dict(lr=protocol.lr, protocol_id=protocol.fingerprint),
                          selection_rule="reuse source fixed solver", label_measurements=0,
                          test_used=False, fixed_solver_for_all_methods=True)
        elif config.tune_dense:
            protocol = _dense_tune(out, selection_tasks, dense, protocol, device,
                dense_learning_rates or [protocol.lr / 3, protocol.lr, protocol.lr * 3],
                measurement_devices=measurement_devices,
                measurement_batch_size=measurement_batch_size)
            tuning = json.loads((out / "dense_tuning.json").read_text())
        else:
            tuning = dict(settings=[], selected=dict(lr=protocol.lr, protocol_id=protocol.fingerprint),
                          selection_rule="fixed solver; tuning omitted", label_measurements=0,
                          test_used=False, fixed_solver_for_all_methods=True)
        tuning["selection_split"] = "within-task independent selection observations"
        save_json(out / "dense_tuning.json", tuning)
        save_json(out / "protocol.json", dict(inner_protocol=asdict(protocol),
                  protocol_id=protocol.fingerprint, search=_config_metadata(config),
                  quality=f"mean fresh terminal query {protocol.metric} across independent initializations",
                  objective=config.quality_objective,
                  selection="independent queries on all training tasks; heldout tasks test only",
                  smoke_only=config.smoke))
    replay = (RealReplay(protocol, split_seed=config.seed) if warm_start is None
              else warm_start.materialize_replay(out))
    starting_bank_topology_ids = initial_evaluator_topology_ids
    if measurement_devices is not None or config.domain == "deepsets":
        from generator_evaluator.evaluation.parallel import ParallelMeasurementStore
        store = cleanup.enter_context(ParallelMeasurementStore(out, replay, device,
            devices=measurement_devices or [device], batch_size=measurement_batch_size))
    elif config.batch_children:
        from generator_evaluator.evaluation.parallel import CooperativeBatchedMeasurementStore as BatchedMeasurementStore
        store = BatchedMeasurementStore(out, replay, device)
    else:
        store = MeasurementStore(out, replay, device)

    def measure_many(entries):
        if measurement_devices is not None or config.batch_children or config.domain == "deepsets":
            return store.measure_many(entries)
        return [store.measure(mask, task, origin) for mask, task, origin in
                progress(entries, desc="Fresh mask fits", unit="fit")]
    contexts = torch.stack([task.context for task in train_tasks]).to(device)
    models = {name: DensityConditionedGenerator(bank.tokens.shape[-1], config.features, config.hidden,
              config.width, config.heads, config.layers, config.noise_dim, target_k=config.k).to(generator_devices[index])
              for index, (name, bank) in enumerate(banks.items())}
    if imported_generator_states is not None and not has_resume_checkpoint:
        try:
            for name, model in models.items():
                model.load_state_dict(imported_generator_states[name], strict=True)
        except (RuntimeError, TypeError) as error:
            raise ValueError(f"generator reconstruction state does not load strictly: {error}") from error
        save_torch(out / "generator_reconstruction.pt", dict(
            models={name: _cpu_state(model) for name, model in models.items()},
            bank_hashes={name: bank_input_fingerprint(bank) for name, bank in banks.items()},
            updates=generator_pretraining["updates"], test_used=False,
            origin=dict(kind="imported", **generator_pretraining),
            optimizer_states_imported=False,
            optimizer_initialization="fresh_adam"))
        generator_pretraining = dict(
            origin="imported", source_path=generator_pretraining["path"],
            source_sha256=generator_pretraining["sha256"],
            source_updates=generator_pretraining["updates"], updates_this_run=0,
            optimizer_states_imported=False,
            optimizer_initialization="fresh_adam")
    optimizers = {name: torch.optim.Adam(model.parameters(), lr=config.generator_lr)
                  for name, model in models.items()}
    ensemble = QualityEnsemble(config.features, contexts.shape[1], num_members=config.ensemble_members,
                               width=config.width, heads=config.heads, layers=config.layers).to(device)
    if warm_start is not None:
        if warm_start.reuse_evaluator:
            ensemble.load_state_dict(warm_start.ensemble_state)
            ensemble.training_state = copy.deepcopy(warm_start.evaluator_training_state)
        else:
            # Older warm starts may have updated their critic online. Start a
            # fresh critic and fit only the current input-bank labels once.
            ensemble.training_state = None
    checkpoint = out / "checkpoint.pt"
    saved = None
    if resume and checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        version = saved.get("algorithm_version")
        if (version != 7 or "trainer_state" not in saved or
                saved.get("training_mode") != config.training_mode):
            raise ValueError("checkpoint predates the frozen initial-bank evaluator policy; "
                             "use --warm-start-from to initialize a fresh critic")
        if tuple(saved.get("generator_devices", ())) != tuple(generator_devices):
            raise ValueError("checkpoint generator devices differ from the requested run")
        replay, banks = saved["replay"], saved["banks"]
        replay.validate()
        if replay.protocol.fingerprint != protocol.fingerprint:
            raise ValueError("checkpoint solver mismatch")
        store.replay = replay

    dense_tasks = train_tasks + selection_tasks
    dense_rows = {task.task_id: result[0] for task, result in
                  zip(dense_tasks, measure_many([(dense, task, "dense") for task in dense_tasks]))}
    dense_quality = torch.tensor([dense_rows[t.task_id]["quality"] for t in train_tasks])

    from generator_evaluator.training.staged import StagedGeneratorTrainer
    trainer = StagedGeneratorTrainer(
        models, banks, optimizers, ensemble, contexts, dense_quality,
        own_rngs, shared_noise_rng, config.noise_dim, config.generator_lr,
        config.latent_lr, seed=config.seed, permutation_weight=config.permutation_weight,
        elite_limit=config.elite_limit,
        reconstruction_weight=config.reconstruction_weight,
        reconstruction_batch_size=config.reconstruction_batch_size,
        quality_objective=config.quality_objective)
    controller = CooperativeSearchController(
        out=out, config=config, device=device, patterns=patterns, banks=banks,
        train_tasks=train_tasks, selection_tasks=selection_tasks, replay=replay, store=store,
        models=models, optimizers=optimizers, ensemble=ensemble, dense_rows=dense_rows,
        dense_quality=dense_quality, contexts=contexts, rng=rng, cpu_rng=cpu_rng, own_rngs=own_rngs,
        checkpoint=checkpoint, measure_many=measure_many, session=session,
        trainer=trainer, generator_devices=generator_devices,
        generator_pretraining=generator_pretraining,
        initial_bank_topology_ids=starting_bank_topology_ids)
    if saved is not None:
        controller.restore(saved)
    controller.export_functional_cards()

    functional_controls = {}
    functional_mean = None
    if config.domain == "pattern":
        from generator_evaluator.search.consensus import build_functional_consensus_proposals
        consensus = build_functional_consensus_proposals(list(banks.values()), config.k)
        functional_controls = dict(functional_consensus=consensus.global_topk,
            functional_balanced_consensus=consensus.balanced_per_column)
        save_torch(out / "functional_consensus.pt", dict(
            importance=consensus.importance, masks=functional_controls,
            train_patterns=patterns, test_used=False,
            normalization="unit column mass per source map, equal means within and across banks",
            alignment="ordered input-coordinate centroids",
            balanced_degree_prior="floor(K/H) per hidden column plus strongest remaining edges",
            analytical_structure_target_used=False))

    if config.domain == "deepsets" and config.phase != "bootstrap":
        from generator_evaluator.search.consensus import build_functional_consensus_proposals
        functional_mean = build_functional_consensus_proposals(list(banks.values()), config.k).global_topk

    bootstrap_critic_only = config.phase == "bootstrap" and not config.bootstrap_generators
    if not controller.initialization_done:
        if warm_start is not None:
            if warm_start.reuse_evaluator:
                controller.fit_evaluator(
                    warm_start.evaluator_bank_topology_ids,
                    row_ids=warm_start.evaluator_bank_rows,
                    expected_fingerprint=warm_start.evaluator_bank_fingerprint,
                    train=False)
            else:
                # Historical warm starts may have trained their critic online;
                # keep its generator/maps but refit a fresh critic only on the
                # current input banks' original mask membership.
                controller.ensure_initial_bank_labels(warm_start.source_bank_masks)
                controller.fit_evaluator(initial_evaluator_topology_ids)
            if functional_controls:
                measure_many([(mask, task, f"control:{name}")
                    for name, mask in functional_controls.items() for task in train_tasks])
            controller.update_actual_archive()
            controller.select([warm_start.best_mask] +
                [bank.baseline_mask for bank in banks.values()] + list(functional_controls.values()) +
                [row["mask"] for row in controller.actual_archive] +
                list(controller.common_elites()),
                controller.best_epoch, stage="warm_start")
        else:
            # Every precollected input-bank mask receives real cross-task labels;
            # the evaluator later filters this pool by immutable bank membership.
            initial = []
            for name, bank in banks.items():
                initial.extend((mask, f"bank:{name}") for mask in bank.masks)
                initial.append((bank.baseline_mask, f"functional:{name}"))
            initial.extend((mask, "random") for mask in _random_masks(
                config.initial_random, config.features, config.hidden, config.k, cpu_rng))
            initial.extend((mask, f"control:{name}") for name, mask in functional_controls.items())
            for partition in ("train", "holdout"):
                for _ in range(1000):
                    distinct = {topology_id(mask) for mask, _ in initial
                                if replay.mask_split(mask) == partition}
                    if len(distinct) >= 2:
                        break
                    initial.append((_random_masks(1, config.features, config.hidden,
                                                   config.k, cpu_rng)[0], "partition_reserve"))
                else:
                    raise RuntimeError("could not create topology partitions")
            seen, entries = set(), []
            for mask, origin in initial:
                identity = topology_id(mask)
                if identity in seen:
                    continue
                seen.add(identity)
                entries.extend((mask, task, origin) for task in train_tasks)
            measure_many(entries)
            controller.fit_evaluator(initial_evaluator_topology_ids)
            if not bootstrap_critic_only:
                controller.feedback()
            controller.update_actual_archive()
            controller.select([bank.baseline_mask for bank in banks.values()] +
                              list(functional_controls.values()) +
                              [row["mask"] for row in controller.actual_archive] +
                              list(controller.common_elites()), 0, stage="prepare")
        controller.initialization_done = True
        controller.stage = (STAGE_TRAINING_COMPLETE if bootstrap_critic_only else
                            STAGE_BOOTSTRAP_QUALITY
                            if config.phase == "bootstrap" and config.training_mode == "staged"
                            else "reconstruction")
        controller.stage_epoch = 0
        controller.save_checkpoint(0)

    if (not bootstrap_critic_only and
            (config.phase == "search" or config.training_mode == "joint") and
            controller.stage in (None, "reconstruction")):
        if imported_generator_states is None and config.generator_pretrain_epochs:
            controller.pretrain_generators()
        controller.stage = (STAGE_JOINT if config.training_mode == "joint" else STAGE_QUALITY)
        controller.stage_epoch = 0
        controller.save_checkpoint(controller.global_epoch)
    elif (not bootstrap_critic_only and config.phase == "bootstrap" and
          controller.stage is None):
        controller.stage, controller.stage_epoch = STAGE_BOOTSTRAP_QUALITY, 0
        controller.save_checkpoint(controller.global_epoch)

    budgets = sorted({config.k, *config.output_budgets})
    auxiliary_budgets = [value for value in budgets if value != config.k]
    if config.training_mode == "joint":
        schedule = joint_search_stages(epochs=config.generator_epochs,
                                       updates=config.updates_per_epoch)
    else:
        schedule = search_stages(
            phase=config.phase, quality_epochs=config.generator_epochs,
            quality_updates=config.updates_per_epoch,
            cooperation_rounds=config.cooperation_rounds,
            cooperation_updates=config.cooperation_updates)
    stage_limits = {STAGE_QUALITY: config.generator_epochs,
                    STAGE_BOOTSTRAP_QUALITY: config.generator_epochs,
                    STAGE_COOPERATION: config.cooperation_rounds,
                    STAGE_JOINT: config.generator_epochs}

    def stage_update(stage, stage_epoch, ordinal):
        stage_updates = (config.cooperation_updates if stage == STAGE_COOPERATION
                         else config.updates_per_epoch)
        update_index = (stage_epoch - 1) * stage_updates + ordinal
        output_k = (config.k if stage == STAGE_COOPERATION or update_index % 2 == 0 or
                    not auxiliary_budgets else
                    auxiliary_budgets[(update_index // 2) % len(auxiliary_budgets)])
        controller.stage_update(stage, stage_epoch, ordinal, output_k, auxiliary_budgets)

    def finish_stage_epoch(stage, stage_epoch):
        if stage == STAGE_COOPERATION:
            due = True
        else:
            last = controller.stage_last_refresh.get(stage, 0)
            due = (stage_epoch - last >= controller.cadence or
                   stage_epoch == stage_limits[stage])
        if due:
            row = controller.refresh(stage, stage_epoch, auxiliary_budgets)
            print(f"{row['stage']} epoch={stage_epoch}: labels={len(controller.replay.records)}, "
                  f"selection_cost={controller.best_cost:.5f}, "
                  f"worst_delta={controller.best_worst_delta:.5f}, archive={len(controller.actual_archive)}",
                  flush=True)

    def checkpoint_stage(stage, stage_epoch):
        controller.global_epoch = controller._global_epoch(stage, stage_epoch)
        controller.save_checkpoint(controller.global_epoch, stage=stage, stage_epoch=stage_epoch)

    from generator_evaluator.training.device_executor import PerDeviceGeneratorExecutor
    with PerDeviceGeneratorExecutor(patterns, generator_devices) as executor:
        controller.generator_executor = executor
        if not bootstrap_critic_only:
            schedule.run(start_stage=controller.stage, start_epoch=controller.stage_epoch,
                update=stage_update, finish_epoch=finish_stage_epoch, checkpoint=checkpoint_stage)
            controller.stage = STAGE_TRAINING_COMPLETE
            controller.save_checkpoint(controller.global_epoch, stage=STAGE_TRAINING_COMPLETE)

    best_mask, best_cost, best_epoch = controller.best_mask, controller.best_cost, controller.best_epoch
    best_models, best_evaluator, best_banks = controller.best_models, controller.best_evaluator, controller.best_banks
    history, evaluator_history = controller.history, controller.evaluator_history
    calibration, refresh_history = controller.calibration, controller.refresh_history
    replay, banks = controller.replay, controller.banks
    if config.phase == "bootstrap":
        # The checkpoint already contains the live critic, banks and train-only
        # replay. Heldout tasks remain unopened throughout preparation.
        replay.save(out / "replay.pt")
        save_torch(out / "final_banks.pt", banks)
        summary = dict(domain=config.domain, phase="bootstrap", train_patterns=list(patterns),
            test_pattern=config.test_pattern, test_patterns=list(config.effective_test_patterns),
            preset=config.preset, training_mode=config.training_mode, generators=len(models),
            global_evaluator_members=config.ensemble_members, best_epoch=best_epoch,
            best_selection_delta=controller.best_worst_delta, best_selection_cost=best_cost,
            best_selection_mean_delta=controller.best_mean_delta,
            quality_objective=config.quality_objective, best_stage=controller.best_stage,
            completed_stage=controller.stage, protocol_id=protocol.fingerprint,
            common_train_elites=len(controller.common_elites()), smoke_only=config.smoke,
            real_label_measurements=len(replay.records), bank_sizes={name:len(bank.masks) for name,bank in banks.items()},
            refreshes=refresh_history, final={}, test_materialized=False,
            test_used_for_training=False, next_stage="search with --warm-start-from this directory")
        summary["generator_pretraining_updates"] = len(controller.pretraining_history)
        summary["quality_updates"] = sum(row["stage"] in (STAGE_BOOTSTRAP_QUALITY, STAGE_JOINT)
                                         for row in history)
        summary["joint_updates"] = sum(row["stage"] == STAGE_JOINT for row in history)
        summary["cooperation_updates"] = 0
        summary["evaluator_task_ids"] = {task.task_id: task.provenance.get("evaluator_task_id")
                                         for task in train_tasks}
        save_json(out / "summary.json", summary)
        write_plots(out, history, [], evaluator_history=evaluator_history,
                    calibration=calibration, within_task_selection=True,
                    generator_pretraining=controller.pretraining_history)
        (out / "COMPLETE").write_text("bootstrap complete; heldout test remains sealed\n")
        return summary
    toeplitz = SlidingWindowMaskPrior().mask() if config.domain == "pattern" and config.k == 32 else None
    methods = {"common": best_mask, "random": _random_masks(1, config.features, config.hidden, config.k, cpu_rng)[0],
               **{f"functional_{name}": bank.baseline_mask for name, bank in banks.items()},
               **functional_controls, "dense": dense}
    if functional_mean is not None:
        methods["functional_mean"] = functional_mean
    if toeplitz is not None:
        methods["toeplitz"] = toeplitz
    frozen = dict(methods=methods, models=best_models, ensemble=best_evaluator,
                  banks=best_banks, protocol=asdict(protocol), best_epoch=best_epoch,
                  best_selection_delta=controller.best_worst_delta,
                  best_selection_cost=best_cost,
                  best_selection_mean_delta=controller.best_mean_delta,
                  quality_objective=config.quality_objective, best_stage=controller.best_stage,
                  training_mode=config.training_mode,
                  train_patterns=patterns,
                  test_pattern=config.test_pattern, test_patterns=tuple(config.effective_test_patterns),
                  selection_role="within-task independent queries", test_used_for_selection=False)
    if (out / "frozen.pt").exists():
        previous = torch.load(out / "frozen.pt", map_location="cpu", weights_only=False)
        if any(tensor_hash(previous["methods"][name]) != tensor_hash(mask) for name, mask in methods.items()):
            raise ValueError("immutable frozen method mismatch on resume")
    else:
        save_torch(out / "frozen.pt", frozen)
    save_json(out / "frozen.json", dict(best_epoch=best_epoch,
        selection_delta=controller.best_worst_delta, selection_cost=best_cost,
        selection_mean_delta=controller.best_mean_delta, quality_objective=config.quality_objective,
        best_stage=controller.best_stage,
        train_patterns=list(patterns), test_pattern=config.test_pattern,
        test_patterns=list(config.effective_test_patterns),
        mask_hashes={name: tensor_hash(mask) for name, mask in methods.items()}, test_used=False))
    final_generator_masks = {}
    with torch.no_grad():
        for name, model in models.items():
            model.eval()
            model.set_budget(config.k)
            trainer.set_bank(name, banks[name])
            prepared = trainer._prepared_bank(name)
            if config.training_mode == "joint":
                random_device = torch.device(own_rngs[name].device)
                latent = torch.randn((1, model.noise_dim), generator=own_rngs[name],
                                    device=random_device, dtype=torch.float32).to(prepared.tokens)
            else:
                latent = trainer.shared_latent.detach().to(prepared.tokens).reshape(1, -1)
            logits = model(prepared.tokens, latent, prepared.quality)[0]
            mask = torch.zeros_like(logits).flatten()
            mask.scatter_(0, logits.flatten().topk(config.k).indices, 1.)
            final_generator_masks[f"generator_final_{name}"] = mask.reshape_as(logits).cpu()
    save_torch(out / "final_generator_proposals.pt", dict(
        masks=final_generator_masks,
        shared_latent=(None if config.training_mode == "joint" else
                       trainer.shared_latent.detach().cpu()),
        latent_mode=("independent_random_per_generator" if config.training_mode == "joint" else
                     "learned_shared_latent"),
        bank_hashes={name: bank_input_fingerprint(bank) for name, bank in banks.items()},
        test_used=False, stage="after all generator updates, before test materialization"))
    if config.domain == "pattern":
        materialized = (test_factory or make_cooperative_test_tasks)(test_spec)
        if isinstance(materialized, TaskData):
            test_tasks = [materialized]
        elif isinstance(materialized, (list, tuple)):
            test_tasks = list(materialized)
        else:
            raise ValueError("test factory must return one TaskData or an ordered task list")
        child_specs = test_spec.get("test_specs", [test_spec])
        valid = len(test_tasks) == len(config.effective_test_patterns) and all(
            isinstance(task, TaskData) for task in test_tasks)
        if valid:
            for pattern, task, child in zip(config.effective_test_patterns, test_tasks, child_specs):
                if (not isinstance(task.provenance, dict) or
                    task.support_ids.ndim != 1 or task.query_ids.ndim != 1 or
                    task.x_support.shape != (child["support_count"], config.features) or
                    task.y_support.shape != (child["support_count"],) or
                    task.x_query.shape != (child["query_count"], config.features) or
                    task.y_query.shape != (child["query_count"],)):
                    valid = False
                    break
                expected_support = set(child["test_support_ids"])
                expected_query = set(child["test_query_ids"])
                support_ids, query_ids = set(task.support_ids.tolist()), set(task.query_ids.tolist())
                context = support_context(task.x_support, task.y_support).to(
                    device=task.context.device, dtype=task.context.dtype)
                if (task.split != "test" or task.provenance.get("family") != "pattern" or
                    task.provenance.get("pattern") != pattern or
                    task.provenance.get("role") != "sealed_test" or
                    task.provenance.get("support_partition") != "test_support" or
                    task.provenance.get("query_partition") != "test_query" or
                    task.task_id != f"pattern:{pattern}:test" or
                    len(task.support_ids) != child["support_count"] or
                    len(task.query_ids) != child["query_count"] or
                    len(support_ids) != len(task.support_ids) or
                    len(query_ids) != len(task.query_ids) or
                    not support_ids.issubset(expected_support) or
                    not query_ids.issubset(expected_query) or
                    not support_ids.isdisjoint(query_ids) or
                    not torch.equal(task.context, context)):
                    valid = False
                    break
        if not valid:
            raise ValueError("test factory violated the frozen third-pattern role or sealed-pattern roles")
    else:
        from generator_evaluator.data.deepsets import make_cooperative_deepsets_test_tasks, validate_deepsets_test_tasks
        test_tasks = (test_factory or make_cooperative_deepsets_test_tasks)(test_spec)
        validate_deepsets_test_tasks(test_tasks, test_spec, train_tasks, selection_tasks)
    save_torch(out / "test_tasks.pt", test_tasks)
    reports, examples = {}, []
    for task in progress(selection_tasks + test_tasks, desc="Frozen final evaluation", unit="task"):
        measured = dict(zip(methods, measure_many([(mask, task, f"final:{name}") for name, mask in methods.items()])))
        dense_losses = measured["dense"][0]["replica_losses"]
        reports[task.task_id] = {
            name: dict(**{("query_bce" if config.domain == "pattern" else "query_nmse"):row["quality"]},
                       query_error=row["quality"], replica_losses=row["replica_losses"],
                       comparison=paired_comparison(row["replica_losses"], dense_losses),
                       plateau_flags=row["plateau_flags"], artifact_path=row["artifact_path"])
            for name, (row, _) in measured.items()}
        if task is test_tasks[0]:
            examples = [dict(method=name, mask=methods[name], effective_weights=result["effective_weights"],
                             history=result["history"]) for name, (_, result) in measured.items()]
    summary = dict(domain=config.domain, phase=config.phase,
        train_patterns=list(patterns), test_pattern=config.test_pattern,
        test_patterns=list(config.effective_test_patterns),
        preset=config.preset, training_mode=config.training_mode,
        generators=len(models), global_evaluator_members=config.ensemble_members,
        best_epoch=best_epoch, best_stage=controller.best_stage,
        best_selection_delta=controller.best_worst_delta, best_selection_cost=best_cost,
        best_selection_mean_delta=controller.best_mean_delta,
        quality_objective=config.quality_objective, protocol_id=protocol.fingerprint,
        common_train_elites=len(controller.common_elites()), smoke_only=config.smoke,
        bank_sizes={name: bank.tokens.shape[1] for name, bank in banks.items()},
        bank_densities={name: sorted(set(bank.masks.sum((1, 2)).long().tolist())) for name, bank in banks.items()},
        real_label_measurements=len(replay.records), refreshes=refresh_history, final=reports,
        structural_prior=None if toeplitz is None else SlidingWindowMaskPrior().diagnostics(best_mask),
        uncertainty="paired fresh initializations conditional on fixed observations; no task-population claim",
        selection_role="independent query observations of all train tasks",
        test_used_for_training=False,
        generator_pretraining_updates=len(controller.pretraining_history),
        generator_pretraining=controller.generator_pretraining,
        quality_updates=sum(row["stage"] in (STAGE_QUALITY, STAGE_JOINT) for row in history),
        joint_updates=sum(row["stage"] == STAGE_JOINT for row in history),
        cooperation_updates=sum(row["stage"] == STAGE_COOPERATION for row in history),
        completed_stage=controller.stage,
        agreement="matched-noise exact-K MSE agreement; archive-backed measured TRAIN MSE distillation")
    summary["selected_functional_control"] = next((name for name, mask in functional_controls.items()
        if topology_id(mask) == topology_id(best_mask)), None)
    summary["functional_control_priors"] = (None if not functional_controls else dict(
        normalization="unit column mass; equal-bank average",
        alignment="ordered input-coordinate centroids",
        balanced_degree="floor(K/H) with remainder allocated by importance",
        analytical_structure_target_used=False))
    summary["final_generators_exact_agreement"] = len({topology_id(mask)
        for mask in final_generator_masks.values()}) == 1
    if toeplitz is not None:
        summary["final_generator_structure"] = {
            name: SlidingWindowMaskPrior().diagnostics(mask)
            for name, mask in final_generator_masks.items()}
    summary["test_task_ids"] = [task.task_id for task in test_tasks]
    summary["evaluator_task_ids"] = {task.task_id: task.provenance.get("evaluator_task_id")
                                     for task in train_tasks}
    summary["improves_every_test_task"] = all(reports[task.task_id]["common"]["comparison"]["point_improvement"]
                                              for task in test_tasks)
    summary["interval_improvement_every_test_task"] = all(reports[task.task_id]["common"]["comparison"]["interval_below_zero"]
                                                       for task in test_tasks)
    replay.save(out / "replay.pt")
    save_torch(out / "final_banks.pt", banks)
    from generator_evaluator.storage.final_report import write_final_report
    summary["final_report"] = write_final_report(out, summary, methods, examples)
    save_json(out / "summary.json", summary)
    write_plots(out, history, examples, evaluator_history=evaluator_history,
                calibration=calibration, within_task_selection=True,
                generator_pretraining=controller.pretraining_history)
    if config.domain == "pattern" and config.k == 32:
        write_toeplitz_report(out, {**methods, **final_generator_masks})
    captions = out / "figures" / "CAPTIONS_RU.md"
    captions.parent.mkdir(exist_ok=True)
    contents = captions.read_text() if captions.exists() else ""
    contents = contents.replace("худшая по задачам", "по собственной задаче в quality и худшая по train-задачам в cooperation")
    contents = contents.replace("новые задачи и их совместный holdout", "отдельные selection-наблюдения train-задач и пересечение с holdout топологий")
    contents = contents.replace("новые задачи", "selection-наблюдения тех же паттернов")
    if config.training_mode == "joint":
        contents += (f"\nГенераторов: {len(models)}. Каждый update использует сумму own-task "
                     "quality, MSE agreement на общих случайных latent draws и дистилляции "
                     "реальных train-архивных масок. Реальная обратная связь "
                     "поступает по расписанию critic refresh. Heldout-задачи используются "
                     "только после фиксации общей маски.\n")
    else:
        contents += (f"\nГенераторов: {len(models)}. Quality обучает каждого на своей задаче; "
                     "cooperation объединяет качество по train-задачам, agreement и реальную "
                     "обратную связь. Heldout-задачи используются только после фиксации общей маски.\n")
    captions.write_text(contents)
    (out / "COMPLETE").write_text("complete\n")
    return summary


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=("pattern", "deepsets"), default="pattern")
    parser.add_argument("--preset", choices=("pattern-small", "full", "deepsets"), default="pattern-small")
    parser.add_argument("--train-patterns", nargs="+")
    parser.add_argument("--train-task-count", type=int, default=2,
                        help="DeepSets own-task generators (at least 2)")
    parser.add_argument("--test-task-count", type=int, default=2,
                        help="DeepSets sealed heldout cost functions (at least 1)")
    heldout = parser.add_mutually_exclusive_group()
    heldout.add_argument("--test-pattern")
    heldout.add_argument("--test-patterns", nargs="+")
    parser.add_argument("--seed", type=int, default=4100)
    parser.add_argument("--training-mode", choices=("joint", "staged"), default="joint",
                        help="joint interleaved updates by default; staged retains quality then cooperation")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--replicas", type=int, default=None)
    parser.add_argument("--k", type=int)
    parser.add_argument("--hidden", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--l2", type=float)
    parser.add_argument("--output-budgets", type=int, nargs="*")
    parser.add_argument("--agreement-weight", "--lambda-agreement", dest="agreement_weight", type=float)
    parser.add_argument("--reconstruction-weight", type=float)
    parser.add_argument("--quality-objective", choices=QUALITY_OBJECTIVES,
                        help="Across-task cost used for policy, acquisition, feedback and selection")
    parser.add_argument("--elite-distillation-weight", "--mu-distill",
                        dest="elite_distillation_weight", type=float)
    parser.add_argument("--bootstrap-generators", action="store_true",
                        help="enable iterative generator exploration during bootstrap")
    parser.add_argument("--bootstrap-only", action="store_true",
                        help="Prepare banks and critic without opening heldout test tasks")
    parser.add_argument("--data-root", type=Path, default=Path("datasets/mnist8m"))
    parser.add_argument("--measurement-devices", nargs="+",
                        help="Child-fit devices; auto uses all CUDA GPUs visible to this process")
    parser.add_argument("--measurement-batch-size", type=int, default=8)
    parser.add_argument("--generator-devices", nargs="+", default=None,
                        help="Generator devices; auto assigns generators round-robin across visible CUDA GPUs")
    parser.add_argument("--bank-support-count", type=int, default=None,
                        help="DeepSets source support sets; default uses the full private pool budget")
    parser.add_argument("--bank-query-count", type=int, default=None,
                        help="DeepSets source query sets; default uses the full private validation pool budget")
    parser.add_argument("--latent-lr", type=float, default=None)
    parser.add_argument("--evaluator-lr", type=float, default=None)
    for name in ("steps", "bank-steps", "teachers", "bank-candidates", "teacher-batch-size",
                 "support-count", "query-count", "selection-count",
                 "probe-count", "generator-epochs", "updates-per-epoch", "cooperation-rounds",
                 "cooperation-updates", "refresh-every", "minimum-refresh-every",
                 "evaluator-epochs", "evaluator-batch-size", "candidates", "acquisition-budget", "initial-random",
                 "bank-capacity", "width", "heads", "layers", "noise-dim", "ensemble-members",
                 "auxiliary-budget", "feedback-masks", "elite-limit",
                 "agreement-ramp-epochs", "generator-pretrain-epochs", "pretrain-updates-per-epoch",
                 "reconstruction-batch-size"):
        parser.add_argument(f"--{name}", type=int, default=None)
    parser.epilog = ("pattern-small: 500 steps x 2 fresh initializations; initial bank teachers "
                     "use 200 steps x 1 initialization; 100 candidates per density, keep 10 "
                     "at each of 10 densities. Two pattern generators, width 16, "
                     "2 heads, 1 layer, probe 32. All independent child fits run in batches.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--warm-start-from", type=Path,
                        help="Reuse live banks, measurements and latest critic; initialize new generators")
    parser.add_argument("--fixed-test-from", type=Path,
                        help="DeepSets: retain sealed test costs and image pools from a prior run directory")
    parser.add_argument("--generator-pretrained-from", type=Path,
                        help="Import heldout-safe generator weights from generator_reconstruction.pt")
    parser.add_argument("--smoke", action="store_true")
    flags = parser.add_mutually_exclusive_group()
    flags.add_argument("--progress", action="store_true")
    flags.add_argument("--no-progress", action="store_true")
    return parser


def resolve_run_settings(args):
    is_deepsets = args.domain == "deepsets" or args.preset == "deepsets"
    if is_deepsets:
        args.domain, args.preset = "deepsets", "deepsets"
    base = (deepsets_config() if is_deepsets else
            pattern_small_config() if args.preset == "pattern-small" else CooperativeConfig())
    if is_deepsets and args.test_patterns is not None:
        raise ValueError("DeepSets uses --test-task-count rather than --test-patterns")
    if is_deepsets:
        requested_test_patterns = ()
        requested_test_pattern = args.test_pattern or base.test_pattern
    elif args.test_patterns is not None:
        requested_test_patterns = tuple(args.test_patterns)
        requested_test_pattern = requested_test_patterns[0] if requested_test_patterns else ""
    elif args.test_pattern is not None:
        requested_test_patterns = (args.test_pattern,)
        requested_test_pattern = args.test_pattern
    else:
        requested_test_patterns = tuple(base.effective_test_patterns)
        requested_test_pattern = requested_test_patterns[0] if requested_test_patterns else ""
    if not is_deepsets and not requested_test_patterns:
        raise ValueError("pattern requires at least one held-out test pattern")
    overrides = dict(seed=args.seed, training_mode=args.training_mode,
                     train_patterns=tuple(args.train_patterns or base.train_patterns),
                     test_pattern=requested_test_pattern,
                     test_patterns=requested_test_patterns,
                     k=base.k if args.k is None else args.k, smoke=args.smoke,
                     phase="bootstrap" if args.bootstrap_only else "search",
                     bootstrap_generators=args.bootstrap_generators)
    if args.latent_lr is not None:
        overrides["latent_lr"] = args.latent_lr
    if args.evaluator_lr is not None:
        overrides["evaluator_lr"] = args.evaluator_lr
    if is_deepsets:
        if args.train_patterns is not None:
            raise ValueError("DeepSets uses --train-task-count rather than --train-patterns")
        overrides.update(train_patterns=tuple(str(index) for index in range(args.train_task_count)),
                         test_task_count=args.test_task_count)
    elif (args.train_task_count, args.test_task_count) != (2, 2):
        raise ValueError("task-count options apply only to DeepSets")
    for name in ("hidden", "agreement_weight", "reconstruction_weight", "elite_distillation_weight",
                 "output_budgets"):
        if getattr(args, name) is not None:
            overrides[name] = tuple(args.output_budgets) if name == "output_budgets" else getattr(args, name)
    if is_deepsets and args.hidden is not None:
        edges = 784 * args.hidden
        if args.k is None:
            overrides["k"] = round(.3 * edges)
        if args.output_budgets is None and base.output_budgets:
            overrides["output_budgets"] = tuple(round(edges * density) for density in (.1, .5, .7))
    for name in ("generator_epochs", "updates_per_epoch", "cooperation_rounds", "cooperation_updates",
                 "refresh_every", "minimum_refresh_every", "evaluator_epochs",
                 "evaluator_batch_size",
                 "candidates", "acquisition_budget", "initial_random", "bank_capacity",
                 "width", "heads", "layers", "noise_dim", "ensemble_members",
                 "auxiliary_budget", "feedback_masks", "elite_limit", "agreement_ramp_epochs",
                 "generator_pretrain_epochs", "pretrain_updates_per_epoch", "reconstruction_batch_size"):
        if getattr(args, name) is not None:
            overrides[name] = getattr(args, name)
    if args.quality_objective is not None:
        overrides["quality_objective"] = args.quality_objective
    config = replace(base, **overrides)
    build = (dict(steps=2000, bank_steps=4000, teachers=1024, bank_candidates=4096,
                  teacher_batch_size=64, support_count=500, query_count=51,
                  selection_count=51, probe_count=32) if is_deepsets else
             dict(steps=500, bank_steps=200, teachers=100, bank_candidates=1000,
                  teacher_batch_size=128, support_count=208, query_count=64,
                  selection_count=32, probe_count=32) if args.preset == "pattern-small" else
             dict(steps=2000, bank_steps=2000, teachers=100, bank_candidates=1000,
                  teacher_batch_size=128, support_count=208, query_count=128,
                  selection_count=64, probe_count=128))
    if (is_deepsets and not args.smoke and
            any(getattr(args, name) is None for name in
                ("support_count", "query_count", "selection_count"))):
        source_run = args.warm_start_from or args.fixed_test_from
        if source_run is not None:
            source_spec = json.loads((source_run / "run_spec.json").read_text())
            for name in ("support_count", "query_count", "selection_count"):
                if getattr(args, name) is None:
                    build[name] = int(source_spec["test_spec"].get(
                        name, source_spec.get("build_settings", {}).get(name, build[name])))
        else:
            from deepsets_vaae.core import load_data
            data = load_data(args.data_root, args.seed, "cpu")
            if args.support_count is None:
                build["support_count"] = len(data["target_train"].features) // (
                    5 * (args.train_task_count + args.test_task_count))
            query_budget = len(data["target_validation"].features) // (5 * args.train_task_count)
            if args.query_count is None and args.selection_count is None:
                build["query_count"] = build["selection_count"] = min(51, query_budget // 2)
            elif args.query_count is None:
                build["query_count"] = min(51, query_budget - args.selection_count)
            elif args.selection_count is None:
                build["selection_count"] = min(51, query_budget - args.query_count)
    if is_deepsets:
        build.update(bank_support_count=None, bank_query_count=None)
    elif args.bank_support_count is not None or args.bank_query_count is not None:
        raise ValueError("bank support/query count overrides apply only to DeepSets")
    for name, default in build.items():
        if getattr(args, name) is None:
            setattr(args, name, default)
    if args.smoke:
        config = replace(config, generator_epochs=2, updates_per_epoch=2,
             cooperation_rounds=1, cooperation_updates=1, refresh_every=1,
             minimum_refresh_every=1,
             evaluator_epochs=2, candidates=4, acquisition_budget=3, initial_random=2,
             auxiliary_budget=0 if not config.output_budgets else 1,
             feedback_masks=1, width=16, layers=1, ensemble_members=2)
        config = replace(config, pretrain_updates_per_epoch=2,
                         reconstruction_batch_size=2)
        args.steps, args.bank_steps, args.teachers = 3, 2, 5
        args.bank_candidates = 5
        args.support_count, args.query_count, args.selection_count = 8, 8, 8
        if is_deepsets:
            args.bank_support_count, args.bank_query_count = 8, 8
    replicas = args.replicas if args.replicas is not None else (4 if is_deepsets else 2)
    protocol = InnerProtocol(steps=args.steps, replicas=replicas, seed=args.seed,
                             lr=args.lr if args.lr is not None else (.005 if is_deepsets else .01),
                             l2=args.l2 if args.l2 is not None else (.0001 if is_deepsets else 0.),
                             metric="nmse" if is_deepsets else "bce",
                             checkpoint_every=max(1, min(25, args.steps // 4)))
    if not is_deepsets and protocol.replicas < 2:
        raise ValueError("pattern comparisons require at least two independent initializations")
    build_settings = {name: getattr(args, name) for name in build if name != "steps"}
    if args.bank_candidates < args.teachers:
        raise ValueError("bank-candidates must be at least teachers")
    if config.bank_capacity < args.teachers:
        config = replace(config, bank_capacity=args.teachers)
    if is_deepsets:
        build_settings["data_root"] = str(args.data_root.resolve())
        build_settings["hidden"] = config.hidden
        build_settings["train_task_count"] = len(config.train_patterns)
        build_settings["test_task_count"] = config.test_task_count
    return config, protocol, build_settings


def main():
    parser = make_parser()
    args = parser.parse_args()
    if args.progress or args.no_progress:
        os.environ["GENERATOR_EVALUATOR_PROGRESS"] = "1" if args.progress else "0"
    torch.set_num_threads(1)
    config, protocol, build_settings = resolve_run_settings(args)
    fixed_test_spec = None
    if args.fixed_test_from is not None:
        if config.domain != "deepsets":
            parser.error("--fixed-test-from is only valid for DeepSets")
        source_path = args.fixed_test_from.resolve() / "run_spec.json"
        try:
            fixed_test_spec = json.loads(source_path.read_text())["test_spec"]
        except (OSError, ValueError, KeyError) as error:
            parser.error(f"cannot read sealed test specification from {source_path}: {error}")
        build_settings["fixed_test_spec_hash"] = hashlib.sha256(
            json.dumps(fixed_test_spec, sort_keys=True).encode()).hexdigest()
    if args.generator_pretrained_from is not None and config.phase != "search":
        parser.error("--generator-pretrained-from is only valid for search runs")
    measurement_devices = args.measurement_devices
    if measurement_devices == ["auto"]:
        measurement_devices = [f"cuda:{index}" for index in range(torch.cuda.device_count())]
        if not measurement_devices:
            parser.error("--measurement-devices auto requires visible CUDA GPUs")
    if args.measurement_batch_size < 1:
        parser.error("--measurement-batch-size must be positive")
    if measurement_devices:
        for value in measurement_devices:
            try:
                parsed = torch.device(value)
            except (ValueError, RuntimeError):
                parser.error(f"invalid measurement device: {value}")
            if parsed.type not in ("cpu", "cuda") or (parsed.type == "cuda" and
                    (not torch.cuda.is_available() or (parsed.index or 0) >= torch.cuda.device_count())):
                parser.error(f"measurement device unavailable: {value}")
    generator_devices = args.generator_devices
    if generator_devices == ["auto"]:
        generator_devices = [f"cuda:{index}" for index in range(torch.cuda.device_count())]
        if not generator_devices:
            generator_devices = [args.device]
    try:
        generator_devices = list(_device_list(generator_devices, args.device,
                                               len(config.train_patterns), label="generator"))
    except ValueError as error:
        parser.error(str(error))
    warm_start = None
    if args.warm_start_from is not None:
        warm_start = load_cooperative_warm_start(args.warm_start_from, config, protocol)
        if args.out.resolve() == args.warm_start_from.resolve():
            parser.error("warm-start requires a new --out")
    training_schedule = (f"joint={config.generator_epochs}×{config.updates_per_epoch}"
                         if config.training_mode == "joint" and config.phase == "search" else
                         f"quality={config.generator_epochs}×{config.updates_per_epoch}, "
                         f"cooperation={config.cooperation_rounds}×{config.cooperation_updates}")
    print(f"Preset={config.preset}: child={protocol.steps}×{protocol.replicas}, "
          f"bank={args.bank_candidates}→{args.teachers}×{args.bank_steps}, width={config.width}, "
          f"heads={config.heads}, layers={config.layers}, probe={args.probe_count}, "
          f"training_mode={config.training_mode} {training_schedule}, "
          f"latent_lr={config.latent_lr:g}, generators={generator_devices}, "
          f"batched={config.batch_children}", flush=True)
    inputs_path = args.out / "inputs.pt"
    if warm_start is not None:
        banks, train, selection, sealed = (warm_start.banks, warm_start.train_tasks,
                                          warm_start.selection_tasks, warm_start.test_spec)
        print(f"Warm start: {args.warm_start_from.resolve()}; banks reused, "
              "latest critic loaded, new generators initialized", flush=True)
    elif args.resume and inputs_path.exists():
        inputs = torch.load(inputs_path, map_location="cpu", weights_only=False)
        banks, train, selection, sealed = (inputs[key] for key in
                                          ("banks", "train_tasks", "selection_tasks", "test_spec"))
        stored = json.loads((args.out / "run_spec.json").read_text())
        build = stored.get("build_settings")
        # Older programmatic fixtures have no CLI build settings.
        if build is not None and build != build_settings:
            parser.error("resume bank/data arguments differ from saved run")
    else:
        if config.domain == "deepsets":
            from generator_evaluator.data.deepsets import build_cooperative_deepsets_fixture
            banks, train, selection, sealed = build_cooperative_deepsets_fixture(
                args.data_root, seed=args.seed, bank_steps=args.bank_steps,
                teachers_per_task=args.teachers, bank_candidates=args.bank_candidates,
                teacher_batch_size=args.teacher_batch_size, support_count=args.support_count,
                query_count=args.query_count, selection_count=args.selection_count,
                bank_support_count=args.bank_support_count, bank_query_count=args.bank_query_count,
                probe_count=args.probe_count, k=config.k, hidden=config.hidden,
                train_task_count=len(config.train_patterns), test_task_count=config.test_task_count,
                device=args.device, measurement_devices=measurement_devices,
                out=args.out / "bank_build", fixed_test_spec=fixed_test_spec)
        else:
            banks, train, selection, sealed = build_cooperative_fixture(
            config.train_patterns,
            (config.effective_test_patterns if len(config.effective_test_patterns) > 1
             else config.effective_test_patterns[0]),
            args.seed, args.bank_steps,
            args.teachers, args.support_count, args.query_count, args.selection_count,
            config.k, args.device, probe_count=args.probe_count, batch_teachers=config.batch_children,
            bank_candidates=args.bank_candidates, teacher_batch_size=args.teacher_batch_size,
            measurement_devices=measurement_devices)
            # Preserve the expensive source fits even if a later runner
            # validation or initialization fails before the first checkpoint.
            save_torch(args.out / "prepared_fixture.pt", dict(
                banks=banks, train_tasks=train, selection_tasks=selection,
                test_spec=sealed, build_settings=build_settings,
                config=_config_metadata(config), protocol=asdict(protocol)))
    if fixed_test_spec is not None and any(sealed.get(key) != fixed_test_spec.get(key) for key in
            ("seed", "data_root", "support_count", "query_count", "test_task_count", "costs",
             "test_support_pools", "test_query_pools")):
        parser.error("prepared or warm-start inputs do not retain the requested fixed test specification")
    result = run_cooperative_experiment(banks, train, selection, sealed, args.out,
              protocol, config, device=args.device, resume=args.resume,
              dense_learning_rates=[protocol.lr] if args.smoke else None,
              build_settings=build_settings, warm_start=warm_start,
              measurement_devices=measurement_devices, measurement_batch_size=args.measurement_batch_size,
              generator_devices=generator_devices,
              generator_pretrained_from=args.generator_pretrained_from)
    if config.phase == "bootstrap":
        print(json.dumps(dict(out=str(args.out.resolve()), phase="bootstrap",
              generators=result["generators"], real_label_measurements=result["real_label_measurements"],
              train_patterns=result["train_patterns"], test_patterns=result["test_patterns"],
              test_materialized=False), indent=2))
        return
    test_deltas = {task_id: result["final"][task_id]["common"]["comparison"]["delta"]
                   for task_id in result["test_task_ids"]}
    reported_test_delta = (next(iter(test_deltas.values()))
                           if config.domain == "pattern" and len(test_deltas) == 1 else test_deltas)
    print(json.dumps(dict(out=str(args.out.resolve()), train_patterns=result["train_patterns"],
          test_pattern=result["test_pattern"], test_patterns=result["test_patterns"],
          best_selection_delta=result["best_selection_delta"],
          test_delta_vs_dense=reported_test_delta,
          qualified_common_train_elites=result["common_train_elites"],
          real_label_measurements=result["real_label_measurements"], smoke_only=result["smoke_only"]),
          ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
