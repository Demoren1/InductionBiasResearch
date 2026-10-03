"""Small shared runtime primitives for reproducible experiment runners."""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import shutil
from typing import Callable, Iterable

import torch

from generator_evaluator.storage.artifacts import save_json, save_torch
from generator_evaluator.data.types import InnerProtocol, TaskData


def cpu_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def random_exact_k(count: int, features: int, hidden: int, k: int,
                   rng: torch.Generator) -> torch.Tensor:
    if min(count, features, hidden, k) < 1 or k > features * hidden:
        raise ValueError("exact-K mask dimensions are invalid")
    # Generate on the generator's device.  This avoids implicit CPU generator
    # use when the policy sampler runs on CUDA.
    generator_device = torch.device(rng.device)
    keys = torch.rand(count, features * hidden, generator=rng, device=generator_device)
    indices = keys.topk(k, dim=-1).indices
    return torch.zeros_like(keys).scatter(-1, indices, 1).reshape(count, features, hidden)


# Transitional aliases let existing runners migrate one import at a time.
_cpu_state = cpu_state
_random_masks = random_exact_k


class RunSession:
    """Own the immutable run spec, source snapshot, inputs, and checkpoints."""

    def __init__(self, out: str | Path, run_spec: dict, source_files: Iterable[str | Path], *,
                 project: str | Path | None = None, resume: bool = False,
                 save_torch_fn: Callable | None = None, save_json_fn: Callable | None = None):
        self.out = Path(out).resolve()
        self.spec = json.loads(json.dumps(run_spec))
        self.source_files = [Path(path).resolve() for path in source_files]
        self.project = Path(project).resolve() if project is not None else Path(__file__).resolve().parents[2]
        self.resume = resume
        # Runner-local callbacks preserve existing test seams that patch their
        # ``save_torch`` symbol to simulate an interruption.
        self._save_torch = save_torch_fn or save_torch
        self._save_json = save_json_fn or save_json

    @staticmethod
    def source_hashes(source_files: Iterable[str | Path], project: str | Path) -> dict[str, str]:
        root = Path(project).resolve()
        hashes = {}
        for raw_path in source_files:
            path = Path(raw_path).resolve()
            if not path.is_file():
                raise ValueError(f"source snapshot file does not exist: {path}")
            try:
                name = str(path.relative_to(root))
            except ValueError as error:
                raise ValueError(f"source snapshot file is outside project: {path}") from error
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return dict(sorted(hashes.items()))

    def prepare(self, *, inputs=None) -> dict | None:
        """Create a run once, or validate a restart before any fitting begins."""
        self.out.mkdir(parents=True, exist_ok=True)
        spec_path = self.out / "run_spec.json"
        if spec_path.exists():
            saved = json.loads(spec_path.read_text(encoding="utf-8"))
            if saved != self.spec:
                raise ValueError("existing output has different inputs, code, or settings; choose new --out")
            complete = self.out / "COMPLETE"
            if complete.exists():
                summary = self.out / "summary.json"
                return json.loads(summary.read_text(encoding="utf-8")) if summary.exists() else None
            if not self.resume:
                raise ValueError("unfinished output exists; use --resume or a new --out")
            return None
        self._save_json(spec_path, self.spec)
        if inputs is not None:
            self._save_torch(self.out / "inputs.pt", inputs)
        for source in self.source_files:
            destination = self.out / "source_snapshot" / source.relative_to(self.project)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        return None

    def save_protocol(self, protocol: InnerProtocol, **metadata) -> None:
        self._save_json(self.out / "protocol.json", dict(inner_protocol=asdict(protocol),
                       protocol_id=protocol.fingerprint, **metadata))

    def load_protocol(self) -> InnerProtocol:
        payload = json.loads((self.out / "protocol.json").read_text(encoding="utf-8"))
        protocol = InnerProtocol(**payload["inner_protocol"])
        if payload.get("protocol_id", protocol.fingerprint) != protocol.fingerprint:
            raise ValueError("saved protocol fingerprint is corrupt")
        return protocol

    def save_checkpoint(self, state: dict, *, replay=None, history=None, name: str = "checkpoint.pt") -> None:
        """Persist recoverable side artifacts before atomically replacing checkpoint."""
        if replay is not None:
            replay.save(self.out / "replay.pt")
        if history is not None:
            self._save_json(self.out / "history.json", history)
        self._save_torch(self.out / name, state)

    def load_checkpoint(self, *, name: str = "checkpoint.pt", map_location="cpu") -> dict:
        path = self.out / name
        if not path.is_file():
            raise FileNotFoundError(path)
        return torch.load(path, map_location=map_location, weights_only=False)


class DenseProtocolSelector:
    """Select a dense solver on a designated validation/selection split."""

    def __init__(self, out: str | Path, measure_fn: Callable | None = None, *,
                 save_torch_fn: Callable | None = None, save_json_fn: Callable | None = None,
                 measure_many_fn: Callable | None = None):
        self.out = Path(out)
        if measure_fn is None:
            from generator_evaluator.data.adapters import measure_mask
            measure_fn = measure_mask
        self.measure_fn = measure_fn
        self.measure_many_fn = measure_many_fn
        self._save_torch = save_torch_fn or save_torch
        self._save_json = save_json_fn or save_json

    def select(self, tasks: Iterable[TaskData], mask: torch.Tensor, protocol: InnerProtocol,
               device: str, learning_rates: Iterable[float], *,
               selection_label: str = "meta-validation") -> InnerProtocol:
        validation = [task for task in tasks if task.split == "validation"]
        if not validation:
            raise ValueError("dense tuning requires independent validation tasks")
        rates = [float(rate) for rate in learning_rates]
        if not rates or any(rate <= 0 for rate in rates):
            raise ValueError("dense tuning learning rates must be positive")
        rows = []
        for rate in rates:
            setting = replace(protocol, lr=rate)
            scores = []
            results = (self.measure_many_fn(mask, validation, setting, device)
                       if self.measure_many_fn is not None else
                       [self.measure_fn(mask, task, setting, device=device) for task in validation])
            if len(results) != len(validation):
                raise ValueError("dense fitter returned an invalid result count")
            for task, result in zip(validation, results):
                scores.append(float(torch.as_tensor(result["replica_losses"]).mean()))
                safe_id = task.task_id.replace(":", "_")
                self._save_torch(self.out / "dense_tuning" / f"{setting.fingerprint}_{safe_id}.pt", result)
            rows.append(dict(lr=rate, mean_query_error=sum(scores) / len(scores),
                             per_task_query_errors=scores, protocol_id=setting.fingerprint))
        selected = min(rows, key=lambda row: row["mean_query_error"])
        self._save_json(self.out / "dense_tuning.json", dict(settings=rows, selected=selected,
                       task_ids=[task.task_id for task in validation], selection_split=selection_label,
                       test_used=False, fixed_solver_for_all_methods=True,
                       label_measurements=len(rows) * len(validation)))
        return replace(protocol, lr=selected["lr"])
