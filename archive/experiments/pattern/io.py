"""Portable artifacts and exclusive experiment directories."""
import hashlib
import json
from pathlib import Path
import tempfile
from datetime import datetime, timezone
import uuid
import os
import torch

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def new_directory(parent, name=None):
    name = name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    if Path(name).name != name or name in (".", ".."):
        raise ValueError("identifier must be a single directory name")
    path = Path(parent) / name
    path.mkdir(parents=True, exist_ok=False)
    return path

def atomic(path, writer, binary):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "wb" if binary else "w") as stream:
            writer(stream)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

def save_json(path, payload):
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    atomic(path, lambda stream: stream.write(text), False)

def save_torch(path, payload):
    atomic(path, lambda stream: torch.save(payload, stream), True)

def load_run(run):
    """Load a checkpoint only with its unchanged source-bank manifest."""
    checkpoint=torch.load(Path(run)/"checkpoints/best.pt",weights_only=True,map_location="cpu")
    reference=checkpoint["bank_reference"]; bank=Path(reference["bank_path"])
    if digest(bank/"manifest.json") != reference["manifest_sha256"]:
        raise ValueError("bank manifest changed after training")
    return checkpoint,bank,json.loads((bank/"manifest.json").read_text())

def load_config(path):
    config = json.loads(Path(path).read_text())
    if config.get("k") != 32:
        raise ValueError("this experiment requires exactly 32 active first-layer connections")
    tasks = config.get("tasks", [])
    if not tasks or len(tasks) != len(set(tasks)) or any(len(t) != 4 or set(t)-set("01") for t in tasks):
        raise ValueError("tasks must be unique four-bit patterns")
    test_tasks = config.get("test_tasks", [])
    if len(test_tasks) != len(set(test_tasks)) or set(tasks) & set(test_tasks) or any(len(t) != 4 or set(t)-set("01") for t in test_tasks):
        raise ValueError("test_tasks must be unique four-bit patterns disjoint from tasks")
    bank, model, train = config["bank"], config["model"], config["training"]
    for key in ("maps_per_task", "steps_per_round", "support_count", "query_count", "probe_count"):
        if bank[key] < 1:
            raise ValueError(f"bank.{key} must be positive")
    if bank["maps_per_task"] < 4 or bank["probe_count"] > 128:
        raise ValueError("need maps_per_task>=4 and probe_count<=128")
    if not 0 < bank["prune_fraction"] < 1 or bank["lr"] <= 0 or bank["l2"] < 0:
        raise ValueError("invalid IMP hyperparameters")
    if any(model[key] < 1 for key in ("nf_channels", "latent_dim", "encoder_width", "decoder_width")):
        raise ValueError("model dimensions must be positive")
    if train["epochs"] < 1 or train["batch_size"] < 1 or train["lr"] <= 0 or train["beta"] < 0:
        raise ValueError("invalid training hyperparameters")
    if train["kl_warmup_epochs"] < 0 or not 0 <= train["hard_loss_weight"] <= 1:
        raise ValueError("invalid KL warmup or hard loss weight")
    return config

def device_name(value):
    if value == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type not in ("cpu", "cuda", "mps"):
        raise ValueError("supported devices: auto, mps, cpu, cuda:N")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable in this PyTorch environment; use --device cpu or install an MPS-enabled build")
    if device.type == "cuda" and (not torch.cuda.is_available() or (device.index or 0) >= torch.cuda.device_count()):
        raise ValueError(f"device unavailable: {value}")
    return str(device)
