"""Atomic artifacts, immutable provenance and experiment configuration."""

import hashlib
import json
import shutil
from pathlib import Path

import torch

PACKAGE = Path(__file__).resolve().parent
REPOSITORY = PACKAGE.parent
DEFAULT_BANK = REPOSITORY / "data/overlapping_groups/banks/wg_v2"
BANK_SCHEMA = "overlapping-groups.wg.v2"
LEGACY_SCHEMA = "overlapping-groups.w-g-agop.v1"


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def save_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(value, temporary)
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_load(path, expected=None):
    if expected is not None and sha256(path) != expected:
        raise ValueError(f"File hash mismatch: {path}")
    return torch.load(path, map_location="cpu", weights_only=True)


def load_manifest(bank):
    manifest = json.loads((Path(bank) / "manifest.json").read_text())
    if manifest["status"] != "complete" or manifest["schema"] not in (
        BANK_SCHEMA,
        LEGACY_SCHEMA,
    ):
        raise ValueError("A complete W/G bank is required")
    return manifest


def source_hashes():
    return {
        str(path.relative_to(PACKAGE)): sha256(path)
        for path in sorted(PACKAGE.rglob("*.py"))
        if "runs" not in path.relative_to(PACKAGE).parts
    }


def snapshot(destination, config=None):
    destination = Path(destination)
    for name in source_hashes():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PACKAGE / name, target)
    if config is not None:
        shutil.copyfile(config, destination / "used_config.json")


def validate_config(config):
    options, evaluation = config["search"], config["evaluation"]
    for key in (
        "starts",
        "steps",
        "inner_steps",
        "replicas",
        "gumbel_samples",
        "hard_batch",
    ):
        if not isinstance(options[key], int) or options[key] < 1:
            raise ValueError(f"search.{key} must be a positive integer")
    if not 0 <= options["inner_warmup_steps"] < options["inner_steps"]:
        raise ValueError(
            "Inner warmup must end before the last optimizer step"
        )
    if options.get("inner_optimizer", "sgd") not in ("sgd", "adam"):
        raise ValueError("Unknown inner optimizer")
    for key in ("inner_lr", "z_lr", "z_clip_norm"):
        if options[key] <= 0:
            raise ValueError(f"search.{key} must be positive")
    if (
        options.get("inner_adam_eps", 1e-6) <= 0
        or options.get("gradient_checkpoint_steps", 32) < 1
    ):
        raise ValueError("Invalid inner optimizer stability settings")
    if options["l2"] < 0 or options["negative_slope"] < 0:
        raise ValueError("Invalid inner regularization")
    for key in ("temperature", "gumbel_noise", "evolution_radius"):
        if len(options[key]) != 2 or min(options[key]) <= 0:
            raise ValueError(f"Invalid schedule: {key}")
    if not options["proposal_scales"] or min(options["proposal_scales"]) <= 0:
        raise ValueError("Proposal scales must be positive")
    if evaluation["functions"] != [6, 7]:
        raise ValueError("Adaptation uses functions 4/5; evaluation uses 6/7")
    for key in ("steps", "replicas", "select_every"):
        if not isinstance(evaluation[key], int) or evaluation[key] < 1:
            raise ValueError(f"evaluation.{key} must be a positive integer")
    if (
        not 0 <= evaluation["warmup_steps"] < evaluation["steps"]
        or evaluation["lr"] <= 0
        or evaluation["l2"] < 0
        or evaluation["negative_slope"] < 0
    ):
        raise ValueError("Invalid final MLP optimization settings")
    if (
        evaluation["initialization"] != "positive_weights_zero_readout"
        or evaluation["layout"] != "packed"
    ):
        raise ValueError(
            "Keep the calibrated initialization and packed layout"
        )


def load_decoder(
    run, bank_hash, seed, families, mode="known_mask", manifest=None
):
    """Validate either an archived or current frozen W/G decoder."""
    run = Path(run)
    protocol = json.loads((run / "protocol.json").read_text())
    complete = json.loads((run / "COMPLETE.json").read_text())
    payload = checked_load(run / "decoder.pt")
    if (
        complete["status"] != "complete"
        or protocol["bank_sha256"] != bank_hash
        or protocol["representation"] != "wg"
        or protocol["seed"] != seed
        or payload["epoch"] != complete["selected_epoch"]
        or payload["config"] != protocol["config"]["model"]
    ):
        raise ValueError(f"Incompatible frozen W/G decoder: {run}")
    for name, expected in protocol["implementation_sha256"].items():
        if sha256(run / "source" / name) != expected:
            raise ValueError(
                f"Decoder source snapshot changed: {run / 'source' / name}"
            )
    if mode == "known_mask":
        if any(
            f"family_{family:02d}_function_{function}" not in protocol["tasks"]
            for family in families
            for function in range(4)
        ):
            raise ValueError("Target family was absent from source training")
    elif mode == "composition":
        if (
            manifest is None
            or manifest["config"].get("structure") != "compositional"
        ):
            raise ValueError("Composition requires a compositional bank")
        definitions = manifest["family_definitions"]
        source = [
            family for family in definitions if family["split"] == "train"
        ]
        expected = {
            f"family_{family['id']:02d}_function_{function}"
            for family in source
            for function in range(4)
        }
        if set(protocol["tasks"]) != expected:
            raise ValueError(
                "Decoder must use exactly the training combinations"
            )
        for family_id in families:
            family = definitions[family_id]
            if family["split"] != "validation":
                raise ValueError(
                    "The pilot evaluates validation combinations only"
                )
            if not any(old["a"] == family["a"] for old in source) or not any(
                old["b"] == family["b"] for old in source
            ):
                raise ValueError(
                    "Both target blocks must have occurred in train"
                )
    else:
        raise ValueError(f"Unknown transfer mode: {mode}")
    return protocol, payload
