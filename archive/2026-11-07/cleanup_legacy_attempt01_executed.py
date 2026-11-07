"""Recorded one-time cleanup of superseded research; never called by training."""

import argparse
import gzip
import hashlib
import json
import math
import shutil
import tarfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from zipfile import ZipFile

import numpy as np


def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def compact_arrays(path):
    """Retain bounded structural fields and metrics, never raw observations."""
    candidates = []
    selected = any(
        word in path.stem.lower()
        for word in ("selected", "best", "learned", "final", "frozen")
    )
    with ZipFile(path) as archive:
        for entry in archive.infolist():
            if not entry.filename.endswith(".npy"):
                continue
            with archive.open(entry) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                else:
                    shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
            if dtype.hasobject:
                continue
            key = entry.filename[:-4]
            name = key.lower()
            size = math.prod(shape) * dtype.itemsize
            structural = name in ("i", "i1", "i2", "j", "j1", "j2") or any(
                word in name
                for word in ("index", "indices", "mask", "structure", "basis")
            )
            codebook = name in (
                "c",
                "c1",
                "c2",
                "codes",
                "codebook",
                "codebooks",
                "centers",
            )
            metric = any(
                word in name
                for word in (
                    "loss",
                    "mse",
                    "accuracy",
                    "score",
                    "risk",
                    "history",
                    "steps",
                    "recall",
                    "precision",
                    "success",
                )
            )
            numerical = selected and name in (
                "parameters",
                "weights",
                "readout",
                "bias",
                "offset",
            )
            limit = (
                2 * 1024**2 if selected and (structural or numerical) else 256 * 1024
            )
            if (structural or codebook or metric or numerical) and size <= limit:
                priority = 0 if structural else 1 if codebook else 2 if metric else 3
                candidates.append((priority, key, size))
    retained, total = {}, 0
    with np.load(path, allow_pickle=False) as archive:
        for _, key, size in sorted(candidates):
            if total + size <= 2 * 1024**2:
                retained[key] = archive[key]
                total += size
    return retained


def run(root):
    recorded = root / "archive/2026-11-07"
    source_manifest = recorded / "source_before_cleanup.manifest.json"
    if not source_manifest.is_file():
        raise RuntimeError("Verify the source backup before destructive cleanup")
    output = recorded / "artifact_cleanup.json.gz"
    if output.exists():
        raise RuntimeError("This cleanup was already recorded")
    core = set(json.loads(Path("/tmp/ibr-cleanup-core.json").read_text()))
    core.update(
        {
            "experiments/sharing_structure_research_v1/__init__.py",
            "experiments/sharing_structure_research_v1/shared_structure_bilevel_optimization.py",
        }
    )
    folders = [
        path
        for path in (root / "experiments").iterdir()
        if path.is_dir()
        and path.name not in ("paper_structure_benchmark_v1", "__pycache__")
    ] + [root / "pattern", root / "overlapping_groups"]
    sources = sorted(
        path for folder in folders for path in folder.rglob("*") if path.is_file()
    )
    local = root / "artifacts/local"
    local.mkdir(parents=True, exist_ok=True)
    local_archive = local / "legacy_records_2026-11-07.tar.gz"
    if local_archive.exists():
        raise FileExistsError(local_archive)
    records, pending = [], []
    counters, before = Counter(), 0
    with tarfile.open(local_archive, "w:gz", compresslevel=1) as bundle:
        for number, source in enumerate(sources, 1):
            relative = source.relative_to(root)
            name = str(relative)
            if name in core:
                continue
            size = source.stat().st_size
            before += size
            record = {"source": name, "bytes": size}
            if "__pycache__" in relative.parts or source.suffix in (".pyc", ".log"):
                source.unlink()
                record["action"] = "removed_regenerable_cache_or_log"
            else:
                record["sha256"] = sha256(source)
                target = (
                    root / "archive/experiments" / Path(*relative.parts[1:])
                    if relative.parts[0] == "experiments"
                    else root / "archive/experiments" / relative
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                if source.suffix == ".npz":
                    arrays = compact_arrays(source)
                    if arrays:
                        target = target.with_name(target.stem + ".summary.npz")
                        np.savez_compressed(target, **arrays)
                        record.update(
                            action="retained_compact_array_summary",
                            retained=str(target.relative_to(root)),
                            retained_bytes=target.stat().st_size,
                            retained_sha256=sha256(target),
                            retained_arrays=list(arrays),
                        )
                    else:
                        record["action"] = "removed_bulk_numerical_array"
                    source.unlink()
                elif (
                    source.name.endswith(".json.gz")
                    or source.suffix == ".json"
                    and size > 64 * 1024
                ):
                    bundle.add(source, arcname=name, recursive=False)
                    pending.append(source)
                    record["action"] = "retained_in_local_records_archive"
                    record["archive_member"] = name
                elif source.suffix == ".py" and any(
                    part in relative.parts for part in ("results", "runs")
                ):
                    source.unlink()
                    record["action"] = "source_snapshot_in_verified_backup"
                elif (
                    source.suffix
                    in (".pt", ".pth", ".npy", ".pkl", ".pickle", ".gz", ".zip")
                    or "datasets" in relative.parts
                    or "raw" in relative.parts
                ):
                    source.unlink()
                    record["action"] = "removed_legacy_checkpoint_or_download"
                elif (
                    source.suffix
                    in (
                        ".py",
                        ".md",
                        ".json",
                        ".csv",
                        ".png",
                        ".pdf",
                        ".svg",
                        ".tex",
                        ".bib",
                        ".txt",
                        ".toml",
                        ".yml",
                        ".yaml",
                        ".sh",
                        ".sha256",
                        ".ipynb",
                    )
                    or size <= 64 * 1024
                ):
                    if target.exists():
                        raise FileExistsError(target)
                    source.rename(target)
                    record.update(
                        action="archived",
                        retained=str(target.relative_to(root)),
                        retained_bytes=size,
                    )
                else:
                    raise RuntimeError(f"Unclassified legacy artifact: {source}")
            counters[record["action"]] += 1
            records.append(record)
            if number % 2000 == 0:
                print(
                    json.dumps(
                        {
                            "processed": number,
                            "total": len(sources),
                            "actions": dict(counters),
                        }
                    ),
                    flush=True,
                )

    expected = {
        row["source"]: row["sha256"]
        for row in records
        if row["action"] == "retained_in_local_records_archive"
    }
    with tarfile.open(local_archive, "r:gz") as bundle:
        for member in bundle:
            value = hashlib.sha256()
            stream = bundle.extractfile(member)
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                value.update(chunk)
            if value.hexdigest() != expected.pop(member.name):
                raise RuntimeError(f"Local archive mismatch: {member.name}")
    if expected:
        raise RuntimeError("Incomplete local records archive")
    for path in pending:
        path.unlink()
    for folder in folders:
        for child in sorted(
            (p for p in folder.rglob("*") if p.is_dir()),
            key=lambda p: len(p.parts),
            reverse=True,
        ):
            if not any(child.iterdir()):
                child.rmdir()
        if not any(folder.iterdir()):
            folder.rmdir()
    kept = (
        sum(row.get("retained_bytes", 0) for row in records)
        + local_archive.stat().st_size
    )
    report = {
        "completed_utc": datetime.now(UTC).isoformat(),
        "protected": "Entire final benchmark results/20261107 and all registered G34 core source files; current predictions and raw data are untouched",
        "source_backup": str(source_manifest.relative_to(root)),
        "local_records_archive": str(local_archive.relative_to(root)),
        "local_records_sha256": sha256(local_archive),
        "bytes_before": before,
        "bytes_retained": kept,
        "net_bytes_freed": before - kept,
        "counts": dict(counters),
        "records": records,
    }
    with gzip.open(output, "wt", encoding="utf-8", compresslevel=6) as stream:
        json.dump(report, stream, ensure_ascii=False)
        stream.write("\n")
    summary = {key: value for key, value in report.items() if key != "records"}
    (recorded / "artifact_cleanup_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "completed": True,
                "freed_GiB": (before - kept) / 2**30,
                "retained_MiB": kept / 2**20,
                "counts": dict(counters),
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", required=True)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    if root != Path("/home/udeneev-av/InductionBiasResearch"):
        raise RuntimeError("This recorded cleanup is specific to its original checkout")
    run(root)


if __name__ == "__main__":
    main()
