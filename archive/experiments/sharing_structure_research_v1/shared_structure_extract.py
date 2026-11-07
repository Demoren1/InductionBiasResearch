"""Source-only candidates for G23; never opens meta labels or teacher I/C."""

import json
import math
import time
from functools import partial

import numpy as np
import torch

from experiments.darts_structured_tasks_v1.data import digest, save_json

from .codebook_counts import curve_fit, integer_segments
from .count_exchanges import search_counts
from .count_projection import census
from .count_qp_active import function_qp
from .shared_structure_data import load_source
from .shared_structure_native import train
from .shared_structure_numeric import canonical_index, fit_codes


@torch.no_grad()
def prefix_candidates(data, plan, seed):
    t, vt = data["axis_t"], data["axis_vt"]
    axis_count = 2 * plan["inputs"] * plan["axis_points"]
    y = (
        data["train_y"][:axis_count]
        .reshape(2, plan["inputs"], len(t))
        .sum((0, 1))
        .cpu()
        .numpy()
    )
    vy = (
        data["axis_validation_y"]
        .reshape(2, plan["inputs"], len(vt))
        .sum((0, 1))
        .cpu()
        .numpy()
    )
    offset = 2 * plan["inputs"] * math.sqrt(plan["hidden"]) * plan["bias"]
    initial, masses, dp = integer_segments(t, y - offset, plan, data["train_x"].device)
    centers, axis = curve_fit(
        t,
        y - offset,
        vt,
        vy - offset,
        plan,
        seed,
        "continuous_dp_start",
        initial,
        masses,
    )
    slices = list(
        data["train_x"][axis_count:].reshape(
            plan["inputs"], plan["slice_points"], plan["inputs"]
        )
    )
    targets = data["train_y"][axis_count:].reshape(plan["inputs"], plan["slice_points"])
    validation_x = list(
        data["prefix_validation_x"].reshape(
            plan["inputs"], plan["prefix_validation"], plan["inputs"]
        )
    )
    validation_y = data["prefix_validation_y"].reshape(
        plan["inputs"], plan["prefix_validation"]
    )
    for method in plan["prefix_settings"]:
        started = time.monotonic()
        base = "qp_integer_refit" if method.endswith("refit") else "qp_integer"
        integer_solver = (
            partial(search_counts, kind="parent_pair", device=data["train_x"].device)
            if method.startswith("parent")
            else None
        )
        output = census(
            slices,
            targets,
            validation_x,
            validation_y,
            data["order"],
            centers,
            base,
            plan,
            qp_solver=function_qp,
            integer_solver=integer_solver,
        )
        if output["complete"]:
            code, polishing = fit_codes(
                output["indices"][-1],
                data["train_x"],
                data["train_y"],
                data["fit_validation_x"],
                data["fit_validation_y"],
                torch.tensor(output["centers"], device=data["train_x"].device),
                plan,
                max_steps=plan["source_polish_steps"],
            )
        else:
            code, polishing = None, None
        yield {
            "setting": method,
            "restart": 0,
            "complete": output["complete"],
            "validation_mse": None
            if polishing is None
            else polishing["validation_mse"],
            "index": output["indices"][-1] if output["complete"] else None,
            "centers": None if code is None else code.cpu().numpy(),
            "trace_indices": output["indices"],
            "trace_fitted_C": output["codebooks"],
            "history": output["trace"],
            "polish": polishing,
            "axis_fit": axis,
            "axis_dp": dp,
            "seconds": time.monotonic() - started,
        }


def run(root, plan, job_id, device):
    path = root / "extract" / f"job_{job_id:04d}"
    if path.with_suffix(".json").exists():
        raise RuntimeError("Refusing to repeat completed extraction")
    started = time.monotonic()
    job = plan["extract_jobs"][job_id]
    family = plan["families"][job["family_id"]]["family"]
    data = load_source(root, plan, job["family_id"], job["task"], job["noise"], device)
    for name in ("train", "axis_validation", "prefix_validation", "fit_validation"):
        data[name + "_y"] = data.pop(f"{name}_y_{job['noise']}")
    records, arrays = [], {}
    if job["method"] == "prefix":
        results = prefix_candidates(
            data, plan, plan["seed"] + family * 23 + job["task"] * 71
        )
    else:

        def native_results():
            for setting_id, setting in enumerate(
                plan["native_settings"][job["method"]]
            ):
                for restart in range(plan["native_restarts"]):
                    result = train(
                        job["method"], setting, restart, family, job["task"], data, plan
                    )
                    yield {
                        **result,
                        "setting": setting,
                        "setting_id": setting_id,
                        "restart": restart,
                        "complete": True,
                    }

        results = native_results()
    for number, result in enumerate(results):
        key = f"candidate_{number}"
        record = {
            key: value
            for key, value in result.items()
            if key
            not in (
                "index",
                "centers",
                "trace_indices",
                "trace_latent_weights",
                "trace_fitted_C",
            )
        }
        record["key"] = key
        for field in (
            "index",
            "centers",
            "trace_indices",
            "trace_latent_weights",
            "trace_fitted_C",
        ):
            if field in result and result[field] is not None:
                arrays[key + "__" + field] = result[field]
        if result["complete"]:
            arrays[key + "__canonical_I"] = canonical_index(
                result["index"], plan["codes"]
            )
        records.append(record)
        print(
            json.dumps(
                {
                    "job_id": job_id,
                    "candidate": number,
                    "method": job["method"],
                    "complete": result["complete"],
                    "validation_mse": result["validation_mse"],
                    "seconds": time.monotonic() - started,
                }
            ),
            flush=True,
        )
    eligible = sorted(
        (n for n, record in enumerate(records) if record["complete"]),
        key=lambda n: records[n]["validation_mse"],
    )
    selected, seen = [], set()
    for number in eligible:
        key = records[number]["key"]
        signature = arrays[key + "__canonical_I"].tobytes()
        if signature not in seen:
            selected.append(number)
            seen.add(signature)
        if len(selected) == plan["bank_per_source"]:
            break
    np.savez_compressed(path.with_suffix(".npz"), **arrays)
    save_json(
        path.with_suffix(".json"),
        {
            "job": job,
            "records": records,
            "selected_records": selected,
            "seconds": time.monotonic() - started,
            "protocol_sha256": digest(root / "protocol.json"),
            "source_observation_sha256": digest(
                root / "data" / f"family_{job['family_id']:03d}_source.npz"
            ),
            "meta_labels_opened": False,
        },
    )
