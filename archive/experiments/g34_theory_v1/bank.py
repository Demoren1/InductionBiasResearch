"""Construct signed and scale-invariant alternatives without teacher access."""

import hashlib
import itertools
import math

import numpy as np

from experiments.sharing_structure_research_v1.blind_contract import (
    canonical_index,
)
from experiments.sharing_structure_research_v1.blind_joint_bank import (
    align_source,
)


def log_observations(normal, uncertainty, mask):
    rows, columns = np.where(mask)
    magnitude = np.abs(normal[:, rows, columns])
    error = np.maximum(uncertainty[:, columns], 0.0003)
    logs = np.log(np.maximum(magnitude, error))
    deviation = np.clip(error / np.maximum(magnitude, error), 0.03, 3.0)
    return rows, columns, logs, deviation


def fit_logs(labels, columns, logs, deviation, codes, hidden):
    incidence = np.eye(codes)[labels - 1]
    matrix = np.c_[incidence, -np.eye(hidden)[columns]]
    predicted, guesses, shifts = [], [], []
    for task in range(len(logs)):
        weight = 1 / deviation[task]
        solution = np.linalg.lstsq(
            matrix * weight[:, None], logs[task] * weight, rcond=1e-10
        )[0]
        for _ in range(2):
            residual = (matrix @ solution - logs[task]) / deviation[task]
            robust = weight / np.sqrt(1 + residual**2 / 4)
            solution = np.linalg.lstsq(
                matrix * robust[:, None], logs[task] * robust, rcond=1e-10
            )[0]
        predicted.append(matrix @ solution)
        guesses.append(solution[:codes])
        shifts.append(solution[codes:])
    predicted, guesses, shifts = map(np.asarray, (predicted, guesses, shifts))
    residual = (predicted - logs) / deviation
    score = float(np.mean(4 * np.log1p(residual**2 / 4)))
    centered = guesses - guesses.mean(1, keepdims=True)
    code_guesses = np.exp(np.clip(centered, -6, 6))
    return score, code_guesses, guesses, shifts


def fill_codes(updated, costs, codes):
    counts = np.bincount(np.abs(updated), minlength=codes + 1)
    for label in range(1, codes + 1):
        if counts[label]:
            continue
        allowed = counts[np.abs(updated)] > 1
        if not allowed.any():
            return None
        local = costs[:, label - 1]
        if costs.shape[1] == 2 * codes:
            local = np.minimum(local, costs[:, label - 1 + codes])
        position = int(np.where(allowed, local, np.inf).argmin())
        counts[abs(updated[position])] -= 1
        sign = 1
        if costs.shape[1] == 2 * codes:
            sign = (
                1
                if costs[position, label - 1]
                <= costs[position, label - 1 + codes]
                else -1
            )
        updated[position] = sign * label
        counts[label] += 1
    return updated


def positive_assign(seed, mask, normal, uncertainty, codes, rng):
    rows, columns, logs, deviation = log_observations(
        normal, uncertainty, mask
    )
    if len(rows) < codes:
        return None
    labels = np.abs(seed[rows, columns]).astype(int)
    absent = labels == 0
    labels[absent] = rng.integers(1, codes + 1, absent.sum())
    for _ in range(20):
        _, _, centers, shifts = fit_logs(
            labels, columns, logs, deviation, codes, normal.shape[2]
        )
        residual = (
            logs[..., None] + shifts[:, columns, None] - centers[:, None]
        ) / deviation[..., None]
        costs = np.mean(4 * np.log1p(residual**2 / 4), axis=0)
        updated = fill_codes(costs.argmin(1) + 1, costs, codes)
        if updated is None:
            return None
        if np.array_equal(updated, labels):
            break
        labels = updated
    result = np.zeros_like(seed)
    result[rows, columns] = labels
    return result


def sign_score(index, normal, uncertainty):
    rows, columns = np.where(index)
    labels = np.abs(index[rows, columns])
    signs = np.sign(index[rows, columns])
    observed = np.where(normal[:, rows, columns] >= 0, 1, -1)
    confidence = np.clip(
        np.abs(normal[:, rows, columns])
        / np.maximum(uncertainty[:, columns], 0.0003),
        0,
        10,
    )
    total, error = float(confidence.sum()), 0.0
    for label in np.unique(labels):
        active = labels == label
        weight = confidence[:, active]
        vote = (weight * observed[:, active] * signs[active]).sum(1)
        error += float((weight.sum(1) - np.abs(vote)).sum() / 2)
    return error / max(total, 1e-12)


def sign_assign(seed, mask, normal, uncertainty, codes, rng):
    rows, columns = np.where(mask)
    if len(rows) < codes:
        return None
    observed = np.where(normal[:, rows, columns] >= 0, 1, -1)
    weight = np.clip(
        np.abs(normal[:, rows, columns])
        / np.maximum(uncertainty[:, columns], 0.0003),
        0.01,
        10,
    )
    labels = seed[rows, columns].astype(int)
    missing = labels == 0
    labels[missing] = rng.integers(1, codes + 1, missing.sum())
    symbols = np.r_[np.arange(1, codes + 1), -np.arange(1, codes + 1)]
    for _ in range(30):
        centers = np.empty((len(normal), codes))
        for label in range(1, codes + 1):
            active = np.abs(labels) == label
            vote = (
                weight[:, active]
                * observed[:, active]
                * np.sign(labels[active])
            ).sum(1)
            centers[:, label - 1] = np.where(vote >= 0, 1, -1)
        extended = np.c_[centers, -centers]
        costs = (
            weight[..., None] * (observed[..., None] != extended[:, None])
        ).sum(0) / weight.sum(0)[:, None]
        updated = fill_codes(symbols[costs.argmin(1)], costs, codes)
        if updated is None:
            return None
        if np.array_equal(updated, labels):
            break
        labels = updated
    result = np.zeros_like(seed)
    result[rows, columns] = labels
    return result


def local_groups(seed, rows, columns, logs, deviation, mode, codes):
    groups = []
    for neuron in np.unique(columns):
        positions = np.flatnonzero(columns == neuron)
        if mode == "anchor":
            labels = np.abs(seed[rows[positions], columns[positions]])
            local = [
                positions[labels == label].tolist()
                for label in np.unique(labels)
                if label
            ]
            for position in positions[labels == 0]:
                if not local:
                    local.append([int(position)])
                    continue
                costs = [
                    np.mean((logs[:, position] - logs[:, group].mean(1)) ** 2)
                    for group in local
                ]
                local[int(np.argmin(costs))].append(int(position))
        else:
            local = [[int(position)] for position in positions]
        while len(local) > 1:
            choices = []
            for a, b in itertools.combinations(range(len(local)), 2):
                delta = logs[:, local[a]].mean(1) - logs[:, local[b]].mean(1)
                scale = deviation[:, local[a]].mean(1) + deviation[
                    :, local[b]
                ].mean(1)
                cost = float(np.mean((delta / scale) ** 2))
                choices.append((cost, a, b))
            cost, a, b = min(choices)
            if len(local) <= codes and (mode == "anchor" or cost > 1):
                break
            local[a].extend(local[b])
            del local[b]
        groups.extend((int(neuron), group) for group in local)
    return groups


def relation_assign(seed, mask, normal, uncertainty, codes, mode, subset, rng):
    rows, columns, logs, deviation = log_observations(
        normal, uncertainty, mask
    )
    groups = local_groups(seed, rows, columns, logs, deviation, mode, codes)
    if len(groups) < codes:
        return None
    profiles = np.stack([logs[:, p].mean(1) for _, p in groups], axis=1)
    errors = np.stack([deviation[:, p].mean(1) for _, p in groups], axis=1)
    pairs = []
    for a, b in itertools.combinations(range(len(groups)), 2):
        if groups[a][0] != groups[b][0]:
            continue
        pairs.append(
            (
                a,
                b,
                profiles[:, a] - profiles[:, b],
                errors[:, a] + errors[:, b],
            )
        )
    edges = []
    for left, right in itertools.combinations(pairs, 2):
        a, b, delta, error = left
        c, d, other, noise = right
        if groups[a][0] == groups[c][0]:
            continue
        scale = np.maximum(error + noise, 0.06)[subset]
        direct = float(np.mean(((delta - other)[subset] / scale) ** 2))
        reverse = float(np.mean(((delta + other)[subset] / scale) ** 2))
        if reverse < direct:
            c, d = d, c
        jitter = 1 + rng.uniform(0, 0.02)
        edges.append((min(direct, reverse) * jitter, a, b, c, d))
    edges.sort()
    parent = np.arange(len(groups))

    def root(node, parents):
        while parents[node] != node:
            node = parents[node]
        return node

    def merge(a, b, parents):
        a, b = root(a, parents), root(b, parents)
        if a == b:
            return True
        first = {
            groups[i][0] for i in range(len(groups)) if root(i, parents) == a
        }
        second = {
            groups[i][0] for i in range(len(groups)) if root(i, parents) == b
        }
        if first & second:
            return False
        parents[b] = a
        return True

    for _, a, b, c, d in edges:
        trial = parent.copy()
        if not merge(a, c, trial) or not merge(b, d, trial):
            continue
        count = len({root(i, trial) for i in range(len(groups))})
        if count < codes:
            continue
        parent = trial
        if count == codes:
            break
    roots = [root(i, parent) for i in range(len(groups))]
    unique = sorted(set(roots))
    if len(unique) != codes:
        return None
    result = np.zeros_like(seed)
    for number, (_, positions) in enumerate(groups):
        result[rows[positions], columns[positions]] = (
            unique.index(roots[number]) + 1
        )
    return result


def canonical_unsigned(index, codes):
    best, selected = None, None
    for permutation in itertools.permutations(range(1, codes + 1)):
        labels = np.r_[0, permutation]
        renamed = labels[np.abs(index)]
        renamed = renamed[:, np.lexsort(renamed[::-1])]
        key = renamed.tobytes()
        if best is None or key < best:
            best, selected = key, renamed
    return selected


def space_key(unsigned, codes):
    rows, columns = np.where(unsigned)
    lookup = np.eye(codes)[unsigned[rows, columns] - 1]
    for neuron in np.unique(columns):
        active = columns == neuron
        lookup[active] -= lookup[active].mean(0)
    basis, values, _ = np.linalg.svd(lookup, full_matrices=False)
    rank = int((values > 1e-9).sum())
    basis = basis[:, :rank]
    projector = np.round(basis @ basis.T, 8)
    projector[np.abs(projector) < 5e-9] = 0
    value = hashlib.sha256((unsigned != 0).tobytes())
    value.update(str(rank).encode())
    value.update(projector.tobytes())
    return value.hexdigest()


def diverse_ids(records, candidates, limit, seen):
    bins = {}
    for number in sorted(candidates, key=lambda i: records[i]["score"]):
        if records[number]["key"] in seen:
            continue
        bins.setdefault(records[number]["space"], []).append(number)
    ordered = sorted(bins.values(), key=lambda ids: records[ids[0]]["score"])
    chosen = []
    for ids in ordered:
        for number in ids[:2]:
            if records[number]["key"] in seen:
                continue
            chosen.append(number)
            seen.add(records[number]["key"])
            if len(chosen) == limit:
                return chosen
    for number in sorted(candidates, key=lambda i: records[i]["score"]):
        if records[number]["key"] in seen:
            continue
        chosen.append(number)
        seen.add(records[number]["key"])
        if len(chosen) == limit:
            break
    return chosen


def build_pool(base, joint, source):
    codes = joint["code_guesses"].shape[-1]
    indices, guesses, records = [], [], []
    known = set()
    space_cache = {}
    canonical_cache = {}
    unsigned_cache = {}
    rng = np.random.default_rng(27110817)

    def add(index, code, family, score, origin, *, already_canonical=False):
        if len(np.unique(np.abs(index[index != 0]))) != codes:
            return
        raw_key = index.astype(np.int16).tobytes()
        if already_canonical:
            canonical = index.astype(np.int16)
        else:
            if raw_key not in canonical_cache:
                canonical_cache[raw_key] = canonical_index(
                    index.astype(np.int16), codes
                )
            canonical = canonical_cache[raw_key]
        key = canonical.tobytes().hex()
        if (family, key) in known:
            return
        known.add((family, key))
        local_key = np.abs(canonical).tobytes()
        if local_key not in unsigned_cache:
            unsigned_cache[local_key] = canonical_unsigned(canonical, codes)
        unsigned = unsigned_cache[local_key]
        unsigned_key = unsigned.tobytes()
        if unsigned_key not in space_cache:
            space_cache[unsigned_key] = space_key(unsigned, codes)
        indices.append(canonical)
        guesses.append(code)
        records.append(
            {
                "key": key,
                "space": space_cache[unsigned_key],
                "family": family,
                "score": float(score),
                **origin,
            }
        )

    for number, index in enumerate(joint["indices"]):
        add(
            index,
            joint["code_guesses"][number],
            "original",
            joint["score"][number],
            {"original_id": number},
            already_canonical=True,
        )
    print(f"Original pool descriptors: {len(records)}", flush=True)
    ranked = base["validation_mse"].mean(1).argsort()[:4]
    for anchor_number in ranked:
        anchor = base["indices"][anchor_number]
        mappings = [
            align_source(
                anchor,
                base["parameters"][anchor_number, t, :codes],
                item["normal"],
                item["uncertainty"],
            )
            for t, item in enumerate(source)
        ]
        normal = np.stack(
            [s["normal"][:, m] for s, m in zip(source, mappings)]
        )
        uncertainty = np.stack(
            [s["uncertainty"][m] for s, m in zip(source, mappings)]
        )
        tasks = len(source)
        evidence = np.abs(normal) > np.maximum(3 * uncertainty[:, None], 0.002)
        levels = sorted(
            {
                1,
                3,
                math.ceil(tasks / 4),
                math.ceil(tasks / 2),
                math.ceil(3 * tasks / 4),
            }
        )
        masks = [("anchor", anchor != 0)]
        masks += [(str(level), evidence.sum(0) >= level) for level in levels]
        observed_signs = np.sign(base["parameters"][anchor_number, :, :codes])
        distances = [
            min(
                np.count_nonzero(observed_signs[:, a] != observed_signs[:, b]),
                np.count_nonzero(observed_signs[:, a] == observed_signs[:, b]),
            )
            for a, b in itertools.combinations(range(codes), 2)
        ]
        sign_diverse = min(distances, default=tasks) >= 2
        for mask_name, mask in masks:
            if mask.sum() < codes or np.any(mask.sum(0) == 0):
                continue
            rows, columns, logs, deviation = log_observations(
                normal, uncertainty, mask
            )
            proposals = []
            for mode in ("anchor", "observed"):
                for subset in (np.arange(tasks), np.arange(0, tasks, 2)):
                    relation = relation_assign(
                        anchor,
                        mask,
                        normal,
                        uncertainty,
                        codes,
                        mode,
                        subset,
                        rng,
                    )
                    if relation is not None:
                        proposals.append(("relations", relation))
            seeds = [anchor]
            for _ in range(2):
                seeds.append(rng.integers(1, codes + 1, size=anchor.shape))
            for seed in seeds:
                result = positive_assign(
                    seed, mask, normal, uncertainty, codes, rng
                )
                if result is not None:
                    proposals.append(("log_positive", result))
                if sign_diverse:
                    result = sign_assign(
                        seed, mask, normal, uncertainty, codes, rng
                    )
                    if result is not None:
                        proposals.append(("signs", result))
            cache = {}
            for family, index in proposals:
                labels = np.abs(index[rows, columns])
                cache_key = labels.tobytes()
                if cache_key not in cache:
                    cache[cache_key] = fit_logs(
                        labels,
                        columns,
                        logs,
                        deviation,
                        codes,
                        normal.shape[2],
                    )[:2]
                geometry, code = cache[cache_key]
                oriented = [index]
                oriented += [
                    np.abs(index) * np.where(normal[t] >= 0, 1, -1)
                    for t in range(tasks)
                ]
                for candidate in oriented:
                    compatibility = sign_score(candidate, normal, uncertainty)
                    score = (
                        compatibility
                        if family == "signs"
                        else (geometry + 2 * compatibility)
                    )
                    add(
                        candidate,
                        code,
                        family,
                        score,
                        {
                            "anchor": int(anchor_number),
                            "mask_votes": mask_name,
                            "log_score": geometry,
                            "sign_score": compatibility,
                        },
                    )
        print(
            f"Anchor {int(anchor_number)}: {len(records)} pool records",
            flush=True,
        )
    return np.stack(indices), np.stack(guesses), records


def choose_bank(records, method, limit=32):
    seen, chosen = set(), []
    if method == "diversity":
        original = [
            i for i, r in enumerate(records) if r["family"] == "original"
        ]
        return diverse_ids(records, original, limit, seen)
    quotas = {"original": 8, "relations": 12, "log_positive": 8, "signs": 4}
    for family, quota in quotas.items():
        available = [
            i for i, row in enumerate(records) if row["family"] == family
        ]
        chosen.extend(diverse_ids(records, available, quota, seen))
    for family in ("relations", "log_positive", "signs", "original"):
        remaining = limit - len(chosen)
        if not remaining:
            break
        available = [
            i for i, row in enumerate(records) if row["family"] == family
        ]
        chosen.extend(diverse_ids(records, available, remaining, seen))
    return chosen
