"""SRR development: local ratio matching with incompatible-merge rejection.

Uncertainty is a local standard-error estimate, not a certified error bound.
Every returned candidate is checked against magnitudes as well as signs.
"""

import itertools

import numpy as np
from scipy.optimize import linear_sum_assignment


def local_groups(normal, uncertainty, multiplier, codes):
    """Form compact magnitude intervals, without single-linkage chaining."""
    hidden = normal.shape[1]
    memberships = np.full(normal.shape, -1, dtype=int)
    centers, errors, neurons, counts = [], [], [], []
    for neuron in range(hidden):
        delta = max(multiplier * uncertainty[neuron], 1e-10)
        values = np.abs(normal[:, neuron])
        active = np.flatnonzero(values > delta)
        order = active[np.argsort(values[active])]
        groups = []
        for position in order:
            if not groups or values[position] - values[groups[-1][0]] > 2 * delta:
                groups.append([position])
            else:
                groups[-1].append(position)
        if len(groups) > codes:
            return None
        for group in groups:
            number = len(centers)
            memberships[group, neuron] = number
            center = float(np.mean(values[group]))
            centers.append(np.log(center))
            errors.append(
                max(uncertainty[neuron], 1e-10) / center / np.sqrt(len(group))
            )
            neurons.append(neuron)
            counts.append(len(group))
    return {
        "memberships": memberships,
        "logs": np.asarray(centers),
        "errors": np.asarray(errors),
        "neurons": np.asarray(neurons),
        "counts": np.asarray(counts),
    }


def pair_matches(local, multiplier):
    """Compare whole local configurations after cancelling their log scale."""
    logs, errors, neurons = (local[key] for key in ("logs", "errors", "neurons"))
    proposals = []
    for first, second in itertools.combinations(np.unique(neurons), 2):
        left, right = (
            np.flatnonzero(neurons == first),
            np.flatnonzero(neurons == second),
        )
        found = []
        for count in range(min(len(left), len(right)), 1, -1):
            for a in itertools.combinations(left, count):
                for b in itertools.combinations(right, count):
                    a, b = np.array(a), np.array(b)
                    variance = errors[a] ** 2 + errors[b] ** 2
                    weight = 1 / np.maximum(variance, 1e-20)
                    difference = logs[a] - logs[b]
                    shift = np.average(difference, weights=weight)
                    residual = difference - shift
                    score = float(np.sum(weight * residual**2) / max(count - 1, 1))
                    if score <= multiplier**2:
                        found.append((score, a, b))
            if found:
                break
        if found:
            found.sort(key=lambda row: row[0])
            score, a, b = found[0]
            # A complete local configuration carries more evidence than one ratio.
            ambiguity = found[1][0] - score if len(found) > 1 else np.inf
            proposals.append((-len(a), score, -ambiguity, a, b))
    proposals.sort(key=lambda row: row[:3])
    return proposals


def synchronize(local, proposals, codes):
    """Merge matched endpoints, forbidding two local classes in one group."""
    neurons = local["neurons"]
    parent = np.arange(len(neurons))
    members = [{int(neuron)} for neuron in neurons]

    def root(node):
        while parent[node] != node:
            node = parent[node]
        return int(node)

    accepted, rejected = 0, 0
    for _, _, _, left, right in proposals:
        changes, valid = [], True
        for a, b in zip(left, right):
            a, b = root(a), root(b)
            if a == b:
                continue
            if members[a] & members[b]:
                valid = False
                break
            changes.append((a, b, members[a].copy()))
            parent[b] = a
            members[a] |= members[b]
        if not valid:
            # All matches in one local configuration are accepted together.
            for a, b, previous in reversed(changes):
                parent[b] = b
                members[a] = previous
            rejected += 1
        else:
            accepted += 1
    roots = np.array([root(node) for node in range(len(neurons))])
    _, labels = np.unique(roots, return_inverse=True)
    count = len(np.unique(labels))
    diagnostics = {
        "groups": count,
        "accepted_matches": accepted,
        "rejected_matches": rejected,
    }
    if count != codes:
        return None, diagnostics
    index = np.zeros(local["memberships"].shape, dtype=np.int16)
    active = local["memberships"] >= 0
    index[active] = labels[local["memberships"][active]] + 1
    return index, diagnostics


def fit_magnitudes(index, normal, uncertainty, codes):
    """Fit signed codes and positive neuron scales to a fixed indexator."""
    rows, columns = np.where(index)
    if len(rows) < codes:
        return None
    labels = np.abs(index[rows, columns]) - 1
    if len(np.unique(labels)) != codes:
        return None
    structural_sign = np.sign(index[rows, columns])
    values = normal[rows, columns] * structural_sign
    deviation = np.maximum(uncertainty, 1e-10)
    precision = 1 / deviation[columns] ** 2
    positive = np.bincount(labels, precision * np.maximum(values, 0), minlength=codes)
    negative = np.bincount(labels, precision * np.maximum(-values, 0), minlength=codes)
    signs = np.where(positive >= negative, 1, -1)
    design = np.c_[np.eye(codes)[labels], np.eye(normal.shape[1])[columns]]
    weight = np.minimum(np.abs(values) / deviation[columns], 1e8)
    logs = np.log(np.maximum(np.abs(values), 1e-12))
    solution = np.linalg.lstsq(design * weight[:, None], logs * weight, rcond=1e-12)[0]
    code = signs * np.exp(np.clip(solution[:codes] - solution[:codes].mean(), -12, 12))
    scales = np.ones(normal.shape[1])
    for _ in range(24):
        local = structural_sign * code[labels]
        numerator = np.bincount(
            columns, local * normal[rows, columns], minlength=normal.shape[1]
        )
        denominator = np.bincount(columns, local**2, minlength=normal.shape[1])
        scales = np.maximum(numerator / np.maximum(denominator, 1e-30), 1e-12)
        local = structural_sign * scales[columns]
        numerator = np.bincount(
            labels, precision * local * normal[rows, columns], minlength=codes
        )
        denominator = np.bincount(labels, precision * local**2, minlength=codes)
        updated = numerator / np.maximum(denominator, 1e-30)
        norm = max(float(np.linalg.norm(updated)), 1e-12)
        updated /= norm
        scales *= norm
        if np.max(np.abs(updated - code)) < 1e-12:
            code = updated
            break
        code = updated
    prediction = np.zeros_like(normal)
    prediction[rows, columns] = structural_sign * code[labels] * scales[columns]
    residual = (prediction - normal) / deviation[None]
    return {
        "code": code,
        "scales": scales,
        "prediction": prediction,
        "score": float(np.mean(residual**2)),
        "max_standardized_residual": float(np.max(np.abs(residual))),
    }


def recover_local(normal, uncertainty, codes, multipliers=(1.5, 2.5, 4.0, 6.0)):
    """Construct local-ratio candidates; keep magnitude verification explicit."""
    candidates, trace = {}, []
    for multiplier in multipliers:
        local = local_groups(normal, uncertainty, multiplier, codes)
        if local is None:
            trace.append({"multiplier": multiplier, "status": "too_many_local_groups"})
            continue
        proposals = pair_matches(local, multiplier)
        index, record = synchronize(local, proposals, codes)
        record["multiplier"] = multiplier
        if index is not None:
            index *= np.sign(normal).astype(np.int16)
            fit = fit_magnitudes(index, normal, uncertainty, codes)
            record.update(
                {key: fit[key] for key in ("score", "max_standardized_residual")}
            )
            record["status"] = "candidate"
            for proposed, fitted, refinement in (
                (index, fit, False),
                (*complete_by_weights(index, normal, uncertainty, codes), True),
            ):
                if proposed is None:
                    continue
                key = proposed.tobytes()
                if key not in candidates or fitted["score"] < candidates[key]["score"]:
                    candidates[key] = {
                        "index": proposed,
                        "multiplier": multiplier,
                        "refined": refinement,
                        **fitted,
                    }
        else:
            record["status"] = "incomplete"
        trace.append(record)
    return sorted(candidates.values(), key=lambda row: row["score"]), trace


def complete_by_weights(initial, normal, uncertainty, codes):
    """Complete ambiguous support using code ratios inferred by SRR.

    This alternating raw-weight projection is a practical refinement, not a
    consequence of the exact pair-union theorem. Zero is an explicit option.
    """
    index = initial.copy()
    for _ in range(24):
        fit = fit_magnitudes(index, normal, uncertainty, codes)
        if fit is None:
            return None, None
        centers = np.r_[0.0, np.abs(fit["code"])]
        centers = centers[:, None] * fit["scales"][None]
        distance = np.abs(np.abs(normal)[..., None] - centers.T[None])
        labels = distance.argmin(-1).astype(np.int16)
        updated = labels * np.sign(normal).astype(np.int16)
        if len(np.unique(labels[labels != 0])) != codes:
            return None, None
        if np.array_equal(updated, index):
            break
        index = updated
    return index, fit_magnitudes(index, normal, uncertainty, codes)


def score_on_tasks(index, weights, uncertainty, codes):
    """Check a fixed signed partition on independently estimated tasks."""
    scores, traces = [], []
    for normal, deviation in zip(weights, uncertainty):
        observed = np.abs(normal) > np.maximum(3 * deviation[None], 1e-10)
        support_cost = ((index != 0)[:, :, None] != observed[:, None]).sum(0)
        direction = np.abs(normal)
        direction /= np.maximum(np.linalg.norm(direction, axis=0), 1e-12)
        template = (index != 0).astype(float)
        template /= np.maximum(np.linalg.norm(template, axis=0), 1e-12)
        distance = ((template[:, :, None] - direction[:, None]) ** 2).sum(0)
        left, right = linear_sum_assignment(
            support_cost + distance / (2 * index.shape[1] + 1)
        )
        mapping = right[np.argsort(left)]
        fit = fit_magnitudes(index, normal[:, mapping], deviation[mapping], codes)
        scores.append(min(fit["score"], 1e12))
        traces.append(
            {
                "score": fit["score"],
                "max_standardized_residual": fit["max_standardized_residual"],
                "mapping": mapping.tolist(),
            }
        )
    # Local covariance need not cover optimization/representation failures.
    # Require agreement of at least half the independently fitted tasks.
    keep = max(1, (len(scores) + 1) // 2)
    return float(np.mean(np.sort(scores)[:keep])), traces
