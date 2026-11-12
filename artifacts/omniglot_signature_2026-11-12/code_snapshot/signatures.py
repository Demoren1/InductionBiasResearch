"""Stream the same two signature merge rules without a dense pair graph."""

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from experiments.theorem_pair_recovery_v1.recover import THRESHOLDS, contrast_rank


def merge(labels, left, right):
    count = int(labels.max()) + 1
    graph = csr_matrix(
        (np.ones(len(left), dtype=bool), (labels[left], labels[right])),
        shape=(count, count),
    )
    count, components = connected_components(graph, directed=False)
    return count, components[labels]


def zero_forest(basis, columns):
    """Single linkage preserves every zero-rule threshold component exactly."""
    left, right, height = [], [], []
    for neuron in np.unique(columns):
        positions = np.flatnonzero(columns == neuron)
        if len(positions) < 2:
            continue
        hierarchy = linkage(basis[positions], method="single")
        representatives = list(positions)
        for first, second, distance, _ in hierarchy:
            a, b = representatives[int(first)], representatives[int(second)]
            left.append(a)
            right.append(b)
            height.append(distance)
            representatives.append(a)
    return np.asarray(left, int), np.asarray(right, int), np.asarray(height)


def pair_cloud(basis, columns):
    total = sum(int(np.sum(columns == h)) * (int(np.sum(columns == h)) - 1)
                for h in np.unique(columns))
    first = np.empty(total, dtype=np.int32)
    second = np.empty(total, dtype=np.int32)
    ranges = np.empty((len(columns), 2), dtype=np.int64)
    offset = 0
    for neuron in np.unique(columns):
        positions = np.flatnonzero(columns == neuron)
        for point in positions:
            others = positions[positions != point]
            stop = offset + len(others)
            first[offset:stop], second[offset:stop] = point, others
            ranges[point] = offset, stop
            offset = stop
    signatures = np.empty((total, basis.shape[1]), dtype=np.float64)
    for start in range(0, total, 100000):
        local = slice(start, start + 100000)
        signatures[local] = basis[first[local]] - basis[second[local]]
    return first, second, ranges, signatures


def complete_partition(basis, columns, threshold, eligible, cloud, forest):
    """Return exact components, or a rigorous certificate of ineligibility.

    A subset of valid equal-pair edges can already leave fewer than m groups.
    Conversely, more than m isolated vertices rule out an m-group result.
    Only these certificates stop a scan early. Every retained partition has
    exhausted all remaining radius matches; nearest-neighbour screening never
    substitutes for the theorem's full radius rule.
    """
    first, second, ranges, signatures, norms, tree = cloud
    left, right, heights = forest
    active = heights <= threshold
    labels = np.arange(len(columns), dtype=np.int32)
    count, labels = merge(labels, left[active], right[active])
    low, high = min(eligible), max(eligible)
    if count < low:
        return None, {"certificate": "zero_components_below_budget", "groups": count}
    initial_sizes = np.bincount(labels)
    zero_singletons = initial_sizes[labels] == 1
    isolated, pending = 0, []
    order = np.random.default_rng(261112).permutation(len(columns))
    radius = 2 * threshold
    query_bound = max(np.nextafter(radius, np.inf), 1e-150)
    examined = 0
    for point in order:
        start, stop = ranges[point]
        query_ids = np.arange(start, stop)
        query_ids = query_ids[norms[query_ids] > threshold]
        if not len(query_ids):
            continue
        distance, neighbor = tree.query(
            signatures[query_ids], k=2, distance_upper_bound=query_bound, workers=1
        )
        valid = (neighbor < len(signatures)) & (distance <= radius)
        valid &= neighbor != query_ids[:, None]
        has_other = bool(valid.any())
        if not has_other:
            if zero_singletons[point]:
                isolated += 1
                if isolated > high:
                    return None, {
                        "certificate": "isolated_vertices_above_budget",
                        "isolated": isolated,
                        "prune_smaller": True,
                        "examined_vertices": examined + 1,
                    }
        else:
            pending.append(query_ids[valid.any(1)])
            query = np.broadcast_to(query_ids[:, None], neighbor.shape)[valid]
            match = neighbor[valid]
            use = norms[match] > threshold
            query, match = query[use], match[use]
            count, labels = merge(
                labels,
                np.r_[first[query], second[query]],
                np.r_[first[match], second[match]],
            )
            if count < low:
                return None, {
                    "certificate": "equal_pair_subset_below_budget",
                    "groups": count,
                    "examined_vertices": examined + 1,
                }
        examined += 1
    # k=2 above certified empty neighbourhoods. Exhaust every other query.
    for query_ids in pending:
        for start in range(0, len(query_ids), 8):
            batch = query_ids[start:start + 8]
            neighborhoods = tree.query_ball_point(
                signatures[batch], query_bound, workers=1, return_sorted=False
            )
            for query, neighbors in zip(batch, neighborhoods):
                match = np.asarray(neighbors, dtype=np.int64)
                match = match[norms[match] > threshold]
                if not len(match):
                    continue
                # query_bound is one ULP larger solely for inclusive retrieval.
                distance = np.linalg.norm(signatures[match] - signatures[query], axis=1)
                match = match[distance <= radius]
                count, labels = merge(
                    labels,
                    np.r_[np.full(len(match), first[query]),
                          np.full(len(match), second[query])],
                    np.r_[first[match], second[match]],
                )
                if count < low:
                    return None, {
                        "certificate": "full_radius_subset_below_budget",
                        "groups": count,
                        "examined_vertices": examined,
                    }
    return labels, {
        "certificate": "complete_radius_scan",
        "groups": count,
        "examined_vertices": examined,
    }


def recover_rank(basis, mask, rank, code_grid):
    rows, columns = np.where(mask)
    neurons = len(np.unique(columns))
    eligible = [m for m in code_grid if rank < m <= rank + neurons and m <= len(rows)]
    if not eligible:
        return [], {"rank": rank, "reason": "calibration_rank_incompatible"}
    if rank == 0:
        labels = np.searchsorted(np.unique(columns), columns)
        count = int(labels.max()) + 1
        if count not in eligible:
            return [], {"rank": rank, "groups": count}
        index = np.zeros(mask.shape, dtype=np.int16)
        index[rows, columns] = labels + 1
        return [{"index": index, "codes": count, "rank": rank,
                 "threshold": 0.0, "threshold_values": THRESHOLDS.tolist()}], {
                     "rank": rank, "groups": count, "complete": True}
    selected = np.ascontiguousarray(basis[:, :rank], dtype=np.float64)
    forest = zero_forest(selected, columns)
    first, second, ranges, signatures = pair_cloud(selected, columns)
    norms = np.linalg.norm(signatures, axis=1)
    tree = cKDTree(signatures)
    cloud = first, second, ranges, signatures, norms, tree
    proposals, diagnostics, seen = [], [], {}
    for threshold in reversed(THRESHOLDS):
        labels, detail = complete_partition(
            selected, columns, float(threshold), eligible, cloud, forest
        )
        detail["threshold"] = float(threshold)
        diagnostics.append(detail)
        if labels is not None:
            count = int(labels.max()) + 1
            if count in eligible:
                index = np.zeros(mask.shape, dtype=np.int16)
                index[rows, columns] = labels + 1
                if contrast_rank(index, count) == rank:
                    key = index.tobytes()
                    if key in seen:
                        proposals[seen[key]]["threshold_values"].append(float(threshold))
                    else:
                        seen[key] = len(proposals)
                        proposals.append({"index": index, "codes": count,
                                          "rank": rank, "threshold": float(threshold),
                                          "threshold_values": [float(threshold)]})
        if detail.get("prune_smaller"):
            break
    return proposals, {"rank": rank, "pairs": len(first), "thresholds": diagnostics}
