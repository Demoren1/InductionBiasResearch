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
    total = sum(
        int(np.sum(columns == h)) * (int(np.sum(columns == h)) - 1)
        for h in np.unique(columns)
    )
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


def calibration_components(labels, columns):
    representatives = np.full(int(labels.max()) + 1, len(labels), dtype=np.int64)
    np.minimum.at(representatives, labels, np.arange(len(labels)))
    graph = csr_matrix(
        (np.ones(len(labels), dtype=bool), (columns, columns[representatives[labels]])),
        shape=(int(columns.max()) + 1,) * 2,
    )
    _, components = connected_components(graph, directed=False)
    return len(np.unique(components[np.unique(columns)]))


def complete_partition(basis, columns, threshold, eligible, cloud, forest):
    """Exhaust radius rules or reject using a monotone partial-graph bound.

    Components and calibration-graph components can only merge as valid edges
    are added. More than m isolated vertices also forbid an m-group result.
    A returned partition has exhausted every unexcluded radius neighbourhood.
    """
    first, second, ranges, signatures, norms, tree, neuron_maxima = cloud
    if hasattr(tree, "threshold"):
        tree.threshold = threshold
    left, right, heights = forest
    active = heights <= threshold
    labels = np.arange(len(columns), dtype=np.int32)
    count, labels = merge(labels, left[active], right[active])
    rank = basis.shape[1]
    inactive_neurons = int(np.sum(neuron_maxima <= threshold))
    minimum_calibration_components = inactive_neurons + int(
        inactive_neurons < len(neuron_maxima)
    )
    eligible = [m for m in eligible if m - rank >= minimum_calibration_components]
    if not eligible:
        return None, {
            "certificate": "inactive_neurons_force_too_many_calibration_components",
            "inactive_neurons": inactive_neurons,
            "minimum_calibration_components": minimum_calibration_components,
        }
    low, high = min(eligible), max(eligible)

    def rejection(current_count, current_labels):
        if current_count < low:
            return {
                "certificate": "partial_groups_below_budget",
                "groups": current_count,
            }
        components = calibration_components(current_labels, columns)
        if current_count - components < rank:
            return {
                "certificate": "partial_contrast_rank_below_required",
                "contrast_rank": current_count - components,
                "calibration_components": components,
                "groups": current_count,
            }
        if components < low - rank:
            return {
                "certificate": "partial_calibration_components_below_required",
                "calibration_components": components,
                "groups": current_count,
            }
        return None

    reason = rejection(count, labels)
    if reason is not None:
        return None, reason
    zero_singletons = np.bincount(labels)[labels] == 1
    all_pairs_nonzero = bool(norms.min(initial=np.inf) > threshold)
    globally_isolated = 0
    currently_isolated = 0
    processed = np.zeros(len(columns), dtype=bool)
    leverage = np.sum(basis ** 2, axis=1)
    radius = 2 * threshold
    query_bound = max(np.nextafter(radius, np.inf), 1e-150)
    examined = 0
    while not processed.all():
        sizes = np.bincount(labels)
        unfinished = np.bincount(labels, weights=~processed, minlength=len(sizes))
        available = np.flatnonzero(unfinished)
        minimum_size = sizes[available].min()
        choices = np.flatnonzero((sizes[labels] == minimum_size) & ~processed)
        point = choices[np.argmax(leverage[choices])]
        start, stop = ranges[point]
        query_ids = np.arange(start, stop)
        query_ids = query_ids[norms[query_ids] > threshold]
        point_possible = False
        point_touched = False
        for begin in range(0, len(query_ids), 256):
            batch = query_ids[begin : begin + 256]
            if hasattr(tree, "iter_radius_edges"):
                edge_batches = tree.iter_radius_edges(signatures[batch], batch, radius)
            else:
                neighborhoods = tree.query_ball_point(
                    signatures[batch], query_bound, workers=1, return_sorted=False
                )
                local_rows = np.repeat(
                    np.arange(len(batch)), [len(x) for x in neighborhoods]
                )
                matches = np.concatenate(neighborhoods).astype(np.int64)
                other = matches != batch[local_rows]
                possible = np.zeros(len(batch), dtype=bool)
                possible[local_rows[other]] = True
                edge_batches = [(local_rows[other], matches[other], possible)]
            for local_query, match, possible in edge_batches:
                point_possible |= bool(possible.any())
                query = batch[local_query]
                use = norms[match] > threshold
                use &= (labels[first[query]] != labels[first[match]]) | (
                    labels[second[query]] != labels[second[match]]
                )
                query, match = query[use], match[use]
                if not len(match):
                    continue
                distance = np.linalg.norm(signatures[match] - signatures[query], axis=1)
                keep = distance <= radius
                query, match = query[keep], match[keep]
                point_touched |= bool(np.any(first[match] != point))
                count, labels = merge(
                    labels,
                    np.r_[first[query], second[query]],
                    np.r_[first[match], second[match]],
                )
                reason = rejection(count, labels)
                if reason is not None:
                    return None, {**reason, "examined_vertices": examined + 1}
        if (
            zero_singletons[point]
            and not point_touched
            and np.count_nonzero(labels == labels[point]) == 1
        ):
            currently_isolated += 1
            # If every query norm exceeds 3t, excluded cloud signatures of
            # norm <=t cannot touch this vertex at radius 2t, or any smaller
            # threshold. All other matches have been checked directly.
            if (
                not point_possible
                or norms[query_ids].min(initial=np.inf) > 3 * threshold
            ):
                globally_isolated += 1
                if globally_isolated > high:
                    return None, {
                        "certificate": "isolated_vertices_above_budget",
                        "isolated": globally_isolated,
                        "prune_smaller": True,
                        "examined_vertices": examined + 1,
                    }
            if currently_isolated > high:
                return None, {
                    "certificate": "current_isolated_vertices_above_budget",
                    "isolated": currently_isolated,
                    "examined_vertices": examined + 1,
                    "prune_smaller": all_pairs_nonzero,
                }
        examined += 1
        processed[point] = True
        sizes = np.bincount(labels)
        finished = np.bincount(labels, weights=processed, minlength=len(sizes))
        closed = int(np.sum(finished == sizes))
        lower_bound = closed + int(not processed.all())
        if lower_bound > high:
            return None, {
                "certificate": "closed_components_above_budget",
                "closed_components": closed,
                "minimum_final_groups": lower_bound,
                "examined_vertices": examined,
                "prune_smaller": all_pairs_nonzero,
            }
    return labels, {
        "certificate": "complete_radius_scan",
        "groups": count,
        "examined_vertices": examined,
    }


def recover_rank(basis, mask, rank, code_grid, device=None, progress=None):
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
        return [
            {
                "index": index,
                "codes": count,
                "rank": rank,
                "threshold": 0.0,
                "threshold_values": THRESHOLDS.tolist(),
            }
        ], {"rank": rank, "groups": count, "complete": True}
    selected = np.ascontiguousarray(basis[:, :rank], dtype=np.float64)
    forest = zero_forest(selected, columns)
    first, second, ranges, signatures = pair_cloud(selected, columns)
    norms = np.linalg.norm(signatures, axis=1)
    if device is None:
        tree = cKDTree(signatures)
    else:
        from .gpu_neighbors import GpuNeighbors

        tree = GpuNeighbors(signatures, device, norms)
    neuron_maxima = []
    for neuron in np.unique(columns):
        positions = np.flatnonzero(columns == neuron)
        start, stop = ranges[positions[0], 0], ranges[positions[-1], 1]
        neuron_maxima.append(float(norms[start:stop].max(initial=0.0)))
    cloud = first, second, ranges, signatures, norms, tree, np.asarray(neuron_maxima)
    proposals, diagnostics, seen = [], [], {}
    for threshold in reversed(THRESHOLDS):
        labels, detail = complete_partition(
            selected, columns, float(threshold), eligible, cloud, forest
        )
        detail["threshold"] = float(threshold)
        diagnostics.append(detail)
        if progress is not None:
            progress({"rank": rank, **detail})
        if labels is not None:
            count = int(labels.max()) + 1
            if count in eligible:
                index = np.zeros(mask.shape, dtype=np.int16)
                index[rows, columns] = labels + 1
                if contrast_rank(index, count) == rank:
                    key = index.tobytes()
                    if key in seen:
                        proposals[seen[key]]["threshold_values"].append(
                            float(threshold)
                        )
                    else:
                        seen[key] = len(proposals)
                        proposals.append(
                            {
                                "index": index,
                                "codes": count,
                                "rank": rank,
                                "threshold": float(threshold),
                                "threshold_values": [float(threshold)],
                            }
                        )
        if detail.get("prune_smaller"):
            break
    return proposals, {"rank": rank, "pairs": len(first), "thresholds": diagnostics}
