"""Conditional stability of rounding to fitted signed code centers.

The check freezes fitted centers and uses supplied direction uncertainties.
It does not certify center estimates, dense accuracy, or unique identification.
"""

import numpy as np

from .srr_local import fit_magnitudes, score_on_tasks


def assignment_margin(index, weights, uncertainty, codes, radius=2.0):
    """Distance to each competing assignment's bisector in whitened weights."""
    if not index.any() or len(np.unique(np.abs(index[index != 0]))) != codes:
        return {
            "accepted": False,
            "reason": "incomplete_index",
            "ambiguous_positions": int(index.size),
        }
    _, checks = score_on_tasks(index, weights, uncertainty, codes)
    keep = max(1, (len(weights) + 1) // 2)
    selected = np.argsort([row["score"] for row in checks])[:keep]
    observations, alternatives = [], []
    for task in selected:
        mapping = np.array(checks[task]["mapping"])
        normal = weights[task][:, mapping]
        deviation = np.maximum(uncertainty[task][mapping], 1e-10)
        fit = fit_magnitudes(index, normal, deviation, codes)
        centers = np.r_[0.0, fit["code"], -fit["code"]]
        alternatives.append(centers[None] * (fit["scales"] / deviation)[:, None])
        observations.append(normal / deviation[None])
    observations = np.stack(observations, axis=-1)
    alternatives = np.stack(alternatives, axis=-1)
    cost = np.sum((observations[:, :, None] - alternatives[None]) ** 2, axis=-1)
    chosen = np.where(index >= 0, index, codes - index).astype(int)
    chosen_cost = np.take_along_axis(cost, chosen[..., None], axis=-1)
    assigned_center = alternatives[np.arange(index.shape[1])[None], chosen]
    distance = np.linalg.norm(alternatives[None] - assigned_center[:, :, None], axis=-1)
    difference = cost - chosen_cost
    margin = difference / np.maximum(2 * distance, 1e-20)
    # Identical competing centers are unresolved; a self-comparison is omitted.
    margin[distance < 1e-12] = 0.0
    np.put_along_axis(margin, chosen[..., None], np.inf, axis=-1)
    per_position = margin.min(-1)
    ambiguous = per_position <= radius
    return {
        "accepted": bool(not ambiguous.any()),
        "radius": radius,
        "minimum_margin": float(per_position.min()),
        "ambiguous_positions": int(ambiguous.sum()),
        "ambiguous_mask": ambiguous,
        "selected_tasks": selected.tolist(),
        "interpretation": "Stability to radius2 perturbations with fitted centers fixed, not a certificate of true I",
    }
