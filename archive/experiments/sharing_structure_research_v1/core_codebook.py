"""G11: independent codebook contrasts from globally certified partial groups."""

import numpy as np
from scipy.stats import t

from .interval_indexator import interval_indexator

CORE_METHODS = (
    "witness_codebook",
    "mean_core_codebook",
    "nonzero_gls_codebook",
    "full_gls_core_codebook",
)


def independent_codebook_contrasts(
    lower, upper, estimate, covariance, variance, degrees, codes, alpha
):
    """Return only uniquely assigned I; all contrast variances use full V."""
    _, core = interval_indexator(lower, upper, codes)
    if core["minimum_nonzero_codes"] != codes:
        return {}, {"core_certificate": core, "reason": "No certified K cores"}
    forced = np.asarray(core["forced_mask"])
    index = np.asarray(core["forced_indices"], dtype=np.int16)
    unknown = np.flatnonzero(~forced)
    zero = np.flatnonzero(forced & (index == 0))
    nonzero = np.flatnonzero(forced & (index != 0))
    flat = np.asarray(estimate, dtype=np.float64).ravel()
    covariance = np.asarray(covariance, dtype=np.float64)
    if covariance.shape != (len(flat), len(flat)):
        raise ValueError("Full covariance must match flattened output-major weights")
    candidates, details = {}, {}
    labels = np.r_[0, np.arange(1, codes + 1), -np.arange(1, codes + 1)]
    identity = np.eye(len(flat))
    witness = np.asarray(core["witness_positions"])
    for method in CORE_METHODS:
        code_operator = np.zeros((codes, len(flat)))
        weight_operator = identity[unknown].copy()
        if method in ("witness_codebook", "mean_core_codebook"):
            selected = witness if method == "witness_codebook" else nonzero
            for code in range(1, codes + 1):
                positions = selected[np.abs(index[selected]) == code]
                if len(positions) == 0:
                    raise ArithmeticError("Certified witnesses must cover all K groups")
                code_operator[code - 1, positions] = np.sign(index[positions]) / len(
                    positions
                )
        else:
            selected = (
                np.r_[nonzero, zero] if method == "full_gls_core_codebook" else nonzero
            )
            basis = (
                np.abs(index[selected, None]) == np.arange(1, codes + 1)
            ) * np.sign(index[selected, None])
            subcovariance = covariance[np.ix_(selected, selected)]
            precision_basis = np.linalg.solve(subcovariance, basis)
            local = np.linalg.solve(basis.T @ precision_basis, precision_basis.T)
            code_operator[:, selected] = local
            if method == "full_gls_core_codebook" and len(unknown):
                correction = np.linalg.solve(
                    subcovariance, covariance[np.ix_(selected, unknown)]
                ).T
                weight_operator -= correction @ (
                    identity[selected] - basis @ code_operator
                )
        code_estimate = code_operator @ flat
        contrast = np.repeat(weight_operator[:, None, :], 2 * codes + 1, axis=1)
        contrast[:, 1 : codes + 1] -= code_operator[None]
        contrast[:, codes + 1 :] += code_operator[None]
        differences = contrast @ flat
        contrast_variance = variance * np.einsum(
            "eak,kl,eal->ea", contrast, covariance, contrast, optimize=True
        )
        if np.any(
            contrast_variance
            < -1e-10 * max(1.0, float(np.max(contrast_variance, initial=0)))
        ):
            raise FloatingPointError("Negative full-covariance contrast variance")
        critical = float(
            t.isf(alpha / (2 * max(1, len(unknown)) * (2 * codes + 1)), degrees)
        )
        accepted = np.abs(differences) <= critical * np.sqrt(
            np.maximum(contrast_variance, 0)
        )
        counts = accepted.sum(-1)
        result = index.copy()
        if np.all(counts == 1):
            result[unknown] = labels[accepted.argmax(-1)]
            candidates[method] = result.reshape(estimate.shape)
        details[method] = {
            "returned_index": method in candidates,
            "forced_nonzero": len(nonzero),
            "forced_zero": len(zero),
            "remaining": len(unknown),
            "ambiguous_remaining": int((counts > 1).sum()),
            "infeasible_remaining": int((counts == 0).sum()),
            "code_estimate": code_estimate.tolist(),
            "code_variance": (
                variance
                * np.einsum("ik,kl,il->i", code_operator, covariance, code_operator)
            ).tolist(),
            "student_degrees": degrees,
            "student_critical": critical,
        }
    return candidates, {"core_certificate": core, "methods": details}
