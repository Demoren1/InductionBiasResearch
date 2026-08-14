"""Data-driven (NO-LEAK) column alignment of importance maps via k-means.

Leak-free alternative to evaluation/align_importance.py: instead of aligning
hidden-unit columns of each importance map to the GOLD mask's columns (which
uses the answer), we:

  1. pool ALL importance-map columns across all 16 patterns (16*2000*8 = 256k
     columns of length SEQ_LEN=8),
  2. run k-means (k = N_WINDOWS = 5) on those columns (torch-only, Lloyd,
     deterministic via torch.Generator(seed=0)),
  3. canonicalize prototypes: proto_id 0 = cluster whose centroid has the
     smallest mean-row (weighted by centroid values), i.e. the earliest
     sliding window,
  4. for EVERY map, assign each of its 8 columns to the nearest centroid and
     reorder the columns into canonical prototype order: columns sorted by
     (proto_id, descending column L2 norm), ties by original index.

The gold mask / ideal_mask() is used ONLY in clearly-labeled diagnostics.

Outputs:
  - outputs/checkpoints/pattern_{pat}/importance_cluster.pt
        {"pattern", "importance": reordered (n,8,8)} for every pattern
  - outputs/checkpoints/col_prototypes.pt
        {"centroids": (5,8), "order": [canonical proto ids], "cluster_sizes"}
  - does NOT touch importance.pt / importance_aligned.pt.

Usage: python evaluation/cluster_align_importance.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from data.generate import ideal_mask  # noqa: E402 (diagnostic use only)

K = config.N_WINDOWS        # 5
SEED = 0                    # determinism for init / assignments
N_ITERS = 50                # Lloyd iterations
EPS = 1e-9


# --------------------------------------------------------------------------
# k-means (torch-only, deterministic)
# --------------------------------------------------------------------------

def kmeans_init(x: torch.Tensor, k: int, seed: int) -> torch.Tensor:
    """Pick k well-spread columns via a seeded random permutation.

    Shuffle all column indices with torch.Generator(manual_seed=seed) and take
    k evenly spaced entries -> 5 well-spread (across the dataset) starting
    points. Deterministic.
    """
    g = torch.Generator().manual_seed(seed)
    n = x.size(0)
    perm = torch.randperm(n, generator=g)
    idx = perm[torch.linspace(0, n - 1, k, dtype=torch.long)]
    return x[idx].clone()


def kmeans(x: torch.Tensor, k: int, seed: int = SEED,
           n_iters: int = N_ITERS) -> tuple:
    """Lloyd's k-means on (n, d) data. Returns (centroids (k,d), assign (n,))."""
    cents = kmeans_init(x, k, seed)
    assign = torch.zeros(x.size(0), dtype=torch.long)
    for it in range(n_iters):
        # (n, k) squared-L2 distances to centroids
        d2 = (x.unsqueeze(1) - cents.unsqueeze(0)).pow(2).sum(dim=-1)
        new_assign = d2.argmin(dim=-1)
        changed = (new_assign != assign).sum().item()
        assign = new_assign
        new_cents = cents.clone()
        for j in range(k):
            mask = assign == j
            cnt = mask.sum().item()
            if cnt > 0:
                new_cents[j] = x[mask].mean(dim=0)
            # empty clusters keep their previous centroid (checked later)
        cents = new_cents
        if changed == 0 and it > 0:
            break
    return cents, assign


# --------------------------------------------------------------------------
# metrics (diagnostics only)
# --------------------------------------------------------------------------

def pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float()
    b = b.flatten().float()
    ca = a - a.mean()
    cb = b - b.mean()
    denom = ca.norm() * cb.norm()
    if denom == 0:
        return 0.0
    return (ca @ cb / denom).item()


def topk_iou(x: torch.Tensor, gold: torch.Tensor, k: int) -> float:
    flat = x.flatten().float()
    _, idx = flat.topk(k)
    pred = torch.zeros_like(flat)
    pred[idx] = 1.0
    g = gold.flatten().float()
    inter = (pred * g).sum().item()
    union = pred.sum().item() + g.sum().item() - inter
    return inter / union if union > 0 else 0.0


def hungarian_cols(m: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Permute columns of (L,H) m to maximize overlap with (L,H) ref columns
    via greedy Hungarian on cost c[i,j] = -(m[:,i] @ ref[:,j])."""
    cost = -(m.float()[:, :, None] * ref.float()[:, None, :]).sum(dim=0)
    used = set()
    col_ind = []
    cost_c = cost.clone()
    for i in range(cost.shape[0]):
        j = (-cost_c[i]).argmax().item()
        while j in used:
            cost_c[i, j] = float("-inf")
            j = (-cost_c[i]).argmax().item()
        col_ind.append(j)
        used.add(j)
    aligned = torch.empty_like(m)
    aligned[:, col_ind] = m
    return aligned


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> None:
    # 1. load ALL 16 patterns' importance maps
    all_cols = []
    per_pat = {}
    for pat in config.PATTERNS:
        d = torch.load(config.pattern_dir(pat) / "importance.pt",
                       weights_only=True)
        imp = d["importance"].float()                 # (n, 8, 8)
        if imp.size(0) != 2000:
            print(f"[cluster] WARNING {pat}: n={imp.size(0)} != 2000")
        n = imp.size(0)
        all_cols.append(imp)                          # keep for reuse
        per_pat[pat] = imp
    X = torch.cat(all_cols, dim=0)                    # (16n, 8, 8)
    Xf = X.reshape(X.size(0), config.SEQ_LEN, config.H)
    cols = Xf.permute(0, 2, 1).reshape(-1, config.SEQ_LEN)   # (16n*8, 8)
    print(f"[cluster] loaded {X.size(0)} maps x {config.H} cols -> "
          f"{tuple(cols.shape)} columns (value range "
          f"[{cols.min().item():.3f}, {cols.max().item():.3f}])")

    # 2. k-means
    cents, assign = kmeans(cols, K, seed=SEED)
    sizes = torch.bincount(assign, minlength=K).tolist()

    # degenerate-centroid guard
    eq = 0
    for i in range(K):
        for j in range(i + 1, K):
            if (cents[i] - cents[j]).abs().max().item() < 1e-6:
                eq += 1
    if sizes.count(0) > 0 or eq > 0:
        print("[cluster] FATAL: degenerate k-means "
              f"(sizes={sizes}, identical-pairs={eq}). Stopping before "
              "training; inspection required.")
        sys.exit(1)

    # 3. diagnostics: centroids as 8x5 matrix etc.
    print("\n[cluster] === centroids (8 rows x 5 prototypes) ===")
    for r in range(config.SEQ_LEN):
        print(f"[cluster] row{r} " + "  ".join(f"{cents[i, r].item():.3f}"
                                               for i in range(K)))
    # best-matching contiguous-4-ones window (rows 0..4 start) by overlap
    gold = ideal_mask().float()                       # (8,8) (8x8 -> .float())
    windows = torch.zeros(config.N_WINDOWS, config.SEQ_LEN)
    for w in range(config.N_WINDOWS):
        windows[w, w:w + config.PATTERN_LEN] = 1.0
    best_w = []
    for i in range(K):
        ov = (cents[i].unsqueeze(0) * windows[:, :]).sum(dim=1)
        best_w.append(int(ov.argmax().item()))
    print("[cluster] per-prototype diagnostics "
          "(cluster_size, top-4 rows, best-matching window start):")
    for i in range(K):
        rows = cents[i].argsort(descending=True)[:4].tolist()
        print(f"  proto {i}: size={sizes[i]:6d}  top4_rows={rows}  "
              f"best_window_start={best_w[i]}")

    # canonical order: sort prototypes by weighted mean row index
    wsum = cents.sum(dim=1).clamp_min(EPS)
    wmean_row = (cents * torch.arange(config.SEQ_LEN, dtype=cents.dtype)
                 ).sum(dim=1) / wsum
    proto_order = torch.argsort(wmean_row).tolist()   # canonical proto ids
    print(f"[cluster] weighted mean row per proto: "
          f"{[f'{v:.3f}' for v in wmean_row.tolist()]}")
    print(f"[cluster] canonical proto order (earliest first): {proto_order}")

    # 4. reorder every map
    #    precompute per-column norms for tie-breaking
    col_norms = cols.norm(dim=1)                       # (16n*8,)
    reordered_parts = []
    for pat in config.PATTERNS:
        imp = per_pat[pat]                             # (n, 8, 8)
        n = imp.size(0)
        # per-map column assignment over the pooled centroids
        # map cols: (n, 8, 8) -> columns (n*8, 8)
        mc = imp.permute(0, 2, 1).reshape(n * config.H, config.SEQ_LEN)
        d2 = (mc.unsqueeze(1) - cents.unsqueeze(0)).pow(2).sum(dim=-1)  # (n*8,5)
        a = d2.argmin(dim=-1).view(n, config.H)       # (n, 8) per-map proto id
        m_norms = imp.norm(dim=1)                     # (n, 8) col L2 norms
        out = []
        for m_i in range(n):
            # key: (proto_id, -L2 norm, orig idx) -> deterministic
            order = sorted(range(config.H),
                           key=lambda h: (a[m_i, h].item(),
                                          -m_norms[m_i, h].item(),
                                          h))
            out.append(imp[m_i, :, order])
        reordered = torch.stack(out, dim=0)           # (n, 8, 8)
        out_path = config.pattern_dir(pat) / "importance_cluster.pt"
        torch.save({"pattern": pat, "importance": reordered}, out_path)
        print(f"[cluster] {pat}: wrote {out_path}  "
              f"col-proto hist: {a.flatten().bincount(minlength=5).tolist()}")
        reordered_parts.append(reordered)

# cluster assignment histogram (all columns, all patterns)
    hist = torch.bincount(assign, minlength=K)
    print(f"[cluster] cluster assignment histogram (all {cols.size(0)} "
          f"columns, pooled): {hist.tolist()}")

    # 5. save prototypes
    proto = {
        "centroids": cents,          # (5, 8)
        "order": proto_order,        # canonical proto ids
        "cluster_sizes": sizes,
        "mean_cluster_sizes": [sizes[i] for i in proto_order],
    }
    torch.save(proto, config.CKPT_DIR / "col_prototypes.pt")
    print(f"[cluster] saved {config.CKPT_DIR / 'col_prototypes.pt'}")

    # 6. pooled mean of the ALIGNED maps
    pooled_aligned = torch.cat(reordered_parts, dim=0).mean(dim=0)  # (8, 8)
    print("\n[cluster] POOLED mean aligned map (16 patterns, 32000 maps):")
    for r in range(config.SEQ_LEN):
        print(f"  " + "  ".join(f"{pooled_aligned[r, c].item():.3f}"
                                for c in range(config.H)))

    # -- DIAGNOSTIC-WITH-GOLD (labeled; not used in alignment) --
    print("\n[cluster] DIAGNOSTIC-WITH-GOLD (allowed for comparison only):")
    pa_raw = pearson(pooled_aligned, gold)
    ioa_raw = topk_iou(pooled_aligned, gold, int(gold.sum().item()))
    pooled_hun = hungarian_cols(pooled_aligned.clone(), gold)
    pa_hun = pearson(pooled_hun, gold)
    ioa_hun = topk_iou(pooled_hun, gold, int(gold.sum().item()))
    print(f"  pooled aligned (RAW canonical col order): "
          f"pearson={pa_raw:.4f} top32_iou={ioa_raw:.4f}")
    print(f"  pooled aligned (Hungarian col-permuted to gold): "
          f"pearson={pa_hun:.4f} top32_iou={ioa_hun:.4f}")

    # reference: gold-aligned run's pooled mean (previous leaky alignment)
    ref_parts = []
    for pat in config.PATTERNS:
        p = config.pattern_dir(pat) / "importance_aligned.pt"
        if p.exists():
            ref_parts.append(torch.load(p, weights_only=True)["importance"])
    if ref_parts:
        ref = torch.cat(ref_parts, dim=0).mean(dim=0)
        print(f"  (ref) gold-aligned pooled mean: "
              f"pearson={pearson(ref, gold):.4f} "
              f"top32_iou={topk_iou(ref, gold, int(gold.sum().item())):.4f}")


if __name__ == "__main__":
    main()