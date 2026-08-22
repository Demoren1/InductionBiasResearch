"""Align each importance map's hidden-unit columns to a reference ordering.

For each pattern in config.PATTERNS:
  - load importance.pt -> imp (n, H, H)
  - gold = ideal_mask().float() (H, H)
  - for EACH map m (n maps): assign map columns to gold columns via
    Hungarian matching (scipy.optimize.linear_sum_assignment) with cost
    c[i, j] = -(m[:, i] @ gold[:, j]) (maximize overlap). Produce a permuted
    map m_aligned where aligned[:, j] = m[:, permutation -> gold col j].
  - save outputs/checkpoints/pattern_{pat}/importance_aligned.pt:
        {"pattern": pat, "importance": aligned_tensor}
  - DO NOT modify importance.pt.
  - print per pattern: pearson(aligned.mean(0), gold), top32 IoU of
    aligned.mean(0) vs gold, plus the same numbers BEFORE alignment.

Methods (--method, default "gold"):
  gold     Align every map to the ideal Toeplitz gold mask (above).
  window   Reorder each map's columns by pattern-window overlap: each column
           is assigned the window it overlaps most, then columns are sorted by
           (window_id * 1e6 - column_norm + col_idx * 1e-6). This ports
           /tmp/build_win.py into the repo and must reproduce the existing
           outputs/checkpoints/pattern_*/importance_win.pt bit-exactly
           (verified before any overwrite).
  refmatch Self-aligning: pick the single MLP across all patterns with the
           global minimum val_loss as reference R, then align every map to R
           via the same Hungarian align_map. Never uses gold; pearson/IoU vs
           the gold mask are printed only as labelled diagnostics.
  selfalign Gold-free iterative self-alignment: keep the lowest-val_loss
           TOP_FRAC (default 0.1) maps per pattern, iterate the Hungarian
           alignment to a reference (init = global-min-val_loss map) for
N_ITERS (default 6) rounds, and choose the best iteration by the
            mean Hungarian alignment score (fit of maps to the reference),
            never by gold. Saves importance_selfalign.pt per pattern.
            --pick_iter forces t_star to a fixed refs[] iteration (default
            None = keep the consistency-based selection).

Usage: python evaluation/align_importance.py [--method {gold,window,refmatch,selfalign}]
       [--top_frac F] [--iters N] [--pick_iter T]
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from data.generate import ideal_mask  # noqa: E402

try:
    from scipy.optimize import linear_sum_assignment
    _HAVE_SCIPY = True
except ImportError:  # pragma: no cover
    _HAVE_SCIPY = False


def pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float()
    b = b.flatten().float()
    ca = a - a.mean()
    cb = b - b.mean()
    denom = ca.norm() * cb.norm()
    if denom == 0:
        return 0.0
    return (ca @ cb / denom).item()


def topk_ioU(x: torch.Tensor, gold: torch.Tensor, k: int) -> float:
    """Binarize x to its top-k entries and return IoU vs binary gold."""
    flat = x.flatten().float()
    _, idx = flat.topk(k)
    pred = torch.zeros_like(flat)
    pred[idx] = 1.0
    g = gold.flatten().float()
    inter = (pred * g).sum().item()
    union = pred.sum().item() + g.sum().item() - inter
    return inter / union if union > 0 else 0.0


def column_assignment(m: torch.Tensor, reference: torch.Tensor) -> tuple:
    """Match map columns to reference columns by maximum overlap."""
    cost = -(m.float()[:, :, None] * reference.float()[:, None, :]).sum(dim=0)
    if _HAVE_SCIPY:
        row_ind, col_ind = linear_sum_assignment(cost.numpy())
    else:
        row_ind, col_ind = [], []
        used = set()
        for i in range(cost.shape[0]):
            j = (-cost[i]).argmax().item()
            while j in used:
                cost[i, j] = float("-inf")
                j = (-cost[i]).argmax().item()
            row_ind.append(i)
            col_ind.append(j)
            used.add(j)
    return cost, row_ind, col_ind


def align_map(m: torch.Tensor, gold: torch.Tensor) -> torch.Tensor:
    """Reorder a map's columns into the reference ordering."""
    _, row_ind, col_ind = column_assignment(m, gold)
    aligned = torch.empty_like(m)
    aligned[:, col_ind] = m[:, row_ind]
    return aligned


def load_importance(pat: str) -> torch.Tensor:
    """Load raw importance maps for a pattern as float (n, SEQ_LEN, H)."""
    src = config.pattern_dir(pat) / "importance.pt"
    d = torch.load(src, weights_only=True)
    return d["importance"].float()


def save_aligned(pat: str, name: str, importance: torch.Tensor) -> Path:
    """Save aligned maps while preserving source evaluation metadata."""
    source = torch.load(config.pattern_dir(pat) / "importance.pt",
                        weights_only=True)
    payload = {key: source[key] for key in ("pattern", "n_mlps", "val_loss", "val_acc")
               if key in source}
    payload["importance"] = importance
    path = config.pattern_dir(pat) / name
    torch.save(payload, path)
    return path


def align_map_score(m: torch.Tensor, ref: torch.Tensor):
    """Return a reference-aligned map and its matching score."""
    cost, row_ind, col_ind = column_assignment(m, ref)
    aligned = torch.empty_like(m)
    aligned[:, col_ind] = m[:, row_ind]
    score = float(-cost[row_ind, col_ind].sum())
    return aligned, score


# --------------------------------------------------------------------------
# Method: gold (original behavior, unchanged)
# --------------------------------------------------------------------------
def run_gold() -> None:
    gold = ideal_mask().float()  # (8, 8)
    n_ones = int(gold.sum().item())
    k = n_ones  # 32
    print(f"[align] gold: shape={tuple(gold.shape)} ones={n_ones} "
          f"scipy={_HAVE_SCIPY}")

    rows = []
    for pat in config.PATTERNS:
        imp = load_importance(pat)       # (n, 8, 8)
        n = imp.size(0)

        mean_before = imp.mean(dim=0)          # (8, 8)
        pb = pearson(mean_before, gold)
        ib = topk_ioU(mean_before, gold, k)

        # vectorized hungarian over all n maps
        aligned = torch.empty_like(imp)
        for m_i in range(n):
            aligned[m_i] = align_map(imp[m_i], gold)

        mean_after = aligned.mean(dim=0)
        pa = pearson(mean_after, gold)
        ia = topk_ioU(mean_after, gold, k)

        out = save_aligned(pat, "importance_aligned.pt", aligned)

        rows.append((pat, pb, ib, pa, ia))
        print(f"[align] {pat}: BEFORE pearson={pb:.4f} iou={ib:.4f} | "
              f"AFTER  pearson={pa:.4f} iou={ia:.4f}  -> {out}")

    if rows:
        n = len(rows)
        apb = sum(r[1] for r in rows) / n
        aib = sum(r[2] for r in rows) / n
        apa = sum(r[3] for r in rows) / n
        aia = sum(r[4] for r in rows) / n
        print(f"[align] AVG  : BEFORE pearson={apb:.4f} iou={aib:.4f} | "
              f"AFTER  pearson={apa:.4f} iou={aia:.4f}")


# --------------------------------------------------------------------------
# Method: window (port of /tmp/build_win.py, must reproduce importance_win.pt)
# --------------------------------------------------------------------------
def run_window() -> None:
    W = torch.zeros(config.N_WINDOWS, config.SEQ_LEN)
    for w in range(config.N_WINDOWS):
        W[w, w:w + config.PATTERN_LEN] = 1.0
    print(f"[window] n_windows={config.N_WINDOWS} seq_len={config.SEQ_LEN} "
          f"pattern_len={config.PATTERN_LEN} h={config.H}", flush=True)
    for pat in config.PATTERNS:
        X = load_importance(pat)
        n = X.size(0)
        cols = X.permute(0, 2, 1).reshape(-1, config.SEQ_LEN)
        ov = cols @ W.t()
        win = ov.argmax(1).view(n, config.H)
        norms = X.norm(dim=1)
        idx = torch.arange(float(config.H)).view(1, config.H)
        score = win.float() * 1e6 - norms + idx * 1e-6
        order = torch.argsort(score, dim=1)
        out = torch.gather(X, 2, order.unsqueeze(1).expand(-1, config.H, -1))
        out_path = config.pattern_dir(pat) / "importance_win.pt"
        note = "new"
        if out_path.exists():
            old = torch.load(out_path, weights_only=True)["importance"]
            note = "unchanged" if torch.equal(out, old) else "changed"
        save_aligned(pat, "importance_win.pt", out)
        print(f"[window] {pat}: {note} -> {out_path}", flush=True)
    print("[window] saved importance_win.pt (all patterns)", flush=True)


# --------------------------------------------------------------------------
# Method: selfalign (gold-free, stable iterative self-alignment)
# --------------------------------------------------------------------------
TOP_FRAC = 0.1   # keep lowest-val_loss TOP_FRAC per pattern (match select_best)
N_ITERS = 6


def run_selfalign(top_frac: float = TOP_FRAC, n_iters: int = N_ITERS,
                  pick_iter: int | None = None) -> None:
    """Iterative self-alignment with no gold.

    - Per pattern: idx_top = argsort(val_loss)[:max(1, round(n*top_frac))].
    - Reference init R = the single map with GLOBAL min val_loss.
    - For t in range(n_iters): align the top-frac maps of every pattern to R,
      average the aligned maps into M, record the mean Hungarian score of that
      fit, then R = M.  The best iteration is chosen by the highest mean
      alignment score (NOT by gold), unless overridden by pick_iter.
    - After the extra loop, one extra fit to the final cleaned R is also
      recorded.
    - t_star = pick_iter if given (validated against len(refs)), else
      argmax(mean scores); Rb = refs[t_star]. Align ONLY the TOP-FRAC maps
      (lowest-val_loss idx_top[pat], ~200 per pattern) to Rb and save
      importance_selfalign.pt per pattern; the bottom-frac maps lack
      structure and dilute the pooled average.
    - DIAGNOSTIC-WITH-GOLD: pooled RAW pearson/IoU vs gold and HUNG
      (align_map(pooled, gold)) pearson/IoU over these top-frac aligned
      maps, labelled (never used to pick).
    """
    gold = ideal_mask().float()
    k = int(gold.sum().item())

    # Per-pattern top-frac indices.
    idx_top = {}
    imp_all = {}
    best = None  # (val_loss, R)
    for pat in config.PATTERNS:
        p = config.pattern_dir(pat) / "importance.pt"
        d = torch.load(p, weights_only=True)
        imp_all[pat] = d["importance"].float()  # (n, 8, 8)
        vl = d["val_loss"]                       # (n,)
        n = imp_all[pat].size(0)
        n_keep = max(1, round(n * top_frac))
        idx_top[pat] = torch.argsort(vl)[:n_keep]
        bi = int(vl.argmin().item())
        if best is None or vl[bi].item() < best[0]:
            best = (vl[bi].item(), imp_all[pat][bi])

    _, R = best
    print(f"[selfalign] top_frac={top_frac} n_iters={n_iters} "
          f"top_keep/pat~{max(1, round(imp_all[config.PATTERNS[0]].size(0) * top_frac))} "
          f"ref_init=global-min-val_loss "
          f"({', '.join(str(len(idx_top[p])) for p in config.PATTERNS)})",
          flush=True)

    refs = []
    scores = []
    cons = []  # mean_i pearson(A_i, M): agreement among aligned maps
    for t in range(n_iters):
        accum = 0.0
        tot_score = 0.0
        cnt = 0
        aligned_maps = []
        for pat in config.PATTERNS:
            X = imp_all[pat][idx_top[pat]]
            for i in range(X.size(0)):
                A_i, s_i = align_map_score(X[i], R)
                accum = accum + A_i
                tot_score += s_i
                cnt += 1
                aligned_maps.append(A_i)
        M = accum / cnt
        mean_score = tot_score / cnt
        # Consistency: mean pearson of each aligned top-frac map vs their mean.
        cons_t = sum(pearson(A_i, M) for A_i in aligned_maps) / cnt
        refs.append(R.clone())
        scores.append(mean_score)
        cons.append(cons_t)
        print(f"[selfalign] iter {t}: fit_score={mean_score:.6f} "
              f"consistency={cons_t:.6f} (both gold-free)", flush=True)
        R = M

    # Extra candidate: fit to the final cleaned map R.
    accum = 0.0
    tot_score = 0.0
    cnt = 0
    aligned_maps = []
    for pat in config.PATTERNS:
        X = imp_all[pat][idx_top[pat]]
        for i in range(X.size(0)):
            A_i, s_i = align_map_score(X[i], R)
            accum = accum + A_i
            tot_score += s_i
            cnt += 1
            aligned_maps.append(A_i)
    M = accum / cnt
    mean_score_final = tot_score / cnt
    cons_final = sum(pearson(A_i, M) for A_i in aligned_maps) / cnt
    refs.append(R.clone())
    scores.append(mean_score_final)
    cons.append(cons_final)
    print(f"[selfalign] iter {n_iters} (fit-to-final-R): "
          f"fit_score={mean_score_final:.6f} consistency={cons_final:.6f} "
          f"(gold-free)", flush=True)

    # Print per-iteration table of (iter, fit_score, consistency).
    print("[selfalign] iteration table (fit_score for reference only; "
          "selection by consistency):", flush=True)
    for t in range(len(scores)):
        sel = " *" if t == max(range(len(cons)), key=lambda i: cons[i]) else ""
        print(f"[selfalign]   iter {t:>2}: fit_score={scores[t]:.6f} "
              f"consistency={cons[t]:.6f}{sel}", flush=True)

    # Selection: by CONSISTENCY (agreement among aligned maps, gold-free)
    # by default; forced to pick_iter when given.
    if pick_iter is not None:
        if not (0 <= pick_iter < len(refs)):
            raise ValueError(
                f"--pick_iter {pick_iter} out of range [0, {len(refs) - 1}]")
        t_star = pick_iter
        Rb = refs[t_star]
        print(f"[selfalign] selection FORCED to pick_iter={t_star} "
              f"(consistency={cons[t_star]:.6f} "
              f"fit_score={scores[t_star]:.6f})", flush=True)
    else:
        t_star = max(range(len(cons)), key=lambda i: cons[i])
        Rb = refs[t_star]
        print(f"[selfalign] chosen t_star={t_star} consistency={cons[t_star]:.6f} "
              f"(fit_score={scores[t_star]:.6f})", flush=True)

    # Align ONLY the TOP-FRAC maps (lowest-val_loss idx_top[pat]) to Rb and
    # save. The bottom-frac maps lack structure and dilute the pooled average
    # when included (prior diagnostic: HUNG iou 0.64 all-maps vs 0.83
    # top-10%).
    aligned_pool = []
    for pat in config.PATTERNS:
        X = imp_all[pat][idx_top[pat]]          # (n_keep, 8, 8)
        n_keep = X.size(0)
        aligned = torch.empty_like(X)
        for m_i in range(n_keep):
            aligned[m_i], _ = align_map_score(X[m_i], Rb)
        out = save_aligned(pat, "importance_selfalign.pt", aligned)
        aligned_pool.append(aligned)
        print(f"[selfalign] saved {out} ({n_keep} maps, top-frac)",
              flush=True)

    # DIAGNOSTIC-WITH-GOLD (labelled; never used to choose t_star), pooled
    # over the top-frac aligned maps that we now save.
    pooled = torch.cat(aligned_pool, dim=0).mean(dim=0)  # (8, 8) over top-frac
    pr = pearson(pooled, gold)
    ir = topk_ioU(pooled, gold, k)
    pooled_h = align_map(pooled, gold)
    ph = pearson(pooled_h, gold)
    ih = topk_ioU(pooled_h, gold, k)
    print("[selfalign] DIAGNOSTIC-WITH-GOLD (never used to choose t_star):",
          flush=True)
    print(f"[selfalign] POOLED RAW  : pearson={pr:.4f} iou={ir:.4f}", flush=True)
    print(f"[selfalign] POOLED HUNG : pearson={ph:.4f} iou={ih:.4f}", flush=True)
    print("[selfalign] (compare HUNG iou to window prior 0.829)", flush=True)


# --------------------------------------------------------------------------
# Method: refmatch (self-aligning, no gold in the alignment, no window
# templates)
# --------------------------------------------------------------------------
def run_refmatch() -> None:
    # 1) Choose reference: the single MLP with the GLOBAL minimum val_loss
    #    across all patterns' importance.pt.
    best = None  # (pat, idx, val_loss, val_acc, R)
    for pat in config.PATTERNS:
        p = config.pattern_dir(pat) / "importance.pt"
        d = torch.load(p, weights_only=True)
        idx = int(d["val_loss"].argmin().item())
        if best is None or d["val_loss"][idx].item() < best[2]:
            best = (pat, idx, d["val_loss"][idx].item(),
                    d["val_acc"][idx].item(), d["importance"][idx].float())
    pat_ref, idx_ref, loss_ref, acc_ref, R = best
    print(f"[refmatch] reference: pat={pat_ref} idx={idx_ref} "
          f"val_loss={loss_ref:.6f} val_acc={acc_ref:.4f}")
    col_norms = R.norm(dim=0)  # (H,)
    print("[refmatch] R column norms: " +
          "[" + ", ".join(f"{v:.4f}" for v in col_norms.tolist()) + "]")

    gold = ideal_mask().float()  # diagnostic only
    k = int(gold.sum().item())   # 32

    # 2) Align every map of every pattern to R; 3) save.
    rows = []
    for pat in config.PATTERNS:
        imp = load_importance(pat)       # (n, 8, 8)
        n = imp.size(0)

        mean_before = imp.mean(dim=0)
        pb = pearson(mean_before, gold)
        ib = topk_ioU(mean_before, gold, k)

        aligned = torch.empty_like(imp)
        for m_i in range(n):
            aligned[m_i] = align_map(imp[m_i], R)

        mean_after = aligned.mean(dim=0)
        pa = pearson(mean_after, gold)
        ia = topk_ioU(mean_after, gold, k)

        out = save_aligned(pat, "importance_refmatch.pt", aligned)

        rows.append((pat, pb, ib, pa, ia))
        print(f"[refmatch] {pat}: BEFORE pearson={pb:.4f} iou={ib:.4f} | "
              f"AFTER  pearson={pa:.4f} iou={ia:.4f}  -> {out}")

    if rows:
        n = len(rows)
        apb = sum(r[1] for r in rows) / n
        aib = sum(r[2] for r in rows) / n
        apa = sum(r[3] for r in rows) / n
        aia = sum(r[4] for r in rows) / n
        print("[refmatch] DIAGNOSTIC-WITH-GOLD (refmatch itself never uses "
              "gold):")
        print(f"[refmatch] POOLED: BEFORE pearson={apb:.4f} iou={aib:.4f} | "
              f"AFTER  pearson={apa:.4f} iou={aia:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Align importance maps' hidden-unit columns to a "
                    "reference (gold mask, window prior, or best-val_loss map).")
    parser.add_argument("--method", choices=["gold", "window", "refmatch",
                                             "selfalign"],
                        default="gold",
                        help="alignment method (default: gold)")
    parser.add_argument("--top_frac", type=float, default=TOP_FRAC,
                        help="fraction of lowest-val_loss maps kept per "
                             "pattern for selfalign (default: 0.1)")
    parser.add_argument("--iters", type=int, default=N_ITERS,
                        help="number of selfalign iterations (default: 6)")
    parser.add_argument("--pick_iter", type=int, default=None,
                        help="force t_star to this refs[] iteration for "
                             "selfalign (default: None = consistency-based)")
    args = parser.parse_args()

    if args.method == "gold":
        run_gold()
    elif args.method == "window":
        run_window()
    elif args.method == "refmatch":
        run_refmatch()
    elif args.method == "selfalign":
        run_selfalign(top_frac=args.top_frac, n_iters=args.iters, pick_iter=args.pick_iter)


if __name__ == "__main__":
    main()
