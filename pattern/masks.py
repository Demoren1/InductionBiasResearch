"""Exact cardinality without imposing a Toeplitz structure."""
import torch

def exact_topk(logits, k=32, straight_through=False):
    if logits.shape[-2:] != (11, 8) or not 1 <= k <= 88:
        raise ValueError("expected [...,11,8] scores and valid K")
    if not torch.isfinite(logits).all():
        raise ValueError("mask scores must be finite")
    flat = logits.flatten(-2)
    # Stable ties favor the smaller input/hidden index, without perturbing scores.
    sort_values = flat.detach().cpu() if flat.device.type == "mps" else flat
    indices = sort_values.argsort(dim=-1, descending=True, stable=True)[..., :k].to(flat.device)
    hard = torch.zeros_like(flat).scatter(-1, indices, 1).reshape_as(logits)
    if straight_through:
        soft = logits.sigmoid()
        return hard + (soft - soft.detach())
    return hard

def canonical_columns(q_abs):
    """Order hidden columns by functional centroid, then their entire profile."""
    q_abs = q_abs.detach().cpu()
    coordinates = torch.arange(11, dtype=torch.float64)
    profiles = q_abs.double()
    mass = profiles.sum(0)
    centroid = torch.where(mass > 0, (coordinates[:, None]*profiles).sum(0)/mass,
                           torch.full_like(mass, float("inf")))
    return torch.tensor(sorted(range(8), key=lambda j:
        (float(centroid[j]), tuple(profiles[:, j].tolist()))), dtype=torch.long)

def toeplitz_metrics(mask):
    """Distance to orthogonal projection onto matrices constant on diagonals."""
    mask = mask.float().cpu()
    rows, cols = torch.meshgrid(torch.arange(11), torch.arange(8), indexing="ij")
    diagonal = rows-cols
    projection = torch.zeros_like(mask)
    for offset in diagonal.unique():
        selected = diagonal == offset
        projection[selected] = mask[selected].mean()
    mse = (mask-projection).square().mean()
    return {"toeplitz_mse": float(mse), "toeplitz_exact": bool(mse < 1e-12),
            "diagonal_agreement": float((mask[1:,1:] == mask[:-1,:-1]).float().mean())}


def analytical_mask():
    """Four translated length-4 windows: 32 edges, independent of task labels."""
    rows, cols = torch.meshgrid(torch.arange(11), torch.arange(8), indexing="ij")
    return ((rows-cols >= 0) & (rows-cols < 4)).float()


def aligned_iou(masks, reference=None):
    """Binary IoU after optimal hidden-column assignment; input rows stay fixed."""
    from scipy.optimize import linear_sum_assignment
    masks=masks.detach().float().cpu()
    reference=analytical_mask() if reference is None else reference.detach().float().cpu()
    batch=masks.reshape(-1,11,8)
    overlaps=torch.einsum("brs,rt->bst",batch,reference)
    intersection=torch.tensor([float(score[linear_sum_assignment(-score.numpy())].sum())
                               for score in overlaps])
    union=batch.sum((-2,-1))+reference.sum()-intersection
    return (intersection/union.clamp_min(1)).reshape(masks.shape[:-2])


def mean_structure(masks):
    """Compute each mask's structural diagnostics once, then average them."""
    metrics=[toeplitz_metrics(mask) for mask in masks]
    return {key:sum(row[key] for row in metrics)/len(metrics) for key in metrics[0]}


def align_masks(masks, reference):
    pairs=[align_to_reference(mask,reference) for mask in masks]
    return tuple(torch.stack(values) for values in zip(*pairs))


def align_to_reference(mask, reference):
    """Assign hidden columns to a fixed reference for visualization only.

    Input positions stay fixed. This does not change canonical structural metrics.
    """
    from scipy.optimize import linear_sum_assignment
    mask, reference = mask.detach().float().cpu(), reference.detach().float().cpu()
    cost = (mask[:, :, None]-reference[:, None, :]).square().sum(0).numpy()
    source, destination = linear_sum_assignment(cost)
    order = torch.empty(8, dtype=torch.long)
    order[torch.as_tensor(destination)] = torch.as_tensor(source)
    return mask[:, order], order
