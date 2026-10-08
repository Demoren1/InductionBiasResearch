"""Train -> magnitude-prune -> rewind to the original initialization."""
import math
import torch
from torch.nn import functional as F

def initialization(seed, device="cpu"):
    generator = torch.Generator().manual_seed(seed)
    return {"w": (torch.randn(11,8,generator=generator)*.1).to(device),
            "b": torch.zeros(8,device=device),
            "a": (torch.randn(8,generator=generator)*.1).to(device),
            "c": torch.zeros((),device=device)}

def _logits(state, weight, x):
    if weight.ndim == 3:
        hidden = F.relu(torch.bmm(x,weight)+state["b"][:,None])
        return torch.bmm(hidden,state["a"][:,:,None]).squeeze(-1)+state["c"][:,None]
    return F.relu(x@weight+state["b"])@state["a"]+state["c"]

def logits(state, mask, x):
    return _logits(state,state["w"]*mask,x)

def prune(mask, weight, keep):
    active = torch.nonzero(mask.flatten(), as_tuple=False).flatten()
    if not 1 <= keep <= len(active):
        raise ValueError("pruning cannot add connections")
    values = weight.detach().abs().flatten()[active]
    winners = active[values.argsort(descending=True, stable=True)[:keep]]
    result = torch.zeros_like(mask).flatten()
    result[winners] = 1
    return result.reshape_as(mask)

def fit_fixed_mask(initial, mask, x, y, steps, lr, l2, *, loss_history=None, log_every=10, show_progress=False, description="Child fit"):
    from ..reporting import progress
    state = {key:value.detach().clone().requires_grad_() for key,value in initial.items()}
    optimizer = torch.optim.Adam(state.values(), lr=lr)
    def losses():
        weight = state["w"]*mask
        prediction = _logits(state,weight,x)
        loss = (F.binary_cross_entropy_with_logits(prediction,y,reduction="none").mean(-1)
                if weight.ndim == 3 else F.binary_cross_entropy_with_logits(prediction,y))
        # One reduction over all effective parameters avoids separate L2 kernels.
        shape = weight.shape[:-2]
        parameters = torch.cat((weight.reshape(*shape,-1), *(state[key].reshape(*shape,-1) for key in ("b","a","c"))),dim=-1)
        return loss, loss+.5*l2*parameters.square().sum(-1)
    for step in progress(range(1,steps+1),show_progress,desc=description,unit="step",leave=False):
        loss, objective = losses()
        if loss_history is not None and (step==1 or step==steps or (step-1)%log_every==0):
            loss_history.append({"step":step-1,"support_bce":float(loss.detach()),"objective":float(objective.detach())})
        if not torch.isfinite(objective).all():
            raise RuntimeError("nonfinite child objective")
        # Sum network objectives: every Adam update matches its independent fit.
        optimizer.zero_grad(set_to_none=True); objective.sum().backward(); optimizer.step()
        with torch.no_grad():
            state["w"].mul_(mask)
    if loss_history is not None:
        with torch.no_grad():
            loss, objective = losses()
            loss_history.append({"step":steps,"support_bce":float(loss),"objective":float(objective)})
    return {key:value.detach().clone() for key,value in state.items()}

def run_imp(seed, support_x, support_y, query_x, query_y, *, k=32,
            steps=1000, prune_fraction=.2, lr=.03, l2=.001, device="cpu"):
    if k != 32 or steps < 1 or not 0 < prune_fraction < 1:
        raise ValueError("IMP needs K=32, positive steps and 0<prune_fraction<1")
    initial = initialization(seed,device)
    mask = torch.ones(11,8,device=device)
    support_x, support_y, query_x, query_y = [v.to(device) for v in (support_x,support_y,query_x,query_y)]
    history, active = [], mask.numel()
    while True:
        # Fresh parameters and Adam state at every round, same original values.
        state = fit_fixed_mask(initial,mask,support_x,support_y,steps,lr,l2)
        history.append({"round":len(history),"active_edges":active,"steps":steps,
            "rewind":"initialization", "support_bce":float(F.binary_cross_entropy_with_logits(logits(state,mask,support_x),support_y)),
            "query_bce":float(F.binary_cross_entropy_with_logits(logits(state,mask,query_x),query_y))})
        if active == k:
            break
        active = max(k, min(active-1, math.ceil(active*(1-prune_fraction))))
        mask = prune(mask,state["w"],active)
    cpu = lambda mapping:{key:value.cpu().clone() for key,value in mapping.items()}
    return cpu(state), mask.cpu(), history, cpu(initial)

def run_imp_batch(seeds, support_x, support_y, query_x, query_y, *, k=32,
                  steps=1000, prune_fraction=.2, lr=.03, l2=.001, device="cpu", show_progress=False):
    """Independent IMP networks packed along the leading tensor dimension.

    Each network retains its seeded initialization, support set, pruning order
    and elementwise Adam moments. Only tensor operations are shared.
    """
    if not seeds or k != 32 or steps < 1 or not 0 < prune_fraction < 1:
        raise ValueError("batched IMP needs seeds, K=32, positive steps and 0<prune_fraction<1")
    originals = [initialization(seed) for seed in seeds]
    initial = {key:torch.stack([state[key] for state in originals]).to(device) for key in originals[0]}
    mask = torch.ones_like(initial["w"])
    support_x, support_y, query_x, query_y = [v.to(device) for v in (support_x,support_y,query_x,query_y)]
    if query_x.ndim == 2: query_x = query_x.expand(len(seeds),-1,-1)
    if query_y.ndim == 1: query_y = query_y.expand(len(seeds),-1)
    for x,y in ((support_x,support_y),(query_x,query_y)):
        if x.ndim != 3 or x.shape[0] != len(seeds) or x.shape[-1] != 11 or x.shape[:-1] != y.shape:
            raise ValueError("IMP batch observations must have shape [networks,rows,11] with matching labels")
    histories, active = [[] for _ in seeds], 88
    while True:
        state = fit_fixed_mask(initial,mask,support_x,support_y,steps,lr,l2,
            show_progress=show_progress,description=f"IMP {active} edges / {len(seeds)} networks")
        with torch.no_grad():
            scores = torch.stack([F.binary_cross_entropy_with_logits(logits(state,mask,x),y,reduction="none").mean(-1)
                                  for x,y in ((support_x,support_y),(query_x,query_y))],-1).cpu().tolist()
        for history,(support_bce,query_bce) in zip(histories,scores):
            history.append({"round":len(history),"active_edges":active,"steps":steps,
                            "rewind":"initialization","support_bce":support_bce,"query_bce":query_bce})
        if active == k: break
        active = max(k,min(active-1,math.ceil(active*(1-prune_fraction))))
        values = state["w"].abs().masked_fill(mask==0,-torch.inf).flatten(1)
        winners = values.argsort(dim=1,descending=True,stable=True)[:,:active]
        mask = torch.zeros_like(values).scatter_(1,winners,1).reshape_as(mask)
    cpu = lambda mapping:{key:value.cpu().clone() for key,value in mapping.items()}
    return cpu(state),mask.cpu(),histories,cpu(initial)
