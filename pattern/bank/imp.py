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

def logits(state, mask, x):
    return F.relu(x@(state["w"]*mask)+state["b"])@state["a"]+state["c"]

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
    for step in progress(range(1,steps+1),show_progress,desc=description,unit="step",leave=False):
        loss = F.binary_cross_entropy_with_logits(logits(state,mask,x),y)
        penalty = (state["w"]*mask).square().sum() + sum(state[key].square().sum() for key in ("b","a","c"))
        objective = loss+.5*l2*penalty
        if loss_history is not None and (step==1 or step==steps or (step-1)%log_every==0):
            loss_history.append({"step":step-1,"support_bce":float(loss.detach()),"objective":float(objective.detach())})
        if not torch.isfinite(objective):
            raise RuntimeError("nonfinite child objective")
        optimizer.zero_grad(set_to_none=True); objective.backward(); optimizer.step()
        with torch.no_grad():
            state["w"].mul_(mask)
    if loss_history is not None:
        with torch.no_grad():
            loss=F.binary_cross_entropy_with_logits(logits(state,mask,x),y)
            penalty=(state["w"]*mask).square().sum()+sum(state[key].square().sum() for key in ("b","a","c"))
            loss_history.append({"step":steps,"support_bce":float(loss),"objective":float(loss+.5*l2*penalty)})
    return {key:value.detach().clone() for key,value in state.items()}

def run_imp(seed, support_x, support_y, query_x, query_y, *, k=32,
            steps=1000, prune_fraction=.2, lr=.03, l2=.001, device="cpu"):
    if k != 32 or steps < 1 or not 0 < prune_fraction < 1:
        raise ValueError("IMP needs K=32, positive steps and 0<prune_fraction<1")
    initial = initialization(seed,device)
    mask = torch.ones(11,8,device=device)
    support_x, support_y, query_x, query_y = [v.to(device) for v in (support_x,support_y,query_x,query_y)]
    history = []
    while True:
        active = int(mask.sum())
        # Fresh parameters and Adam state at every round, same original values.
        state = fit_fixed_mask(initial,mask,support_x,support_y,steps,lr,l2)
        history.append({"round":len(history),"active_edges":active,"steps":steps,
            "rewind":"initialization", "support_bce":float(F.binary_cross_entropy_with_logits(logits(state,mask,support_x),support_y)),
            "query_bce":float(F.binary_cross_entropy_with_logits(logits(state,mask,query_x),query_y))})
        if active == k:
            break
        keep = max(k, min(active-1, math.ceil(active*(1-prune_fraction))))
        mask = prune(mask,state["w"],keep)
    cpu = lambda mapping:{key:value.cpu().clone() for key,value in mapping.items()}
    return cpu(state), mask.cpu(), history, cpu(initial)
