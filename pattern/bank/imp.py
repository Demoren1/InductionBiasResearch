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


def mean_bce(prediction, target, class_balanced=False):
    loss=F.binary_cross_entropy_with_logits(prediction,target,reduction="none")
    if class_balanced:
        prevalence=target.mean(-1,keepdim=True).clamp(1e-6,1-1e-6)
        loss=loss*(target*.5/prevalence+(1-target)*.5/(1-prevalence))
    return loss.mean(-1)

def l2_penalty(state, mask, strength):
    """Reduce all effective parameters once, preserving network batch dimensions."""
    if not strength: return 0.
    parameters = torch.cat(((state["w"]*mask).flatten(-2), state["b"],
                            state["a"], state["c"][...,None]), -1)
    return .5*strength*parameters.square().sum(-1)


class QueryCheckpoint:
    """Keep each independent network's best parameters and query step on device."""
    def __init__(self, state, shape):
        self.loss = next(iter(state.values())).new_full(shape, float("inf"))
        self.steps = torch.zeros_like(self.loss, dtype=torch.long)
        self.state = {key:value.detach().clone() for key,value in state.items()}

    @torch.no_grad()
    def update(self, state, score, step):
        improved = score < self.loss
        self.loss = torch.minimum(self.loss, score)
        self.steps = torch.where(improved, step, self.steps)
        for key,value in state.items():
            condition = improved.reshape(improved.shape + (1,)*(value.ndim-improved.ndim))
            self.state[key] = torch.where(condition, value.detach(), self.state[key])


def prune(mask, weight, keep):
    active = torch.nonzero(mask.flatten(), as_tuple=False).flatten()
    if not 1 <= keep <= len(active):
        raise ValueError("pruning cannot add connections")
    values = weight.detach().abs().flatten()[active]
    winners = active[values.argsort(descending=True, stable=True)[:keep]]
    result = torch.zeros_like(mask).flatten()
    result[winners] = 1
    return result.reshape_as(mask)

def fit_fixed_mask(initial, mask, x, y, steps, lr, l2, *, loss_history=None, log_every=10,
                   show_progress=False, description="Child fit", class_balanced=False,
                   query=None, select_every=0, selection=None):
    from ..reporting import progress
    state = {key:value.detach().clone().requires_grad_() for key,value in initial.items()}
    optimizer = torch.optim.Adam(state.values(), lr=lr)
    if select_every<0 or (select_every and query is None):raise ValueError("checkpoint selection needs query data")
    checkpoint = QueryCheckpoint(state, y.shape[:-1]) if select_every else None
    def losses():
        loss = mean_bce(logits(state,mask,x),y,class_balanced)
        return loss, loss+l2_penalty(state,mask,l2)
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
            if select_every and (step%select_every==0 or step==steps):
                score=mean_bce(logits(state,mask,query[0]),query[1],class_balanced)
                checkpoint.update(state, score, step)
    if loss_history is not None:
        with torch.no_grad():
            loss, objective = losses()
            loss_history.append({"step":steps,"support_bce":float(loss),"objective":float(objective)})
    if select_every:
        state=checkpoint.state
        if selection is not None:selection.update(selected_steps=checkpoint.steps.cpu().tolist())
    return {key:value.detach().clone() for key,value in state.items()}

def run_imp(seed, support_x, support_y, query_x, query_y, *, k=32,
            steps=1000, prune_fraction=.2, lr=.03, l2=.001, device="cpu"):
    """Use the same IMP implementation for a single seeded network."""
    state,mask,histories,initial=run_imp_batch([seed],support_x.unsqueeze(0),support_y.unsqueeze(0),
        query_x,query_y,k=k,steps=steps,prune_fraction=prune_fraction,lr=lr,l2=l2,device=device)
    unbatch=lambda mapping:{key:value[0].clone() for key,value in mapping.items()}
    return unbatch(state),mask[0].clone(),histories[0],unbatch(initial)


def run_imp_batch(seeds, support_x, support_y, query_x, query_y, *, k=32,
                  steps=1000, prune_fraction=.2, lr=.03, l2=.001, device="cpu", show_progress=False,
                  class_balanced=False, select_every=0):
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
        selection={}
        state = fit_fixed_mask(initial,mask,support_x,support_y,steps,lr,l2,
            show_progress=show_progress,description=f"IMP {active} edges / {len(seeds)} networks",
            class_balanced=class_balanced,query=(query_x,query_y),select_every=select_every,selection=selection)
        with torch.no_grad():
            scores = torch.stack([mean_bce(logits(state,mask,x),y,class_balanced)
                                  for x,y in ((support_x,support_y),(query_x,query_y))],-1).cpu().tolist()
        for history,(support_bce,query_bce) in zip(histories,scores):
            history.append({"round":len(history),"active_edges":active,"steps":steps,
                            "rewind":"initialization","support_bce":support_bce,"query_bce":query_bce})
        if select_every:
            for history,selected in zip(histories,selection["selected_steps"]):history[-1]["selected_step"]=selected
        if active == k: break
        active = max(k,min(active-1,math.ceil(active*(1-prune_fraction))))
        values = state["w"].abs().masked_fill(mask==0,-torch.inf).flatten(1)
        winners = values.argsort(dim=1,descending=True,stable=True)[:,:active]
        mask = torch.zeros_like(values).scatter_(1,winners,1).reshape_as(mask)
    cpu = lambda mapping:{key:value.cpu().clone() for key,value in mapping.items()}
    return cpu(state),mask.cpu(),histories,cpu(initial)
