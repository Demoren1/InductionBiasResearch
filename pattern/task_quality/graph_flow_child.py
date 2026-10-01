"""Short, paired full-batch child fits; masks are evaluated without STE.

Only support labels supply gradients. Query labels select checkpoints and
monitor a finite, explicitly capped solver, never a mathematical optimum.
"""
import time
import torch
from .meta import _init_children, child_logits_batch
from .repaired_eval import balanced
from .convergence import loss_plateau


def fit_short(masks, x, y, qx, qy, seeds, *, device='cpu', lr=.03,
              l2=.001, cap=1400, minimum=500, tolerance=.01, fixed_horizon=False):
    started = time.monotonic()
    device = torch.device(device)
    masks = masks.detach().to(device).float()
    count = len(masks)
    def expand(z, rank):
        z = z.to(device).float()
        return z.unsqueeze(0).expand(count, *z.shape) if z.ndim == rank else z
    x, y, qx, qy = expand(x, 2), expand(y, 1), expand(qx, 2), expand(qy, 1)
    assert x.shape[0] == y.shape[0] == qx.shape[0] == qy.shape[0] == count
    params = _init_children(seeds, device)
    opt = torch.optim.Adam(params.values(), lr=lr)
    active = torch.ones(count, dtype=torch.bool, device=device)
    passes = torch.zeros(count, dtype=torch.long, device=device)
    stopping = torch.full((count,), cap, dtype=torch.long, device=device)
    best = torch.full((count,), float('inf'), device=device)
    best_step = torch.zeros(count, dtype=torch.long, device=device)
    best_params = {k: v.detach().clone() for k, v in params.items()}
    history = []
    def penalty():
        return .5*l2*((params['w']*masks).square().sum((1,2))
                     + params['b'].square().sum(1) + params['a'].square().sum(1)
                     + params['c'].square())
    for step in range(1, cap+1):
        opt.param_groups[0]['lr'] = lr * max(1/64, .5**((step-1)//200))
        opt.zero_grad(set_to_none=True)
        objective = balanced(child_logits_batch(x, masks, params), y) + penalty()
        (objective*active).sum().backward()
        if not all(torch.isfinite(v.grad).all() for v in params.values()):
            raise RuntimeError('Nonfinite child gradient')
        with torch.no_grad():
            gradient_norm = sum(v.grad.square().flatten(1).sum(1)
                                if v.ndim > 1 else v.grad.square()
                                for v in params.values()).sqrt()
            old = {k: v.detach().clone() for k, v in params.items()}
        opt.step()
        # Adam momentum must not continue moving already stopped children.
        with torch.no_grad():
            for k, v in params.items():
                v[~active] = old[k][~active]
        if step % 25 and step != cap:
            continue
        with torch.no_grad():
            support = balanced(child_logits_batch(x, masks, params), y)
            objective = support + penalty()
            query = balanced(child_logits_batch(qx, masks, params), qy)
            better = (query < best) & active
            best[better], best_step[better] = query[better], step
            for k in params:
                best_params[k][better] = params[k][better]
            history.append({'step': step, 'support_bce': support.cpu(),
                            'objective': objective.cpu(), 'query_bce': query.cpu(),
                            'gradient_norm': gradient_norm.cpu(),
                            'active_before_check': active.cpu().clone()})
            if step >= minimum:
                flat = torch.ones_like(active)
                for key in ('support_bce', 'objective', 'query_bce'):
                    flat &= loss_plateau(torch.stack([r[key] for r in history]),
                                         width=4, tolerance=tolerance).to(device)
                passes = torch.where(flat, passes+1, torch.zeros_like(passes))
                done = active & (passes >= 3)
                if not fixed_horizon:
                    stopping[done], active[done] = step, False
            if not active.any():
                break
    trajectory_best_params = {k:v.detach().clone().cpu() for k,v in best_params.items()}
    trajectory_best_query = best.detach().cpu().clone()
    trajectory_best_steps = best_step.detach().cpu().clone()
    if fixed_horizon:
        # Fixed steps and terminal parameters: query cannot choose checkpoints
        # or stopping on new tasks; all candidates receive the same budget.
        best_params = {k:v.detach().clone() for k,v in params.items()}
        best = query.clone()
        best_step.fill_(step)
        active = passes < 3
    return {'masks': masks.cpu(), 'best_params': {k:v.cpu() for k,v in best_params.items()},
            'last_params': {k:v.detach().cpu() for k,v in params.items()},
            'best_query': best.cpu(), 'best_steps': best_step.cpu(),
            'converged': (~active).cpu(), 'stopping_steps': stopping.cpu(),
            'history': history, 'steps': step, 'lr': lr, 'l2': l2, 'cap': cap,
            'trajectory_best_params':trajectory_best_params,
            'trajectory_best_query':trajectory_best_query,
            'trajectory_best_steps':trajectory_best_steps,
            'minimum': minimum, 'tolerance': tolerance, 'width':4, 'plateau_passes':3,
            'elapsed_seconds': time.monotonic()-started,
            'selected_on': 'fixed_terminal_step' if fixed_horizon else 'query',
            'fixed_horizon':fixed_horizon,'gradient_data': 'support_only'}
