"""Task observations and seed-disjoint functional-map datasets."""
import json
from pathlib import Path
import torch
from .io import digest
from .masks import canonical_columns
from .bank.functional_maps import NF_CHANNELS, NF_BANK_SCHEMA

def input_table():
    ids = torch.arange(2048)
    x = ((ids[:, None] >> torch.arange(10, -1, -1)) & 1).float()*2-1
    return ids, x

def labels_for(x, pattern):
    bits = torch.tensor([int(bit) for bit in pattern], dtype=x.dtype, device=x.device)
    return ((x+1).div(2).unfold(1,4,1) == bits).all(-1).any(-1).float()

def partitions(seed):
    order = torch.randperm(2048, generator=torch.Generator().manual_seed(seed+19731))
    return {"support": order[:768], "query": order[768:1024],
            "probe": order[1024:1152], "validation": order[1152:1536], "test": order[1536:]}

def balanced_rows(labels, pool, count, seed):
    if count < 2 or count % 2:
        raise ValueError("support/query budgets must be positive even counts")
    groups = [pool[labels[pool] == cls] for cls in (0,1)]
    if min(map(len, groups)) < count//2:
        raise ValueError("observation pool lacks a class for the requested balanced budget")
    generator = torch.Generator().manual_seed(seed)
    rows = torch.cat([g[torch.randperm(len(g), generator=generator)[:count//2]] for g in groups])
    return rows[torch.randperm(count, generator=generator)]

def load_task(bank, task, split):
    bank = Path(bank)
    manifest = json.loads((bank/"manifest.json").read_text())
    if manifest.get("schema") != NF_BANK_SCHEMA or manifest.get("input_representation",{}).get("channels") != list(NF_CHANNELS):
        raise ValueError("bank must contain weight/gradient/functional channels (v2); collect a new bank ID")
    if manifest["status"] != "complete" or manifest["k"] != 32:
        raise ValueError("bank must be complete and use IMP K=32")
    records = [r for r in manifest["tasks"][task] if r["split"] == split]
    if not records:
        raise ValueError(f"no maps for {task}/{split}")
    features, targets, names, orders, signed_scores = [], [], [], [], []
    for record in records:
        directory = bank/record["path"]
        for filename, expected in record["hashes"].items():
            if digest(directory/filename) != expected:
                raise ValueError(f"bank artifact changed: {directory/filename}")
        raw = torch.load(directory/"nf_features.pt", weights_only=True)
        mask = torch.load(directory/"imp_mask.pt", weights_only=True).float()
        if mask.shape != (11,8) or not ((mask==0)|(mask==1)).all() or int(mask.sum()) != 32:
            raise ValueError("each IMP target must contain exactly 32 binary connections")
        if raw.get("channels") != list(NF_CHANNELS) or any(raw[key].shape != (11,8) for key in NF_CHANNELS):
            raise ValueError("NF artifact must contain weight/gradient/functional tensors with shape [11,8]")
        order = canonical_columns(raw["functional_map"])
        feature = torch.stack([raw[key][:,order] for key in NF_CHANNELS], -1)
        feature = feature/feature.abs().amax(dim=(0,1)).clamp_min(1e-8)
        if not torch.isfinite(feature).all():
            raise ValueError("NF input channels must be finite")
        features.append(feature); targets.append(mask[:,order]); orders.append(order)
        signed_scores.append(raw["functional_map"][:,order]); names.append(record["network_seed"])
    return {"features": torch.stack(features), "targets": torch.stack(targets),
            "seeds": names, "column_orders": torch.stack(orders),
            "functional_scores": torch.stack(signed_scores)}
