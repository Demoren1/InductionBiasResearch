"""After ALL task shards exit, verify and atomically aggregate single-z masks."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config
from data.generate import ideal_mask
from evaluation.eval_generated_masks import _load_model
from evaluation.oracle_ideal import hard_topk, soft_topk
from evaluation.structural import best_permutation_iou


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.no_grad()
def run(root, profile):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for witness re-decoding")
    protocol_path = root / "single_z_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    source = protocol["profiles"][profile]
    assert protocol["parent_protocol_sha256"] == sha(root / "protocol.json")
    assert protocol["source_sha256"] == sha(ROOT / "evaluation/single_z_mlp.py")
    assert sha(source["checkpoint"]) == source["checkpoint_sha256"]
    assert sha(source["split"]) == source["split_sha256"]
    split = json.loads(Path(source["split"]).read_text())
    tasks = split["test_tasks"]
    search = torch.load(root / profile / "search/masks.pt", weights_only=True, map_location="cpu")
    assert search["metadata"]["protocol_profile"] == profile
    assert search["metadata"]["checkpoint1_sha256"] == source["checkpoint_sha256"]
    assert search["metadata"]["protocol_sha256"] == sha(root / "protocol.json")
    model, _ = _load_model(Path(source["checkpoint"]), torch.device("cuda"))
    model.requires_grad_(False)
    out = root / profile / "single_z"
    rows, report = [], {}
    for index, task in enumerate(tasks):
        row = torch.load(out / f"task_{task}.pt", weights_only=True, map_location="cpu")
        meta = row["metadata"]
        assert row["task"] == task and row["task_index"] == index
        assert meta["protocol_sha256"] == sha(protocol_path)
        assert meta["source_sha256"] == protocol["source_sha256"]
        assert meta["checkpoint_sha256"] == source["checkpoint_sha256"]
        assert meta["split_sha256"] == source["split_sha256"]
        assert meta["parent_masks_sha256"] == sha(root / profile / "search/masks.pt")
        assert meta["settings"] == protocol["settings"]
        assert row["decoder_unchanged"]
        assert len(row["history"]) == protocol["settings"]["outer_steps"]
        assert all(len(h["inner_train_loss_mean"]) == protocol["settings"]["inner_steps"] for h in row["history"])
        gap = config.parse_task(task).gap
        gap_index = search["gaps"].tolist().index(gap)
        prior = search["latents"]["prior"]["z1"][gap_index * 64:(gap_index + 1) * 64]
        assert torch.equal(prior, row["initial_z"])
        assert (row["final_z"].norm(dim=1) <= 8.00002).all()
        z = row["final_z"].to("cuda")
        c = model.condition([task], device="cuda").expand(len(z), -1)
        logits = model.decode(z, c)
        hard = hard_topk(logits, 96).cpu()
        assert torch.equal(hard, row["final_hard"])
        torch.testing.assert_close(soft_topk(logits, 96, .5).cpu(), row["final_soft"], rtol=1e-5, atol=1e-7)
        target = ideal_mask(task).float()
        iou = torch.tensor([best_permutation_iou(m.reshape(16, 16), target)["iou"] for m in hard])
        report[task] = {"mean_iou": float(iou.mean()), "max_iou": float(iou.max()),
                        "exact_ideal_count": int((iou == 1).sum()),
                        "mean_z_norm": float(z.norm(dim=1).mean()),
                        "mean_z_change": float((row["final_z"] - prior).norm(dim=1).mean()),
                        "changed_hard_count": int((hard != search["masks"]["prior1"][gap_index]).any(dim=1).sum()),
                        "iou": iou.tolist()}
        rows.append(row)
    assert all(row["metadata"] == rows[0]["metadata"] for row in rows)
    aggregate = {"tasks": tasks, "masks": torch.stack([r["final_hard"] for r in rows]),
                 "latents": torch.stack([r["final_z"] for r in rows]), "metadata": rows[0]["metadata"]}
    temporary = out / "masks.pt.audit.tmp"
    torch.save(aggregate, temporary)
    os.replace(temporary, out / "masks.pt")
    report = {"profile": profile, "validated_masks": len(tasks) * 64,
              "aggregation": "rebuilt from validated per-task artifacts after all workers exited; atomic replace",
              "masks_sha256": sha(out / "masks.pt"), "tasks": report}
    (out / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(profile, "validated", report["validated_masks"], "single-z masks", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "outputs/decoder_agreement/20260906")
    parser.add_argument("--profile", choices=("interp", "extrap"), required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    run(args.root, args.profile)
