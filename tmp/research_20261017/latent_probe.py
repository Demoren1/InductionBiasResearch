"""Source-only probes: does each frozen decoder need its encoder's code?"""

import torch
from torch.nn import functional as F

from multimask.common import matches, save_json, tasks, topk
from tmp.research_20261017.experiment import (
    HOME,
    KINDS,
    load_model,
    prepare,
)


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    if (HOME / "latent_probe.json").exists():
        print("Reusing completed latent interventions", flush=True)
        return
    config, cards, _ = prepare("cpu")
    names = tasks(config)
    rows = []
    permutation_key = "encoder_column_permutation_error"
    for kind in KINDS:
        for seed in (4213, 4214, 4215):
            model = load_model(kind, seed, config, "cpu")
            codes, features, targets = [], [], []
            for index, name in enumerate(names):
                selected = cards[name]["split"] == 1
                x = cards[name]["features"][selected]
                features.append(x)
                targets.append(cards[name]["masks"][selected])
                codes.append(model(index, x)[1])
            for index, name in enumerate(names):
                mu = codes[index]
                permutation = torch.arange(7, -1, -1)
                permuted = model(index, features[index][:, :, permutation])[1]
                invariant_error = float((mu - permuted).abs().max())
                for mode, code in (
                    ("posterior", mu),
                    ("wrong_family_posterior", codes[(index + 6) % 12]),
                    ("zero", torch.zeros_like(mu)),
                ):
                    scores = model.decode(code, index, name[0])
                    loss = F.binary_cross_entropy_with_logits(
                        scores,
                        targets[index],
                        pos_weight=scores.new_tensor(16),
                    )
                    rows.append(
                        {
                            "kind": kind,
                            "seed": seed,
                            "task": name,
                            "mode": mode,
                            "iou": float(
                                matches(topk(scores), name)[0].mean()
                            ),
                            "teacher_weighted_bce": float(loss),
                            permutation_key: invariant_error,
                        }
                    )
            print(f"Latent probes complete: {kind}/{seed}", flush=True)
    save_json(HOME / "latent_probe.json", rows)


if __name__ == "__main__":
    main()
