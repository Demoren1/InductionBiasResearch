"""Source-only feasibility check; this is NOT learned structure transfer."""

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import MNIST, FashionMNIST


class DeepSets(nn.Module):
    def __init__(self):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU()
        )
        self.rho = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10))

    def forward(self, points):
        return self.rho(self.phi(points).mean(1))


class Dense(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(196 * 3, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, points):
        return self.layers(points)


def point_inputs(images, device):
    intensity = F.avg_pool2d(images[:, None].to(device).float() / 255, 2).flatten(1)
    u, v = torch.meshgrid(
        torch.linspace(-1, 1, 14, device=device),
        torch.linspace(-1, 1, 14, device=device),
        indexing="ij",
    )
    coords = torch.stack([u.flatten(), v.flatten()], -1)
    return torch.cat(
        [coords[None].expand(len(images), -1, -1), intensity[:, :, None] * 2 - 1], -1
    )


def permute_points(points, generator):
    permutation = torch.rand(
        points.shape[:2], device=points.device, generator=generator
    ).argsort(1)
    return points.gather(1, permutation[:, :, None].expand(-1, -1, 3))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["MNIST", "FashionMNIST"], required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device("cuda")
    root = Path("experiments/structure_search_v2")
    out = root / "runs" / ("pointset_pilot_" + args.dataset)
    out.mkdir(parents=True, exist_ok=False)
    data = {"MNIST": MNIST, "FashionMNIST": FashionMNIST}[args.dataset](
        root / "datasets", train=True, download=False
    )
    order = torch.randperm(len(data), generator=torch.Generator().manual_seed(281090))
    train_ids, val_ids = order[:6000], order[30000:32000]
    torch.save({"training_ids": train_ids, "validation_ids": val_ids}, out / "split.pt")
    x = point_inputs(data.data[train_ids], device)
    y = data.targets[train_ids].to(device)
    xv = point_inputs(data.data[val_ids], device)
    yv = data.targets[val_ids].to(device)
    started = time.time()
    results = []
    for method, cls in [("dense_adam", Dense), ("deepsets_given", DeepSets)]:
        for seed in (281091, 281092):
            torch.manual_seed(seed)
            model = cls().to(device)
            lr = 0.001 if method == "dense_adam" else 0.003
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            generator = torch.Generator(device=device).manual_seed(seed + 10)
            validation_generator = torch.Generator(device=device).manual_seed(281099)
            xval = permute_points(xv, validation_generator)
            best = math.inf
            history = []
            for step in range(2049):
                if step % 128 == 0:
                    model.eval()
                    with torch.no_grad():
                        logits = torch.cat([model(batch) for batch in xval.split(256)])
                        loss = float(F.cross_entropy(logits, yv))
                        accuracy = float((logits.argmax(-1) == yv).float().mean())
                    history.append(
                        {
                            "step": step,
                            "validation_ce": loss,
                            "validation_accuracy": accuracy,
                        }
                    )
                    if loss < best:
                        best = loss
                        best_accuracy = accuracy
                        torch.save(model.state_dict(), out / f"{method}_{seed}.pt")
                    print(
                        args.dataset,
                        method,
                        seed,
                        step,
                        round(loss, 4),
                        round(accuracy, 4),
                        flush=True,
                    )
                if step == 2048:
                    break
                model.train()
                ids = torch.randint(len(x), (128,), device=device, generator=generator)
                batch = permute_points(x[ids], generator)
                loss = F.cross_entropy(model(batch), y[ids])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            row = {
                "method": method,
                "seed": seed,
                "validation_ce": best,
                "validation_accuracy": best_accuracy,
                "parameters": sum(p.numel() for p in model.parameters()),
                "history": history,
            }
            results.append(row)
            (out / "results.json").write_text(json.dumps(results, indent=2))
    (out / "complete.json").write_text(
        json.dumps(
            {
                "seconds": time.time() - started,
                "official_test_opened": False,
                "status": "feasibility_only_not_structure_search",
                "dataset": args.dataset,
                "source_support": 6000,
                "source_validation": 2000,
                "steps": 2048,
                "point_count": 196,
                "features_per_point": ["row", "column", "intensity"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
