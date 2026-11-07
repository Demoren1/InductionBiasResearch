import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from pattern.codebook.bank import predictions, scores
from pattern.datasets import input_table, labels_for


def main():
    plan = json.loads(Path("pattern/autoregressive/protocol.json").read_text())
    source = Path(plan["source_run"])
    output = Path(plan["run"])
    source_plan = json.loads(Path(plan["source_protocol"]).read_text())
    bank = np.load(source / "bank.npz")
    indices = np.load(source / "extracted.npz")["targets"]
    device = torch.device("cuda")
    torch.set_num_threads(4)
    torch.manual_seed(plan["seed"] + 40)
    _, inputs = input_table()
    inputs = inputs.to(device)
    all_labels = torch.stack(
        [labels_for(inputs, pattern) for pattern in source_plan["source_patterns"]]
    )
    split = {
        key: torch.tensor(value, device=device)
        for key, value in np.load(source / "splits.npz").items()
    }
    restarts, steps, batch_size = 4, 2048, 384
    arrays = {
        key: [] for key in ("query_loss", "query_accuracy", "population_accuracy")
    }
    histories = []
    for start in range(0, len(indices), batch_size):
        end = min(start + batch_size, len(indices))
        assignments = torch.tensor(indices[start:end], device=device)
        assignments = assignments.repeat_interleave(restarts, 0)
        task_ids = torch.tensor(bank["task_ids"][start:end], device=device)
        target = all_labels[task_ids].repeat_interleave(restarts, 0)
        count = len(target)
        numeric = torch.zeros(count, 7, device=device)
        numeric[:, :4] = F.normalize(torch.randn(count, 4, device=device), dim=1) * 0.5
        numeric[:, 5] = 1
        numeric.requires_grad_()
        optimizer = torch.optim.Adam([numeric], lr=0.1)

        def current(numeric=numeric, assignments=assignments):
            codebook = F.pad(numeric[:, :4], (1, 0))
            weights = codebook.gather(1, assignments.flatten(1)).reshape(-1, 11, 8)
            return [
                weights,
                numeric[:, 4:5].expand(-1, 8),
                numeric[:, 5:6].expand(-1, 8),
                numeric[:, 6],
            ]

        best_loss = torch.full((count,), torch.inf, device=device)
        best = [parameter.detach().clone() for parameter in current()]
        history = []
        for step in range(steps + 1):
            if step % 64 == 0:
                with torch.no_grad():
                    parameters = current()
                    query = scores(
                        predictions(parameters, inputs[split["query"]]),
                        target[:, split["query"]],
                        balanced=True,
                    )
                    better = query["loss"] < best_loss
                    best_loss[better] = query["loss"][better]
                    for stored, parameter in zip(best, parameters):
                        stored[better] = parameter[better]
            training = scores(
                predictions(current(), inputs[split["support"]]),
                target[:, split["support"]],
                balanced=True,
            )
            if step % 256 == 0:
                history.append(
                    {
                        "step": step,
                        "train_bce": float(training["loss"].detach().mean()),
                        "query_bce": float(query["loss"].mean()),
                        "train_recall": float(training["recall_positive"].mean()),
                        "query_recall": float(query["recall_positive"].mean()),
                    }
                )
                print("SOURCE AUDIT", start, step, history[-1], flush=True)
            if step == steps:
                break
            optimizer.zero_grad(set_to_none=True)
            training["loss"].sum().backward()
            optimizer.step()
        with torch.no_grad():
            chosen = best_loss.reshape(-1, restarts).argmin(1)
            ids = torch.arange(end - start, device=device) * restarts + chosen
            for name, selection in [
                ("query", split["query"]),
                ("population", torch.arange(2048, device=device)),
            ]:
                values = scores(
                    predictions(
                        [parameter[ids] for parameter in best], inputs[selection]
                    ),
                    target[ids][:, selection],
                    balanced=True,
                )
                for key, value in values.items():
                    if name + "_" + key in arrays:
                        arrays[name + "_" + key].append(value.cpu().numpy())
        histories.append(history)
    arrays = {key: np.concatenate(value) for key, value in arrays.items()}
    np.savez_compressed(output / "source_codebook_audit.npz", **arrays)
    (output / "source_codebook_audit_history.json").write_text(json.dumps(histories))
    result = {
        "networks": len(indices),
        "starts_per_indexator": restarts,
        "steps": steps,
        "selection": "balanced source-query BCE",
        "mean_original_query_accuracy": float(bank["query_accuracy"].mean()),
        "mean_codebook_query_accuracy": float(arrays["query_accuracy"].mean()),
        "mean_codebook_query_bce": float(arrays["query_loss"].mean()),
        "perfect_population_indexators": int(
            (arrays["population_accuracy"] == 1).sum()
        ),
        "codebook_query_bce_quantiles": np.quantile(
            arrays["query_loss"], [0, 0.1, 0.25, 0.5, 0.75, 1]
        ).tolist(),
        "caution": "Finite-start codebook fitting is not an analytic optimum over C.",
    }
    (output / "source_codebook_audit.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(result, flush=True)


if __name__ == "__main__":
    main()
