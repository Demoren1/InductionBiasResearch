"""Small checks that adding follow-up methods preserves paired target fits."""
from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from .core import MaskedDeepSets, Split, evaluate_masks


def main() -> None:
    torch.set_num_threads(2)
    gen = torch.Generator().manual_seed(735)
    masks = (torch.rand(20, 784, 32, generator=gen) < .2).float()
    for count in (20, 28, 32, 40):
        reference = MaskedDeepSets(masks, seed=19)
        expanded = MaskedDeepSets(torch.cat([masks, masks[:1].expand(count-20, -1, -1)]),
                                 seed=19, initialization_reference_models=20)
        for name, value in reference.named_parameters():
            assert torch.equal(value[:4], dict(expanded.named_parameters())[name][:4]), name
    data = {}
    for name in ('target_train', 'target_validation', 'target_test'):
        data[name] = Split(torch.rand(60, 784, generator=gen),
                           torch.arange(60) % 10, torch.arange(60))
    controls = {f'control{i}': masks[i*4:(i+1)*4] for i in range(5)}
    expanded = {**controls, 'new1': masks[:4], 'new2': masks[4:8]}
    costs = torch.arange(10).float()[None]
    with tempfile.TemporaryDirectory() as temporary:
        kwargs = dict(data=data, costs=costs, seed=483, device='cpu',
                      support_sizes=(4, 8), steps=3, batch_size=4, set_size=5,
                      validation_sets=4, test_sets=4)
        old = evaluate_masks(masks=controls, artifact_dir=Path(temporary)/'old', **kwargs)
        new = evaluate_masks(masks=expanded, artifact_dir=Path(temporary)/'new',
                             initialization_reference_models=20, **kwargs)
        key = lambda r: (r['task'], r['support_size'], r['method'], r['init'])
        old_records = {key(r): r for r in old}
        maximum = max(abs(r['mse']-old_records[key(r)]['mse']) for r in new
                      if r['method'] in controls)
        assert maximum < 1e-6, maximum
        for budget in (4, 8):
            a = torch.load(Path(temporary)/'old'/f'target_task0_budget{budget}.pt', weights_only=False)
            b = torch.load(Path(temporary)/'new'/f'target_task0_budget{budget}.pt', weights_only=False)
            for name, value in a['state_dict'].items():
                assert torch.allclose(value, b['state_dict'][name][:20], atol=1e-6, rtol=0), name
        print(f'Initialization and fit checks passed; maximum control MSE delta={maximum:.9g}')


if __name__ == '__main__':
    main()
