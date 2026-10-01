"""Fixed source-only dense/random/functional controls for dual-loss wiring."""
import hashlib
import json

import torch

from .permutation_utility_run import OUT, BANK, load_source_only, save_json
from .rebuilt_bank_run import costs
from .utility_graph_context import task_sets
from .utility_graph_models import exact_topk
from .utility_graph_child import fit_children


def main():
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu');seed=4100
    settings=json.loads((BANK/'new_child_pilot/selection.json').read_text())
    data=load_source_only(seed,device)
    sets=task_sets(data,costs()['source'],seed+920000,
                   train_name='source_train',query_name='source_validation')
    context_path=BANK/f'seed_{seed}/functional_context.pt'
    context=torch.load(context_path,map_location='cpu',weights_only=False,mmap=True)
    mean=context['mean_score'].to(device)
    rng=torch.Generator(device=device).manual_seed(seed+930000)
    random=torch.rand((1,784,32),generator=rng,device=device)
    masks=torch.cat((torch.ones((1,784,32),device=device),exact_topk(random,7526),exact_topk(mean[None],7526)))
    save_json(OUT/'source_controls_protocol.json',dict(scope='source-only descriptive fixed controls; no target selection',
        seed=seed,methods=['dense','random','functional'],replicas=[0,1],target_edges=7526,
        settings=settings,context_sha256=hashlib.sha256(context_path.read_bytes()).hexdigest(),
        source_split_hashes=data['split_hashes'],test_opened=False))
    fitted=fit_children(masks.repeat_interleave(2,0),torch.stack([s['x'] for s in sets]),
        torch.stack([s['y'] for s in sets]),torch.stack([s['qx'] for s in sets]),torch.stack([s['qy'] for s in sets]),
        [seed+940000+3001*i for i in range(4)],[0,1]*3,steps=settings['steps'],
        lr=settings['sparse']['lr'],l2=settings['sparse']['l2'],device=device,
        chunk_size=256,checkpoint_every=100,lr_decay_every=settings['lr_decay_every'])
    torch.save(dict(methods=['dense','random','functional'],masks=masks.cpu(),children=fitted),OUT/'source_controls.pt')
    save_json(OUT/'source_controls.json',dict(methods=['dense','random','functional'],
        mean_query_nmse=fitted['query_loss'].reshape(4,3,2).mean((0,2)).tolist(),
        per_task_query_nmse=fitted['query_loss'].reshape(4,3,2).mean(2).tolist(),
        plateau=int(fitted['plateau_flags'].sum()),fits=fitted['plateau_flags'].numel(),test_opened=False))
    print((OUT/'source_controls.json').read_text(),flush=True)


if __name__=='__main__':main()
