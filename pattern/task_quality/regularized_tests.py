"""Compare vectorized per-run Adam scaling against ordinary independent Adam."""
import tempfile,unittest
from pathlib import Path
import torch
import torch.nn.functional as F
from .repaired_eval import fit,balanced
from .eval_run import _candidate_init_seed
from .meta import _init_children,child_logits_batch

class RegularizedFitTests(unittest.TestCase):
    def test_independent_adam_matches_batched_fitter(self):
        torch.set_num_threads(1);torch.manual_seed(31)
        x=torch.randn(128,11);y=torch.tensor([0.,1.]*64)
        masks=torch.ones(2,11,8);masks[0].flatten()[32:]=0
        specs=[{'method':'test','task_id':'k4:0000','seed':8100,'init_id':i,'lr':lr,'l2':reg,'budget':128}
               for i,(lr,reg) in enumerate(((.003,.001),(.01,.01)))]
        manifest={'specs':specs,'masks':masks,'x':x.expand(2,-1,-1),'y':y.expand(2,-1),
                  'qx':x.expand(2,-1,-1),'qy':y.expand(2,-1)}
        with tempfile.TemporaryDirectory() as folder:
            fitted=fit(manifest,Path(folder),'unit','cpu',cap=100)
        largest=0.
        for i,spec in enumerate(specs):
            seed=_candidate_init_seed('k4:0000',128,i)
            p=_init_children([seed],torch.device('cpu'));opt=torch.optim.Adam(p.values(),lr=spec['lr'])
            for step in range(100):
                opt.zero_grad(set_to_none=True)
                bce=balanced(child_logits_batch(x,masks[i:i+1],p),y[None])
                reg=.5*spec['l2']*((p['w']*masks[i:i+1]).square().sum()+p['b'].square().sum()+p['a'].square().sum()+p['c'].square().sum())
                (bce.sum()+reg).backward();opt.step()
            for key in p:
                delta=float((p[key][0]-fitted['last_params'][key][i]).abs().max())
                largest=max(largest,delta)
                self.assertTrue(torch.allclose(p[key][0],fitted['last_params'][key][i],atol=2e-6,rtol=2e-6),(i,key,delta))
        print('Independent versus vectorized Adam maximum parameter difference:',largest)

if __name__=='__main__':unittest.main()
