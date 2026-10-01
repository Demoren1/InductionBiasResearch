"""Meaningful checks for fixed-horizon, query-independent child training."""
import unittest
import torch
from .graph_flow_child import fit_short


class ShortChildTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        gen=torch.Generator().manual_seed(901)
        self.x=torch.sign(torch.randn(32,11,generator=gen))
        self.y=torch.arange(32).remainder(2).float()
        self.masks=torch.zeros(2,11,8)
        self.masks.reshape(2,-1)[:,:32]=1
        self.masks[1]=self.masks[1].roll(2,0)

    def test_query_labels_cannot_change_fixed_weights(self):
        kwargs=dict(seeds=[201,202],cap=100,minimum=50,fixed_horizon=True,lr=.03,l2=.01)
        a=fit_short(self.masks,self.x,self.y,self.x,self.y,**kwargs)
        b=fit_short(self.masks,self.x,self.y,self.x,1-self.y,**kwargs)
        for key in a['best_params']:
            torch.testing.assert_close(a['best_params'][key],b['best_params'][key],rtol=0,atol=0)
        self.assertEqual(a['selected_on'],'fixed_terminal_step')
        self.assertTrue((a['best_steps']==100).all())
        self.assertFalse(torch.equal(a['best_query'],b['best_query']))

    def test_batched_matches_independent_fits(self):
        kwargs=dict(cap=100,minimum=50,fixed_horizon=True,lr=.03,l2=.01)
        batch=fit_short(self.masks,self.x,self.y,self.x,self.y,[201,202],**kwargs)
        for i,seed in enumerate((201,202)):
            solo=fit_short(self.masks[i:i+1],self.x,self.y,self.x,self.y,[seed],**kwargs)
            for key in batch['best_params']:
                torch.testing.assert_close(batch['best_params'][key][i:i+1],solo['best_params'][key],atol=2e-6,rtol=2e-5)


if __name__=='__main__':unittest.main()
