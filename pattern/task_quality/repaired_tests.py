"""Meaningful checks for surrogate geometry and permutation/hypergradients."""
import unittest
import torch
import torch.nn.functional as F
from .repaired_generator import fixed_mass_soft,repaired_topk_ste,make_model
from .generator import FEATURE_DIM,permute_hidden_columns
from .meta import _inner_adapt,_init_child_batch,_batched_logits

class RepairedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def test_soft_mass_and_implicit_gradient(self):
        torch.manual_seed(13)
        x=torch.randn(2,11,8,dtype=torch.float64,requires_grad=True)
        self.assertTrue(torch.allclose(fixed_mass_soft(x).sum((-2,-1)),torch.full((2,),32.,dtype=torch.float64),atol=1e-9))
        self.assertTrue(torch.autograd.gradcheck(fixed_mass_soft,(x,),eps=1e-5,atol=1e-5,rtol=1e-4))
    def test_hard_and_gradient_shift_invariance(self):
        torch.manual_seed(2)
        x=torch.randn(3,11,8,dtype=torch.float64,requires_grad=True)
        u=torch.randn_like(x)
        m=repaired_topk_ste(x)
        g=torch.autograd.grad((m*u).sum(),x)[0]
        shifted=(x.detach()+100).requires_grad_()
        ms=repaired_topk_ste(shifted)
        gs=torch.autograd.grad((ms*u).sum(),shifted)[0]
        self.assertTrue(torch.equal(m,ms));self.assertTrue(torch.equal(m.sum((-2,-1)),torch.full((3,),32.,dtype=torch.float64)))
        self.assertTrue(torch.allclose(g,gs,atol=1e-10,rtol=1e-9))
        self.assertTrue(torch.allclose(g.sum((-2,-1)),torch.zeros(3,dtype=torch.float64),atol=1e-10))
        self.assertGreater(float(g.norm()),0)
    def test_equal_scores_have_finite_gradient(self):
        x=torch.zeros(2,11,8,requires_grad=True)
        u=torch.arange(88.).reshape(1,11,8)
        (repaired_topk_ste(x)*u).sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(float(x.grad.norm()),0.)

    def test_functional_pool_preserves_map_association(self):
        from .functional_pool import FunctionalPool
        torch.manual_seed(32)
        bank={'q_abs':torch.rand(9,11,8)}
        m=FunctionalPool(bank,bounded_weights=True)
        feature=torch.randn(9,8,FEATURE_DIM)
        x=torch.randn(2,16,11);y=torch.tensor([[0.,1.]*8]*2)
        # Exercise nonuniform learned weights, not just the uniform initial case.
        torch.nn.init.normal_(m.scorer[-1].weight,std=.2)
        mask,scores,weights=m.forward_with_weights(feature,x,y)
        mp=torch.randperm(9)
        cp=torch.stack([torch.randperm(8) for _ in range(9)])
        pm,ps,pw=m.forward_with_weights((permute_hidden_columns(feature,cp)[mp],m.proposals[mp]),x,y)
        self.assertTrue(torch.equal(mask,pm))
        self.assertTrue(torch.allclose(scores,ps,atol=1e-6,rtol=1e-6))
        self.assertTrue(torch.allclose(weights[:,mp],pw,atol=1e-6,rtol=1e-6))
        p=_init_child_batch(2,2,torch.Generator().manual_seed(6),torch.device('cpu'))
        p,_=_inner_adapt(x,y,mask,p,steps=3,learning_rate=.1,momentum=.9,create_graph=True)
        loss=F.binary_cross_entropy_with_logits(_batched_logits(x,mask,p),y[:,None,:].expand(2,2,16))
        loss.backward()
        for name in ('token_mlp.0.weight','support_mlp.0.weight','scorer.2.weight'):
            g=dict(m.named_parameters())[name].grad
            self.assertIsNotNone(g);self.assertTrue(torch.isfinite(g).all());self.assertGreater(float(g.norm()),0.)

    def test_column_map_invariance_and_hypergradient(self):
        for variant in ('legacy_fixedmass','column_set_fixedmass'):
            torch.manual_seed(5);m=make_model(variant)
            bank=torch.randn(7,8,FEATURE_DIM)
            x=torch.randn(2,16,11);y=torch.tensor([[0.,1.]*8]*2)
            mask,l=m(bank,x,y)
            perm=permute_hidden_columns(bank,torch.stack([torch.randperm(8) for _ in range(7)]))[torch.randperm(7)]
            pm,pl=m(perm,x,y)
            self.assertTrue(torch.equal(mask,pm));self.assertTrue(torch.allclose(l,pl,atol=3e-6,rtol=3e-6))
            p=_init_child_batch(2,2,torch.Generator().manual_seed(6),torch.device('cpu'))
            p,_=_inner_adapt(x,y,mask,p,steps=3,learning_rate=.1,momentum=.9,create_graph=True)
            loss=F.binary_cross_entropy_with_logits(_batched_logits(x,mask,p),y[:,None,:].expand(2,2,16))
            loss.backward()
            for name in ('token_mlp.0.weight','support_mlp.0.weight'):
                g=dict(m.named_parameters())[name].grad
                self.assertIsNotNone(g);self.assertTrue(torch.isfinite(g).all());self.assertGreater(float(g.norm()),0)

if __name__=='__main__':unittest.main()
