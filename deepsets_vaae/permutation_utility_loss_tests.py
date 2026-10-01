"""Exact small-law checks for discrete utility gradients and consistency."""
import itertools
import unittest

import torch

from .permutation_utility_loss import sample_ordered_topk, quality_policy_loss, dual_objective


class UtilityLossTests(unittest.TestCase):
    def test_ordered_probabilities_normalize_and_match_enumeration(self):
        logits=torch.tensor([.2,-.6,.7],dtype=torch.float64,requires_grad=True)
        total=0.
        for order in itertools.permutations(range(3),2):
            noise=torch.full((1,1,3),-100.,dtype=torch.float64)
            for rank,index in enumerate(order):noise[0,0,index]=100-10*rank
            mask,lp,chosen=sample_ordered_topk(logits.reshape(1,1,3),2,gumbel=noise)
            self.assertEqual(chosen.reshape(-1).tolist(),list(order));self.assertEqual(float(mask.sum()),2)
            ref=logits[order[0]]-logits.logsumexp(0)
            ref+=logits[order[1]]-logits[[i for i in range(3) if i!=order[0]]].logsumexp(0)
            torch.testing.assert_close(lp[0],ref);total=total+lp.exp().sum()
        torch.testing.assert_close(total,torch.ones_like(total))
        torch.testing.assert_close(torch.autograd.grad(total,logits)[0],torch.zeros_like(logits),atol=1e-12,rtol=0)

    def test_leave_one_out_estimator_equals_exact_expected_risk_gradient(self):
        # K=1 makes exact enumeration of all independent paired draws small.
        logits=torch.tensor([.1,-.2,.4],dtype=torch.float64,requires_grad=True)
        p=logits.softmax(0);q=torch.tensor([.8,.2,.6],dtype=torch.float64)
        expected=(p*q).sum();gradient=torch.autograd.grad(expected,logits,retain_graph=True)[0]
        estimator=torch.zeros_like(logits)
        for i,j in itertools.product(range(3),repeat=2):
            lp=logits.log_softmax(0)[torch.tensor([i,j])][None]
            loss,_=quality_policy_loss(lp,q[torch.tensor([i,j])][None])
            estimator=estimator+(p[i]*p[j]).detach()*torch.autograd.grad(loss,logits,retain_graph=True)[0]
        torch.testing.assert_close(estimator,gradient,atol=1e-12,rtol=1e-12)

    def test_quality_does_not_backpropagate_through_child_metric(self):
        lp=torch.tensor([[-2.,-3.]],requires_grad=True);q=torch.tensor([[.1,.9]],requires_grad=True)
        loss,adv=quality_policy_loss(lp,q);loss.backward()
        self.assertIsNone(q.grad);self.assertLess(float(lp.grad[0,0]),0);self.assertGreater(float(lp.grad[0,1]),0)

    def test_consistency_has_gradients_and_zero_when_matched(self):
        a=torch.randn(2,5,requires_grad=True);b=torch.randn(2,5,requires_grad=True)
        z=torch.randn(2,3,4,requires_grad=True);zz=torch.randn(2,3,4,requires_grad=True)
        lp=torch.ones(2,2,requires_grad=True);q=torch.ones(2,2)
        out=dual_objective(lp,q,a,b,z,zz);out['loss'].backward()
        self.assertGreater(float(a.grad.norm()),0);self.assertGreater(float(z.grad.norm()),0)
        same=dual_objective(lp,q,a,a,z,z);self.assertEqual(float(same['consistency']),0.)


if __name__=='__main__':unittest.main()
