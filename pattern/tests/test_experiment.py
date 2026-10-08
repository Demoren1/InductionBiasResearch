"""Scientific invariants and a small complete experiment."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import torch
from pattern.bank.imp import run_imp, initialization, fit_fixed_mask, prune
from pattern.bank.functional_maps import extract_maps
from pattern.bank.collect import collect
from pattern.datasets import input_table, labels_for, partitions, load_task
from pattern.io import ROOT, load_config, digest
from pattern.masks import exact_topk, canonical_columns, toeplitz_metrics
from pattern.models.nf import NFLayer
from pattern.models.task_model import Experiment
from pattern.losses import mask_vae_loss
from pattern.train import train
from pattern.evaluate import evaluate


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_exact_projection_and_gradient(self):
        x=torch.zeros(4,11,8,requires_grad=True)
        hard=exact_topk(x,straight_through=True)
        self.assertTrue(((hard==0)|(hard==1)).all())
        self.assertTrue((hard.sum((-2,-1))==32).all())
        self.assertTrue(hard[0].flatten()[:32].all())
        (hard*torch.arange(88).reshape(11,8)).sum().backward()
        self.assertGreater(float(x.grad.abs().sum()),0)

    def test_nf_equivariance_and_latent_invariance(self):
        torch.manual_seed(12)
        layer=NFLayer(3,5)
        x=torch.randn(3,11,8,3); permutation=torch.randperm(8)
        torch.testing.assert_close(layer(x[:,:,permutation]),layer(x)[:,:,permutation],rtol=1e-5,atol=1e-6)
        model=Experiment(["0001","0011"],{"nf_channels":4,"encoder_width":16,"latent_dim":4,"decoder_width":16})
        a=model.encoders["0001"](x);b=model.encoders["0001"](x[:,:,permutation])
        for left,right in zip(a,b):torch.testing.assert_close(left,right,rtol=1e-5,atol=1e-6)
        output=model("0001",x)
        objective,_=mask_vae_loss(output,exact_topk(torch.randn(3,11,8)))
        objective.backward()
        self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in model.decoder.parameters()))
        self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in model.encoders['0001'].parameters()))
        self.assertTrue(all(p.grad is None for p in model.encoders['0011'].parameters()))
        self.assertTrue(((output['mask']==0)|(output['mask']==1)).all())

    def test_imp_rewinds_and_prunes_monotonically(self):
        _,x=input_table();y=labels_for(x,'0001')
        state,mask,history,initial=run_imp(17,x[:32],y[:32],x[32:64],y[32:64],steps=2)
        counts=[entry['active_edges'] for entry in history]
        self.assertEqual(counts,[88,71,57,46,37,32])
        self.assertEqual(int(mask.sum()),32)
        self.assertTrue((state['w'][mask==0]==0).all())
        expected=fit_fixed_mask(initial,mask,x[:32],y[:32],2,.03,.001)
        for key in state:torch.testing.assert_close(state[key],expected[key],rtol=0,atol=0)
        zeros=torch.zeros(11,8)
        sparse=prune(torch.ones(11,8),zeros,32)
        self.assertTrue(sparse.flatten()[:32].all())
        self.assertFalse((prune(sparse,zeros,20)*(1-sparse)).any())

    def test_functional_derivatives_and_mask_pairing(self):
        _,x=input_table();probe=x[:8].clone().requires_grad_()
        state=initialization(2);mask=exact_topk(torch.rand(11,8))
        raw=extract_maps(state,mask,probe)
        psi=torch.relu(probe@(state['w']*mask)+state['b'])*state['a']
        derivative=torch.stack([torch.autograd.grad(psi[:,j].sum(),probe,retain_graph=True)[0]*probe for j in range(8)],-1)
        torch.testing.assert_close(raw['q_signed'],derivative.mean(0))
        torch.testing.assert_close(raw['q_abs'],derivative.abs().mean(0))
        torch.testing.assert_close(raw['q_rms'],derivative.square().mean(0).sqrt())
        permutation=torch.randperm(8)
        order=canonical_columns(raw['q_abs'])
        permuted_order=canonical_columns(raw['q_abs'][:,permutation])
        # Equal profiles may tie; they have exactly the same canonical functional input.
        torch.testing.assert_close(raw['q_abs'][:,order],raw['q_abs'][:,permutation][:,permuted_order])

    def test_toeplitz_reference_and_observation_boundaries(self):
        rows,cols=torch.meshgrid(torch.arange(11),torch.arange(8),indexing='ij')
        ideal=((rows-cols>=0)&(rows-cols<4)).float()
        self.assertEqual(int(ideal.sum()),32)
        self.assertTrue(toeplitz_metrics(ideal)['toeplitz_exact'])
        parts=partitions(4100)
        keys=list(parts)
        self.assertEqual(sum(len(value) for value in parts.values()),2048)
        for i,left in enumerate(keys):
            for right in keys[i+1:]:self.assertFalse(set(parts[left].tolist())&set(parts[right].tolist()))

    def test_end_to_end_and_bank_immutability(self):
        config=load_config(ROOT/'pattern/configs/smoke.json')
        config=copy.deepcopy(config);config['tasks']=['0001'];config['bank']['maps_per_task']=4
        config['training']['epochs']=1
        with tempfile.TemporaryDirectory(dir=ROOT/'data',prefix='test_imp32_') as temporary:
            parent=Path(temporary)
            bank=collect(config,parent=parent,bank_id='bank')
            before={str(p):digest(p) for p in bank.rglob('*') if p.is_file()}
            with self.assertRaises(FileExistsError):collect(config,parent=parent,bank_id='bank')
            datasets={split:load_task(bank,'0001',split) for split in ('train','validation','test')}
            sets=[set(value['seeds']) for value in datasets.values()]
            self.assertFalse(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
            self.assertEqual(datasets['train']['features'].shape[-1],3)
            self.assertTrue((datasets['train']['targets'].sum((-2,-1))==32).all())
            run=train(config,bank,'run',parent=parent/'runs')
            result=evaluate(run,child_steps=1,replicas=1,evaluation_id='eval')
            metrics=json.loads((result/'metrics.json').read_text())
            self.assertIn('fresh_quality',metrics['tasks']['0001'])
            self.assertEqual(before,{str(p):digest(p) for p in bank.rglob('*') if p.is_file()})
            card=bank/json.loads((bank/'manifest.json').read_text())['tasks']['0001'][0]['path']/'imp_mask.pt'
            original=card.read_bytes();card.write_bytes(b'changed')
            role=json.loads((bank/'manifest.json').read_text())['tasks']['0001'][0]['split']
            with self.assertRaisesRegex(ValueError,'artifact changed'):load_task(bank,'0001',role)
            card.write_bytes(original)

if __name__=='__main__':unittest.main()
