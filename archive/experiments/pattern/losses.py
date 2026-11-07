import torch
from torch.nn import functional as F

def mask_vae_loss(output,target,beta=1e-3,hard_weight=.2):
    positive_weight=target.new_tensor(56/32)
    bce=F.binary_cross_entropy_with_logits(output["logits"],target,pos_weight=positive_weight)
    hard_mse=F.mse_loss(output["mask"],target)
    kl=.5*(output["mu"].square()+output["logvar"].exp()-1-output["logvar"]).sum(-1).mean()
    reconstruction=(1-hard_weight)*bce+hard_weight*hard_mse
    return reconstruction+beta*kl,{"bce":bce,"hard_mse":hard_mse,"kl":kl,"reconstruction":reconstruction}
