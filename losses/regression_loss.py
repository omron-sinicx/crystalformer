import torch
import torch.nn.functional as F
import torch.nn as nn

def regression_loss(pred, data, targets, scale, bias, loss_fn:str):
    loss_fn = loss_fn.lower()
    assert(len(targets)==pred.shape[1])
    assert loss_fn in ("l1", "mse", "smooth_l1")

    if loss_fn == "l1":
        loss_fn = F.l1_loss
    elif loss_fn == "mse":
        loss_fn = F.mse_loss
    elif loss_fn == "smooth_l1":
        loss_fn = F.smooth_l1_loss
    
    loss = 0
    for i, t in enumerate(targets):
        labels = (data[t]-bias[i])/scale[i]
        loss += loss_fn(pred[:, i], labels, reduction='none')
    return loss
