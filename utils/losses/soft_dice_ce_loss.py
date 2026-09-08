import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceCrossEntropyLoss(nn.Module):
    def __init__(self,num_classes:int,ce_weight=1.0,dice_weight=1.0,ignore_index=-1):
        super().__init__(); self.num_classes=num_classes; self.ce_weight=ce_weight
        self.dice_weight=dice_weight; self.ignore_index=ignore_index

    def forward(self,logits,target):
        ce=F.cross_entropy(logits,target,ignore_index=self.ignore_index)
        valid=target.ne(self.ignore_index)
        safe=target.masked_fill(~valid,0)
        one_hot=F.one_hot(safe,self.num_classes).permute(0,3,1,2).to(logits.dtype)
        probs=logits.softmax(dim=1); mask=valid[:,None]
        inter=(probs*one_hot*mask).sum((0,2,3)); denom=((probs+one_hot)*mask).sum((0,2,3))
        dice=((2*inter+1e-6)/(denom+1e-6)).mean()
        loss=self.ce_weight*ce+self.dice_weight*(1-dice)
        return loss,{'loss_seg':float(loss.detach()),'loss_ce':float(ce.detach()),'dice_soft':float(dice.detach())}
