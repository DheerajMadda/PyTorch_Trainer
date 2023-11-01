import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Compute Focal loss given target labels
    and model predictions.
    """
    def __init__(self, alpha=0.25, gamma=2, reduction="mean", ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, preds, targets):
    
        ce_loss = F.cross_entropy(
            preds,
            targets,
            ignore_index=self.ignore_index,
            reduction='none'
        )

        pt = torch.exp(-ce_loss)
        F_loss = self.alpha * (1 - pt)**self.gamma * ce_loss

        if self.reduction == "mean":
            return torch.mean(F_loss)
        else:
            return F_loss