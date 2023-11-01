import torch
import torch.nn as nn
import torchmetrics as tmt

class MulticlassAccuracy(nn.Module):
    """
    Computes the multiclass accuracy given target labels
    and model predictions.
    """
    def __init__(self, num_classes, device="cpu"):
        super(MulticlassAccuracy, self).__init__()
        self.accuracy = tmt.classification.MulticlassAccuracy(
            num_classes=num_classes, 
            average='micro',
            validate_args=False
        ).to(device)

    def forward(self, preds, targets):
        preds = torch.argmax(preds, dim=1)
        accuracy = self.accuracy(preds, targets)
        return accuracy

class MulticlassPrecision(nn.Module):
    """
    Computes the multiclass precision given target labels
    and model predictions.
    """
    def __init__(self, num_classes, device="cpu"):
        super(MulticlassPrecision, self).__init__()
        self.precision = tmt.classification.MulticlassPrecision(
            num_classes=num_classes, 
            average='micro',
            validate_args=False
        ).to(device)

    def forward(self, preds, targets):
        preds = torch.argmax(preds, dim=1)
        precision = self.precision(preds, targets)
        return precision

class MulticlassRecall(nn.Module):
    """
    Computes the multiclass recall given target labels
    and model predictions.
    """
    def __init__(self, num_classes, device="cpu"):
        super(MulticlassRecall, self).__init__()
        self.recall = tmt.classification.MulticlassRecall(
            num_classes=num_classes, 
            average='micro',
            validate_args=False
        ).to(device)

    def forward(self, preds, targets):
        preds = torch.argmax(preds, dim=1)
        recall = self.recall(preds, targets)
        return recall

class MulticlassF1Score(nn.Module):
    """
    Computes the multiclass f1_score given target labels
    and model predictions.
    """
    def __init__(self, num_classes, device="cpu"):
        super(MulticlassF1Score, self).__init__()
        self.f1_score = tmt.classification.MulticlassF1Score(
            num_classes=num_classes, 
            average='micro',
            validate_args=False
        ).to(device)

    def forward(self, preds, targets):
        preds = torch.argmax(preds, dim=1)
        f1_score = self.f1_score(preds, targets)
        return f1_score

class DiceScore(nn.Module):
    """
    Computes the dice score given target labels
    and model predictions.
    """
    def __init__(self, num_classes, device="cpu"):
        super(DiceScore, self).__init__()
        self.dice_score = tmt.Dice(
            num_classes=num_classes, 
            average='micro'
        ).to(device)
        
    def forward(self, preds, targets):
        preds = torch.argmax(preds, dim=1)
        dice_score = self.dice_score(preds, targets)
        return dice_score