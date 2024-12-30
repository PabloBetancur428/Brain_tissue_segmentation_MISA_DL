import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedTverskyCELoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-5, ce_weight=0.5, tversky_weight=0.5):
        """
        alpha: Weight for False Negatives (FN) in Tversky Loss
        beta:  Weight for False Positives (FP) in Tversky Loss
        smooth: Small value to avoid division by zero
        ce_weight: Weight for the Cross-Entropy loss component
        tversky_weight: Weight for the Tversky loss component
        """
        super(CombinedTverskyCELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.ce_weight = ce_weight
        self.tversky_weight = tversky_weight
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, preds, targets):
        """
        preds: [B, C, H, W] - Raw logits from the model
        targets: [B, H, W] - Ground truth labels (integer class IDs)
        """
        preds_softmax = F.softmax(preds, dim=1)  # Convert logits to probabilities
        targets_one_hot = torch.zeros_like(preds_softmax).scatter_(1, targets.unsqueeze(1), 1)  # One-hot encoding

        # Compute true positives, false negatives, and false positives
        true_pos = (preds_softmax * targets_one_hot).sum(dim=(2, 3))  # Sum over spatial dimensions
        false_neg = ((1 - preds_softmax) * targets_one_hot).sum(dim=(2, 3))
        false_pos = (preds_softmax * (1 - targets_one_hot)).sum(dim=(2, 3))

        # Compute Tversky index
        tversky_index = (true_pos + self.smooth) / (
            true_pos + self.alpha * false_neg + self.beta * false_pos + self.smooth
        )
        tversky_loss = 1 - tversky_index.mean()

        # Compute Cross-Entropy Loss
        ce_loss = self.ce_loss(preds, targets)

        # Weighted Combination of Losses
        total_loss = self.ce_weight * ce_loss + self.tversky_weight * tversky_loss
        return total_loss
    
def weighted_dice_loss(pred, target, class_weights):
    """
    pred:   Tensor of shape [B, C, H, W], raw logits from the model (NOT softmaxed yet).
    target: Tensor of shape [B, H, W] (integer labels) or [B, C, H, W] (one-hot).
    class_weights: Tensor of shape [C] (weight for each class).
    """
    # 1) Convert logits -> probabilities with softmax
    pred_softmax = torch.softmax(pred, dim=1)

    # 2) If target is [B, H, W], convert to one-hot -> [B, C, H, W]
    if target.dim() == 3:
        target_one_hot = torch.zeros_like(pred_softmax).scatter_(1, target.unsqueeze(1), 1)
    else:
        # Already one-hot
        target_one_hot = target

    smooth = 1e-5
    dice_loss = 0.0
    C = pred_softmax.shape[1]  # number of classes

    # 3) Compute Dice for each class, weighted by class_weights[c]
    for c in range(C):
        pred_c   = pred_softmax[:, c, :, :]
        target_c = target_one_hot[:, c, :, :]

        intersection = (pred_c * target_c).sum()
        union        = pred_c.sum() + target_c.sum() + smooth
        dice_score_c = (2.0 * intersection) / union

        # Weighted dice loss for class c
        dice_loss_c = (1.0 - dice_score_c) * class_weights[c]
        dice_loss  += dice_loss_c

    return dice_loss