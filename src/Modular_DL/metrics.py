import torch

def dice_coefficient(pred, target, epsilon=1e-6):
    """
    Compute the Dice Coefficient.
    Args:
        pred: Predicted output (logits or probabilities).
        target: Ground truth labels (one-hot or integer).
        epsilon: Small value to avoid division by zero.
    """
    pred = torch.argmax(pred, dim=1)  # Convert logits to class predictions
    pred = pred.float()
    target = target.float()

    intersection = torch.sum(pred * target)
    union = torch.sum(pred) + torch.sum(target)

    dice = (2.0 * intersection + epsilon) / (union + epsilon)
    return dice