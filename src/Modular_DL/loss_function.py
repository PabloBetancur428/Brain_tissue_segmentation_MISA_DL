import torch.nn.functional as F

def dice_loss(pred, target, smooth = 1e-5):
    pred_softmax = F.softmax(pred, dim = 1)

    num_classes = pred_softmax.shape[1]
    target_one_hot = F.one_hot(target, num_classes).permute(0,3,1,2).float()

    #Intersection
    intersection = (pred_softmax * target_one_hot).sum(dim = (2,3))
    union = pred_softmax.sum(dim=(2,3)) + target_one_hot.sum(dim=(2,3))
    dice = (2. * intersection + smooth) / (union + smooth)

    dice = dice.mean(dim=1)

    return 1 - dice