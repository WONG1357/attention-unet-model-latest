import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

def create_attention_unet(in_channels=1):
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=1,
        attention_type='scse'
    )
    return model

class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, preds, targets):
        bce_loss = self.bce(preds, targets)
        preds_sigmoid = torch.sigmoid(preds)
        intersection = (preds_sigmoid * targets).sum()
        union = preds_sigmoid.sum() + targets.sum()
        dice_loss = 1 - (2. * intersection + 1e-6) / (union + 1e-6)
        return bce_loss + dice_loss