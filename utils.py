import torch
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

def dice_score(preds, targets, smooth=1e-6):
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()
    preds = preds.view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    return (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)

def post_process_mask(mask):
    labels, num_features = ndimage.label(mask)
    if num_features == 0:
        return mask
    component_sizes = np.bincount(labels.ravel())
    if len(component_sizes) > 1:
        largest_component_label = component_sizes[1:].argmax() + 1
        processed_mask = (labels == largest_component_label)
        processed_mask = ndimage.binary_fill_holes(processed_mask)
        return processed_mask.astype(np.uint8)
    return mask

def visualize_and_save(processed_img, gt_mask, pred_raw, pred_post, save_path, title):
    if processed_img.shape[0] == 1:
        processed_img_display = processed_img.cpu().squeeze(0).numpy()
        processed_img_display = (processed_img_display - processed_img_display.min()) / (processed_img_display.max() - processed_img_display.min())
    else:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        processed_img_display = processed_img.cpu().permute(1, 2, 0).numpy()
        processed_img_display = std * processed_img_display + mean
        processed_img_display = np.clip(processed_img_display, 0, 1)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(processed_img_display, cmap='gray')
    axes[0].set_title("Original Black-White Input")
    axes[0].axis('off')
    axes[1].imshow(np.squeeze(gt_mask), cmap='gray')
    axes[1].set_title("Ground Truth")
    axes[1].axis('off')
    axes[2].imshow(np.squeeze(pred_raw), cmap='gray')
    axes[2].set_title("Raw Prediction (Before Post-Processing)")
    axes[2].axis('off')
    axes[3].imshow(np.squeeze(pred_post), cmap='gray')
    axes[3].set_title("Post-Processed Prediction")
    axes[3].axis('off')
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)