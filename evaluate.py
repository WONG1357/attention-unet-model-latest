import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from model import create_attention_unet
from dataset import UltrasoundNpyDataset_NoTransforms
from utils import dice_score, post_process_mask, visualize_and_save

def evaluate_model(data_folder, model_save_path, base_save_dir, in_channels=1, batch_size=8):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load data
    x_test = np.load(os.path.join(data_folder, 'X_test.npy'))
    y_test = np.load(os.path.join(data_folder, 'y_test.npy'))
    x_train = np.load(os.path.join(data_folder, 'X_train.npy'))
    y_train = np.load(os.path.join(data_folder, 'y_train.npy'))

    # Create datasets
    test_dataset = UltrasoundNpyDataset_NoTransforms(x_test, y_test)
    train_dataset = UltrasoundNpyDataset_NoTransforms(x_train, y_train)

    # Create data loaders
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    # Load model
    model = create_attention_unet(in_channels=in_channels)
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.to(device)
    model.eval()

    # Define save directories
    train_save_dir = os.path.join(base_save_dir, 'train_set_predictions')
    test_save_dir = os.path.join(base_save_dir, 'test_set_predictions')
    os.makedirs(train_save_dir, exist_ok=True)
    os.makedirs(test_save_dir, exist_ok=True)

    print(f"Train set predictions will be saved to: {train_save_dir}")
    print(f"Test set predictions will be saved to: {test_save_dir}")

    # Evaluate test set
    print("\n--- Evaluating Test Set ---")
    total_dice_before = 0
    total_dice_after = 0
    num_samples_test = 0
    smooth = 1e-6

    with torch.no_grad():
        for i, (images, gt_masks) in enumerate(tqdm(test_loader, desc="Testing")):
            images = images.to(device)
            gt_masks_np = gt_masks.cpu().numpy()
            preds_logits = model(images)
            preds_sigmoid = torch.sigmoid(preds_logits)
            preds_before_np = (preds_sigmoid > 0.5).cpu().numpy()
            preds_after_np = np.array([post_process_mask(np.squeeze(p)) for p in preds_before_np])

            for j in range(images.shape[0]):
                image_idx = i * test_loader.batch_size + j
                gt = np.squeeze(gt_masks_np[j]).flatten()
                pred_before = np.squeeze(preds_before_np[j]).flatten()
                pred_after = np.squeeze(preds_after_np[j]).flatten()

                intersection_before = (pred_before * gt).sum()
                total_dice_before += (2. * intersection_before + smooth) / (pred_before.sum() + gt.sum() + smooth)
                intersection_after = (pred_after * gt).sum()
                total_dice_after += (2. * intersection_after + smooth) / (pred_after.sum() + gt.sum() + smooth)

                num_samples_test += 1
                save_path = os.path.join(test_save_dir, f"test_prediction_{image_idx+1}.png")
                visualize_and_save(
                    processed_img=images[j],
                    gt_mask=gt_masks_np[j],
                    pred_raw=preds_before_np[j],
                    pred_post=preds_after_np[j],
                    save_path=save_path,
                    title=f"Test Set - Prediction {image_idx+1}"
                )

    avg_dice_before_test = total_dice_before / num_samples_test
    avg_dice_after_test = total_dice_after / num_samples_test
    print(f"\nTest Set Evaluation Complete")
    print(f"Total Test Images Processed: {num_samples_test}")
    print(f"Average Dice (Before Post-Processing): {avg_dice_before_test:.4f}")
    print(f"Average Dice (After Post-Processing):  {avg_dice_after_test:.4f}")

    # Evaluate train set
    print("\n--- Evaluating Train Set ---")
    total_dice_before_train = 0
    total_dice_after_train = 0
    num_samples_train = 0

    with torch.no_grad():
        for i, (images, gt_masks) in enumerate(tqdm(train_loader, desc="Training Set Prediction")):
            images = images.to(device)
            gt_masks_np = gt_masks.cpu().numpy()
            preds_logits = model(images)
            preds_sigmoid = torch.sigmoid(preds_logits)
            preds_before_np = (preds_sigmoid > 0.5).cpu().numpy()
            preds_after_np = np.array([post_process_mask(np.squeeze(p)) for p in preds_before_np])

            for j in range(images.shape[0]):
                image_idx = i * train_loader.batch_size + j
                gt = np.squeeze(gt_masks_np[j]).flatten()
                pred_before = np.squeeze(preds_before_np[j]).flatten()
                pred_after = np.squeeze(preds_after_np[j]).flatten()

                intersection_before = (pred_before * gt).sum()
                total_dice_before_train += (2. * intersection_before + smooth) / (pred_before.sum() + gt.sum() + smooth)
                intersection_after = (pred_after * gt).sum()
                total_dice_after_train += (2. * intersection_after + smooth) / (pred_after.sum() + gt.sum() + smooth)

                num_samples_train += 1
                save_path = os.path.join(train_save_dir, f"train_prediction_{image_idx+1}.png")
                visualize_and_save(
                    processed_img=images[j],
                    gt_mask=gt_masks_np[j],
                    pred_raw=preds_before_np[j],
                    pred_post=preds_after_np[j],
                    save_path=save_path,
                    title=f"Train Set - Prediction {image_idx+1}"
                )

    avg_dice_before_train = total_dice_before_train / num_samples_train
    avg_dice_after_train = total_dice_after_train / num_samples_train
    print(f"\nTrain Set Evaluation Complete")
    print(f"Total Train Images Processed: {num_samples_train}")
    print(f"Average Dice (Before Post-Processing): {avg_dice_before_train:.4f}")
    print(f"Average Dice (After Post-Processing):  {avg_dice_after_train:.4f}")