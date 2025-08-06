import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from model import create_attention_unet, CombinedLoss
from dataset import UltrasoundNpyDataset_NoTransforms
from utils import dice_score
import os

def train_model(data_folder, model_save_path, num_epochs=100, batch_size=8, in_channels=1, patience=10):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load data
    x_train = np.load(os.path.join(data_folder, 'X_train.npy'))
    y_train = np.load(os.path.join(data_folder, 'y_train.npy'))
    x_val = np.load(os.path.join(data_folder, 'X_val.npy'))
    y_val = np.load(os.path.join(data_folder, 'y_val.npy'))

    # Create datasets
    train_dataset = UltrasoundNpyDataset_NoTransforms(x_train, y_train)
    val_dataset = UltrasoundNpyDataset_NoTransforms(x_val, y_val)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Created {len(train_dataset)} training examples.")
    print(f"Created {len(val_dataset)} validation examples.")

    # Initialize model, loss, optimizer, and scheduler
    model = create_attention_unet(in_channels=in_channels)
    model.to(device)
    loss_fn = CombinedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

    best_val_dice = 0.0
    trigger_times = 0

    for epoch in range(num_epochs):
        print(f"--- Epoch {epoch+1}/{num_epochs} ---")

        # Training phase
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        for images, masks in tqdm(train_loader, desc="Training"):
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            if masks.dim() != outputs.dim() or masks.size() != outputs.size():
                masks = masks.squeeze(-1)
            loss = loss_fn(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_dice += dice_score(outputs, masks).item()

        avg_train_loss = train_loss / len(train_loader)
        avg_train_dice = train_dice / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        with torch.no_grad():
            for i, (images, masks) in enumerate(tqdm(val_loader, desc="Validation")):
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                if masks.dim() != outputs.dim() or masks.size() != outputs.size():
                    masks = masks.squeeze(-1)
                loss = loss_fn(outputs, masks)
                val_loss += loss.item()
                val_dice += dice_score(outputs, masks).item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)

        scheduler.step(avg_val_dice)

        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            trigger_times = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"New best model saved at epoch {epoch+1} with Val Dice: {avg_val_dice:.4f}")
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"Early stopping triggered at epoch {epoch+1} with no improvement.")
                break

        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Dice: {avg_train_dice:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f}, Val Dice:   {avg_val_dice:.4f}\n")

    print("Training finished!")
    print(f"Best model saved with Validation Dice: {best_val_dice:.4f}")