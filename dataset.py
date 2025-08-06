import torch
from torch.utils.data import Dataset
import numpy as np

class UltrasoundNpyDataset_NoTransforms(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        image_np = self.x_data[idx]
        mask_np = self.y_data[idx]

        # Normalize image to [0, 1] based on min/max of the dataset
        image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min() + 1e-6)
        image_tensor = torch.from_numpy(image_np).float()
        if image_tensor.ndim == 3 and image_tensor.shape[-1] in [1, 3]:
            image_tensor = image_tensor.permute(2, 0, 1)
        elif image_tensor.ndim == 2:
            image_tensor = image_tensor.unsqueeze(0)

        # Process mask: ensure it’s 2D (H, W) and convert to binary float
        if mask_np.ndim == 3 and mask_np.shape[-1] == 1:
            mask_np = np.squeeze(mask_np, axis=-1)
        mask_np = (mask_np > 0).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_np).float()
        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)

        return image_tensor, mask_tensor