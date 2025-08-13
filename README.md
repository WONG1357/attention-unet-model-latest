## Ultrasound Muscle Segmentation with Attention U-Net

This repository contains code for training and evaluating an Attention U-Net model for segmenting the rectus femoris and vastus medialis muscles in ultrasound images. The model is built using PyTorch and the `segmentation-models-pytorch` library.


## Project Structure

- `dataset.py`: Defines the `UltrasoundNpyDataset_NoTransforms` class for loading and preprocessing ultrasound images and masks.
- `model.py`: Contains the Attention U-Net model definition and the combined loss function (BCE + Dice).
- `utils.py`: Includes utility functions for calculating the Dice score, post-processing masks, and visualizing results.
- `train.py`: Handles data loading, model training, and saving the best model based on validation Dice score.
- `evaluate.py`: Manages evaluation of the model on train and test sets, including visualization and saving of predictions.
- `main.py`: Orchestrates the training and evaluation for both rectus femoris and vastus medialis datasets.
- `requirements.txt`: Lists the required Python packages.

## Setup

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install Dependencies**:
   Install the required packages using:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare Data**:
   - Ensure your data is stored in `.npy` format for training, validation, and test sets (`X_train.npy`, `y_train.npy`, `X_val.npy`, `y_val.npy`, `X_test.npy`, `y_test.npy`).
   - Update the data paths in `main.py` to point to your data directories:
     - For rectus femoris: `/content/drive/MyDrive/intern RF transverse latest file/`
     - For vastus medialis: `/content/drive/MyDrive/intern RF longitudinal latest file/`
   - Update the model save paths and output directories in `main.py` as needed.

4. **Google Drive Setup** (if running on Google Colab):
   - Mount your Google Drive in Colab:
     ```python
     from google.colab import drive
     drive.mount('/content/drive')
     ```
   - Ensure the data and model save paths are accessible in your Google Drive.

## Usage

Run the main script to train and evaluate the model for both muscles:
```bash
python main.py
```

- The script will:
  - Train the Attention U-Net model for the rectus femoris dataset (100 epochs, early stopping with patience=10).
  - Evaluate and save predictions for the rectus femoris dataset.
  - Train the model for the vastus medialis dataset (50 epochs, early stopping with patience=50).
  - Evaluate and save predictions for the vastus medialis dataset.
- Outputs (saved predictions and trained models) will be stored in the directories specified in `main.py`.

## Notes

- The model uses a ResNet34 backbone with pre-trained ImageNet weights and SCSE attention.
- The input images are grayscale (1 channel), and the model outputs a single-channel binary mask.
- The combined loss function uses BCE and Dice loss.
- Post-processing retains the largest connected component and fills holes in the predicted masks.
- Predictions are visualized and saved as PNG files, comparing the input image, ground truth, raw prediction, and post-processed prediction.

## Requirements

See `requirements.txt` for the list of required packages. Key dependencies include:
- PyTorch
- NumPy
- segmentation-models-pytorch
- Matplotlib
- TQDM
- SciPy


