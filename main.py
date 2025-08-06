import os
from train import train_model
from evaluate import evaluate_model

def main():
    # Define paths for rectus femoris
    rf_data_folder = '/content/drive/MyDrive/intern RF transverse latest file/'
    rf_model_save_path = '/content/drive/MyDrive/internship models/attention unet model/rectus femoris/attentionunet_resnet34_best.pth'
    rf_base_save_dir = '/content/drive/MyDrive/internship models/attention unet model/rectus femoris/segmentation_results_with_preprocessing'

    # Train and evaluate for rectus femoris
    print("Processing Rectus Femoris...")
    train_model(rf_data_folder, rf_model_save_path, num_epochs=100, batch_size=8, in_channels=1, patience=10)
    evaluate_model(rf_data_folder, rf_model_save_path, rf_base_save_dir, in_channels=1, batch_size=8)

    # Define paths for vastus medialis
    vm_data_folder = '/content/drive/MyDrive/intern RF longitudinal latest file/'
    vm_model_save_path = '/content/drive/MyDrive/internship models/attention unet model/vastus medialis/attentionunet_resnet34_best.pth'
    vm_base_save_dir = '/content/drive/MyDrive/internship models/attention unet model/vastus medialis/segmentation_results_with_preprocessing'

    # Train and evaluate for vastus medialis
    print("\nProcessing Vastus Medialis...")
    train_model(vm_data_folder, vm_model_save_path, num_epochs=50, batch_size=8, in_channels=1, patience=50)
    evaluate_model(vm_data_folder, vm_model_save_path, vm_base_save_dir, in_channels=1, batch_size=8)

if __name__ == "__main__":
    main()