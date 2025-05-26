"""
Script to run GNN Shortest Path training in Google Colab.
This script handles installation, setup, and training in one go.
"""

import os
import sys
import subprocess
import torch
from google.colab import drive
from src.train import train_and_evaluate

def mount_google_drive():
    """Mount Google Drive and create necessary directories."""
    print("Mounting Google Drive...")
    drive.mount('/content/drive')
    
    # Create directory for model checkpoints
    model_dir = '/content/drive/MyDrive/gnn_shortest_path'
    os.makedirs(model_dir, exist_ok=True)
    print(f"Model directory created at: {model_dir}")
    return model_dir

def setup_environment():
    """Setup the environment and install required packages."""
    print("Setting up environment...")
    
    # Install required packages
    packages = ['torch', 'torch-geometric', 'networkx', 'matplotlib', 'numpy', 'tqdm']
    for package in packages:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
    
    # Verify CUDA installation
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")

def verify_gpu():
    """Verify GPU availability and print information."""
    print("\nVerifying GPU setup...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"GPU Memory cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    else:
        print("WARNING: CUDA is not available. Training will be slow on CPU.")

def save_model_callback(model, optimizer, epoch, accuracy, path_accuracy, avg_error):
    """Callback function to save model checkpoints to Google Drive."""
    model_dir = '/content/drive/MyDrive/gnn_shortest_path'
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': accuracy,
        'path_accuracy': path_accuracy,
        'avg_error': avg_error,
    }
    
    # Save best model
    torch.save(checkpoint, os.path.join(model_dir, 'best_gnn_shortest_path.pt'))
    print(f"Best model saved with accuracy: {accuracy:.2f}%")
    
    # Save final model
    torch.save(checkpoint, os.path.join(model_dir, 'final_gnn_shortest_path.pt'))
    print(f"Final model saved at epoch {epoch + 1}")

def main():
    """Main function to run the training process."""
    # Setup environment
    setup_environment()
    
    # Mount Google Drive
    model_dir = mount_google_drive()
    
    # Run training with save callback
    print("Starting training...")
    train_and_evaluate(resume_training=False, save_callback=save_model_callback)
    
    print("Training completed!")
    print(f"Model checkpoints saved in: {model_dir}")

if __name__ == "__main__":
    main() 