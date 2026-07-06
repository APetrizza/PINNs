import torch
import torch.nn as nn
import numpy as np
import os
from src.pinn import Surrogate_PINN_3D
from src.visualize import load_and_generate_animation

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.device_count() > 1:
        print(f"Multi-GPU support activated! Detected GPUs: {torch.cuda.device_count()}")
        
    alpha_min, alpha_max = 0.01, 0.1
    layers = [5, 128, 128, 128, 128, 128, 1] # Input 5D -> Output 1D
    
    model = Surrogate_PINN_3D(layers=layers)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        
    model = model.to(device)
    
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'surrogate_3d_model.pth')
    
    # Smart Orchestration
    if not os.path.exists(model_path):
        print("\nNo pre-trained weights found. Starting training phase...")
        model.train_model(device=device, model_path=model_path, alpha_min=alpha_min, alpha_max=alpha_max, epochs=4000)
    else:
        print("\nPre-trained model found! Skipping training phase.")
        
    # Post-processing Phase
    base_model = model.module if isinstance(model, nn.DataParallel) else model
    load_and_generate_animation(base_model, model_path, device, filename="pinn_3d_heat_transfer.gif")

if __name__ == "__main__":
    main()