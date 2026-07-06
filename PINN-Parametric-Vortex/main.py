import torch
import torch.nn as nn
import numpy as np
import os
from src.pinn import ParametricVortexPINN
from src.visualize import load_and_generate_animation

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.device_count() > 1:
        print(f"Multi-GPU support activated! Detected GPUs: {torch.cuda.device_count()}")
        
    RHO = 1.225
    NU = 1.5e-5
    R_bounds = (0.1, 0.5)
    U_bounds = (1.0, 10.0)
    
    layers = [5, 256, 256, 256, 256, 256, 256, 3] # Input 5D -> Output 3D (u,v,p)
    
    model = ParametricVortexPINN(layers=layers, rho=RHO, nu=NU)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        
    model = model.to(device)
    
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vortex_navier_stokes.pth')
    
    if not os.path.exists(model_path):
        print("\nNo pre-trained weights found. Starting training phase...")
        model.train_model(device=device, model_path=model_path, R_bounds=R_bounds, U_bounds=U_bounds, epochs=5000)
    else:
        print("\nPre-trained model found! Skipping training phase.")
        
    base_model = model.module if isinstance(model, nn.DataParallel) else model
    load_and_generate_animation(base_model, model_path, device, filename="vortex_evolution.gif")

if __name__ == "__main__":
    main()