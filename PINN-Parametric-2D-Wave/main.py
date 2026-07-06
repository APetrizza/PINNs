import torch
import numpy as np
from src.pinn import ParametricWavePINN
from src.visualize import generate_split_screen_animation

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo in uso: {device}\n")
    
    rho_min, rho_max = 1.0, 4.0
    layers = [4, 64, 64, 64, 64, 64, 1]  # 4 inputs, 5 hidden layers, 1 output
    
    model = ParametricWavePINN(layers=layers).to(device)
    
    model.train_model(device=device, rho_min=rho_min, rho_max=rho_max, epochs_adam=3000, epochs_lbfgs=500)
    
    generate_split_screen_animation(model, device, filename="parametric_2d_explosion.gif")

if __name__ == "__main__":
    main()