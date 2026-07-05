import torch
import numpy as np
from src.pinn import BurgersPINN
from src.visualize import create_and_save_animation

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo in uso: {device}\n")
    
    nu = 0.01 / np.pi
    layers = [2, 40, 40, 40, 40, 1]
    
    model = BurgersPINN(layers=layers, nu=nu).to(device)
    
    # Avvia l'addestramento
    model.train_model(device=device, epochs_adam=2500, epochs_lbfgs=100)
    
    # Crea e salva la GIF
    create_and_save_animation(model, device, filename="burgers_shock_wave.gif")

if __name__ == "__main__":
    main()