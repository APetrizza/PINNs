import torch
import numpy as np
from src.pinn import HeatPINN
from src.visualize import generate_heatmap

def main():
    # Setup seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo in uso: {device}\n")
    
    # Parametri Fisici e Architettura
    alpha = 0.01
    layers = [2, 64, 64, 64, 64, 1]  # 4 hidden layers da 64 neuroni
    
    # Inizializza modello
    model = HeatPINN(layers=layers, alpha=alpha).to(device)
    
    # Addestramento
    model.train_model(device=device, epochs_adam=3000, epochs_lbfgs=500)
    
    # Visualizzazione (Genera mappa di calore)
    generate_heatmap(model, device, filename="heat_transfer_results.png")

if __name__ == "__main__":
    main()