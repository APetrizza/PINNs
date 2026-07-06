import torch
import matplotlib.pyplot as plt

def generate_heatmap(model: torch.nn.Module, device: torch.device, filename: str = "heat_transfer_results.png"):
    print("\nGenerazione della mappa di calore in corso...")
    
    x_test = torch.linspace(-1, 1, 100)
    t_test = torch.linspace(0, 1, 100)
    
    X, T = torch.meshgrid(x_test, t_test, indexing='ij')
    
    X_flat = X.reshape(-1, 1).to(device)
    T_flat = T.reshape(-1, 1).to(device)
    
    with torch.no_grad():
        U_pred = model(X_flat, T_flat).reshape(100, 100).cpu().numpy()
        
    plt.figure(figsize=(10, 6))
    plt.contourf(T.numpy(), X.numpy(), U_pred, levels=100, cmap='inferno')
    plt.colorbar(label=r'Temperature $u(x,t)$')
    plt.xlabel(r'Time $t$')
    plt.ylabel(r'Space $x$')
    plt.title('PINN Prediction: 1D Transient Heat Transfer')
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Grafico salvato con successo: {filename}")
    plt.close()