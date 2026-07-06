import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def generate_split_screen_animation(model: torch.nn.Module, device: torch.device, filename: str = "parametric_2d_explosion.gif"):
    print("\nGenerazione dell'animazione Split-Screen (potrebbe richiedere qualche minuto)...")
    
    grid_points = 60
    x_test = torch.linspace(-1, 1, grid_points)
    y_test = torch.linspace(-1, 1, grid_points)
    X, Y = torch.meshgrid(x_test, y_test, indexing='ij')

    X_flat = X.reshape(-1, 1).to(device)
    Y_flat = Y.reshape(-1, 1).to(device)

    rho_low = (torch.ones_like(X_flat) * 1.0).to(device)
    rho_high = (torch.ones_like(X_flat) * 4.0).to(device)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    cax1 = ax1.contourf(X.numpy(), Y.numpy(), np.zeros((grid_points, grid_points)), 50, cmap='RdBu_r', vmin=-0.5, vmax=1.0)
    fig.colorbar(cax1, ax=[ax1, ax2], label='Pressure Anomaly (p)', location='bottom', shrink=0.5)

    def update(frame):
        t_val = frame / 30.0 
        T_flat = (torch.ones_like(X_flat) * t_val).to(device)
        
        with torch.no_grad():
            p_low = model(X_flat, Y_flat, T_flat, rho_low).cpu().numpy().reshape(grid_points, grid_points)
            p_high = model(X_flat, Y_flat, T_flat, rho_high).cpu().numpy().reshape(grid_points, grid_points)
            
        ax1.clear()
        ax2.clear()
        
        ax1.contourf(X.numpy(), Y.numpy(), p_low, 50, cmap='magma', vmin=-0.2, vmax=1.0)
        ax2.contourf(X.numpy(), Y.numpy(), p_high, 50, cmap='magma', vmin=-0.2, vmax=1.0)
        
        ax1.set_title(r'Low Density ($\rho = 1.0$)' + '\nFast Shockwave')
        ax2.set_title(r'High Density ($\rho = 4.0$)' + '\nSlow Shockwave')
        
        fig.suptitle(f'Parametric PINN: 2D Explosion Surrogate Model | t = {t_val:.2f} s', fontsize=14, fontweight='bold')
        return ax1, ax2

    ani = animation.FuncAnimation(fig, update, frames=30, interval=150, blit=False)
    ani.save(filename, writer='pillow', fps=10)
    print(f"Animazione salvata con successo: {filename}")
    plt.close()