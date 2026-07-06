import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import OrderedDict
import os

def load_and_generate_animation(model: nn.Module, model_path: str, device: torch.device, filename: str = "pinn_3d_heat_transfer.gif"):
    print(f"\n--- STARTING POST-PROCESSING ---")
    print(f"Looking for model weights at: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}. Please train the model first.")
        return

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    print("Model loaded successfully!")
    
    print("Generating 2D cross-section animation (Z=0)...")
    grid_pts = 60  
    x_test = torch.linspace(-1, 1, grid_pts)
    y_test = torch.linspace(-1, 1, grid_pts)
    X, Y = torch.meshgrid(x_test, y_test, indexing='ij')
    
    X_flat = X.reshape(-1, 1).to(device)
    Y_flat = Y.reshape(-1, 1).to(device)
    Z_flat = torch.zeros_like(X_flat).to(device)
    
    alpha_ceramic = (torch.ones_like(X_flat) * 0.01).to(device)
    alpha_metal = (torch.ones_like(X_flat) * 0.1).to(device)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    
    levels = np.linspace(0.0, 1.0, 50)
    
    cax1 = ax1.contourf(X.numpy(), Y.numpy(), np.zeros((grid_pts, grid_pts)), levels=levels, cmap='inferno')
    cax2 = ax2.contourf(X.numpy(), Y.numpy(), np.zeros((grid_pts, grid_pts)), levels=levels, cmap='inferno')
    
    fig.subplots_adjust(left=0.05, right=0.85, top=0.82, bottom=0.15, wspace=0.3)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.67])
    cbar = fig.colorbar(cax1, cax=cbar_ax, label='Temperature $u(x,y,0,t)$')
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    
    def update(frame):
        t_val = frame / 20.0
        T_flat = (torch.ones_like(X_flat) * t_val).to(device)
        
        with torch.no_grad():
            u_cer = model(X_flat, Y_flat, Z_flat, T_flat, alpha_ceramic).cpu().numpy().reshape(grid_pts, grid_pts)
            u_met = model(X_flat, Y_flat, Z_flat, T_flat, alpha_metal).cpu().numpy().reshape(grid_pts, grid_pts)
            
        ax1.clear()
        ax2.clear()
        
        ax1.contourf(X.numpy(), Y.numpy(), u_cer, levels=levels, cmap='inferno')
        ax2.contourf(X.numpy(), Y.numpy(), u_met, levels=levels, cmap='inferno')
        
        ax1.set_title(r'Ceramic ($\alpha=0.01$)' + '\nCross-section $Z=0$')
        ax2.set_title(r'Metal ($\alpha=0.1$)' + '\nCross-section $Z=0$')
        
        ax1.set_xlabel('X axis')
        ax1.set_ylabel('Y axis')
        ax2.set_xlabel('X axis')
        
        fig.suptitle(f'3D Parametric Heat PINN | Time $t = {t_val:.2f}$ s', fontsize=16, fontweight='bold')
        
    ani = animation.FuncAnimation(fig, update, frames=20, interval=200)
    ani.save(filename, writer='pillow')
    print(f"PROCESSING COMPLETE! Animation saved as: {filename}")
    plt.close()