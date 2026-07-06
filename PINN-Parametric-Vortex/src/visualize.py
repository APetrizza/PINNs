import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import OrderedDict
import os

def get_vorticity(model, X_f, Y_f, T_f, R_f, U_f, grid_pts):
    u, v, _ = model(X_f, Y_f, T_f, R_f, U_f)
    
    # EXACT VORTICITY COMPUTATION VIA AUTOGRAD
    # retain_graph=True is crucial here to compute the second derivative (v_x) immediately after
    u_y = torch.autograd.grad(u, Y_f, grad_outputs=torch.ones_like(u), retain_graph=True)[0]
    v_x = torch.autograd.grad(v, X_f, grad_outputs=torch.ones_like(v), create_graph=False)[0]
    
    vorticity = v_x - u_y
    return vorticity.detach().cpu().numpy().reshape(grid_pts, grid_pts)

def load_and_generate_animation(model: nn.Module, model_path: str, device: torch.device, filename: str = "vortex_evolution.gif"):
    print(f"\n--- STARTING POST-PROCESSING ---")
    print(f"Looking for model weights at: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"Error: Model weights not found. Train the model first.")
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
    
    print("Generating Vorticity animation...")
    grid_pts = 80
    x_test = torch.linspace(-2, 2, grid_pts)
    y_test = torch.linspace(-2, 2, grid_pts)
    X, Y = torch.meshgrid(x_test, y_test, indexing='ij')
    
    X_flat = X.reshape(-1, 1).to(device).requires_grad_(True)
    Y_flat = Y.reshape(-1, 1).to(device).requires_grad_(True)
    
    R_case1 = (torch.ones_like(X_flat) * 0.15).to(device)
    U_case1 = (torch.ones_like(X_flat) * 2.0).to(device)
    R_case2 = (torch.ones_like(X_flat) * 0.40).to(device)
    U_case2 = (torch.ones_like(X_flat) * 8.0).to(device)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    levels = np.linspace(-10, 10, 50)
    
    cax1 = ax1.contourf(X.numpy(), Y.numpy(), np.zeros((grid_pts, grid_pts)), levels=levels, cmap='RdBu')
    cax2 = ax2.contourf(X.numpy(), Y.numpy(), np.zeros((grid_pts, grid_pts)), levels=levels, cmap='RdBu')
    
    fig.subplots_adjust(left=0.05, right=0.85, top=0.82, bottom=0.15, wspace=0.3)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.67])
    fig.colorbar(cax1, cax=cbar_ax, label=r'Vorticity $\omega$')
    
    def update(frame):
        t_val = frame / 20.0
        T_flat = (torch.ones_like(X_flat) * t_val).to(device)
        
        vort1 = get_vorticity(model, X_flat, Y_flat, T_flat, R_case1, U_case1, grid_pts)
        vort2 = get_vorticity(model, X_flat, Y_flat, T_flat, R_case2, U_case2, grid_pts)
            
        ax1.clear()
        ax2.clear()
        
        ax1.contourf(X.numpy(), Y.numpy(), vort1, levels=levels, cmap='RdBu', extend='both')
        ax2.contourf(X.numpy(), Y.numpy(), vort2, levels=levels, cmap='RdBu', extend='both')
        
        ax1.set_title(r'Vortex $R=0.15$, Wind $U=2.0$ m/s')
        ax2.set_title(r'Vortex $R=0.40$, Wind $U=8.0$ m/s')
        
        ax1.set_xlabel('X axis')
        ax1.set_ylabel('Y axis')
        ax2.set_xlabel('X axis')
        
        fig.suptitle(f'Oseen-Lamb Vortex Evolution | Time $t = {t_val:.2f}$ s', fontsize=16, fontweight='bold')
        
    ani = animation.FuncAnimation(fig, update, frames=20, interval=200)
    ani.save(filename, writer='pillow')
    print(f"PROCESSING COMPLETE! Animation saved as: {filename}")
    plt.close()