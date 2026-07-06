import torch
import torch.nn as nn
import os
from typing import Tuple

class Surrogate_PINN_3D(nn.Module):
    def __init__(self, layers: list):
        super(Surrogate_PINN_3D, self).__init__()
        
        modules = []
        for i in range(len(layers) - 1):
            modules.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:
                modules.append(nn.Tanh())
        self.net = nn.Sequential(*modules)

    def forward(self, x, y, z, t, alpha):
        return self.net(torch.cat([x, y, z, t, alpha], dim=1))

    def compute_loss(self, x_pde, y_pde, z_pde, t_pde, alpha_pde,
                     x_ic, y_ic, z_ic, t_ic, alpha_ic, u_ic_true,
                     x_bc, y_bc, z_bc, t_bc, alpha_bc, u_bc_true) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # --- 3D PDE Loss ---
        x_pde.requires_grad_(True)
        y_pde.requires_grad_(True)
        z_pde.requires_grad_(True)
        t_pde.requires_grad_(True)
        
        u = self.forward(x_pde, y_pde, z_pde, t_pde, alpha_pde)
        
        u_t = torch.autograd.grad(u, t_pde, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x_pde, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_y = torch.autograd.grad(u, y_pde, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_z = torch.autograd.grad(u, z_pde, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        
        u_xx = torch.autograd.grad(u_x, x_pde, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y_pde, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
        u_zz = torch.autograd.grad(u_z, z_pde, grad_outputs=torch.ones_like(u_z), create_graph=True)[0]
        
        laplacian = u_xx + u_yy + u_zz
        f = u_t - alpha_pde * laplacian
        loss_pde = torch.mean(f**2)
        
        # --- IC Loss ---
        u_ic_pred = self.forward(x_ic, y_ic, z_ic, t_ic, alpha_ic)
        loss_ic = torch.mean((u_ic_pred - u_ic_true)**2)

        # --- BC Loss ---
        u_bc_pred = self.forward(x_bc, y_bc, z_bc, t_bc, alpha_bc)
        loss_bc = torch.mean((u_bc_pred - u_bc_true)**2)
        
        total_loss = loss_pde + loss_ic + loss_bc
        return total_loss, loss_pde

    def train_model(self, device: torch.device, model_path: str, alpha_min: float, alpha_max: float, epochs: int = 4000):
        from src.data_gen import get_collocation_points, get_initial_conditions, get_boundary_conditions
        
        N_pde, N_ic, N_bc = 50000, 10000, 12000  # HPC Scale (12000 diviso 6 facce = 2000 punti/faccia)
        
        print("Generating IC and BC datasets...")
        x_ic, y_ic, z_ic, t_ic, alpha_ic, u_ic_true = get_initial_conditions(N_ic, alpha_min, alpha_max, device)
        x_bc, y_bc, z_bc, t_bc, alpha_bc, u_bc_true = get_boundary_conditions(N_bc, alpha_min, alpha_max, device)
        
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        
        print("Starting Training (HPC Scale)...")
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # 5D Resampling at every epoch
            x_pde, y_pde, z_pde, t_pde, alpha_pde = get_collocation_points(N_pde, alpha_min, alpha_max, device)
            
            loss, loss_pde = self.compute_loss(x_pde, y_pde, z_pde, t_pde, alpha_pde,
                                               x_ic, y_ic, z_ic, t_ic, alpha_ic, u_ic_true,
                                               x_bc, y_bc, z_bc, t_bc, alpha_bc, u_bc_true)
            loss.backward()
            optimizer.step()
            
            if epoch % 500 == 0:
                print(f"Epoch {epoch:4d}/{epochs} | Total Loss: {loss.item():.5f} | PDE Loss: {loss_pde.item():.5f}")

        # Extract native state_dict if under DataParallel
        model_to_save = self.module if isinstance(self, nn.DataParallel) else self
        torch.save(model_to_save.state_dict(), model_path)
        print(f"Model weights saved to '{model_path}'")