import torch
import torch.nn as nn
import os
from typing import Tuple

class ParametricVortexPINN(nn.Module):
    def __init__(self, layers: list, rho: float, nu: float):
        super(ParametricVortexPINN, self).__init__()
        self.rho = rho
        self.nu = nu
        
        modules = []
        for i in range(len(layers) - 1):
            modules.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:
                modules.append(nn.Tanh())
        self.net = nn.Sequential(*modules)

    def forward(self, x, y, t, R, U0):
        out = self.net(torch.cat([x, y, t, R, U0], dim=1))
        u = out[:, 0:1]
        v = out[:, 1:2]
        p = out[:, 2:3]
        return u, v, p

    def compute_loss(self, x_pde, y_pde, t_pde, R_pde, U0_pde,
                     x_ic, y_ic, t_ic, R_ic, U0_ic, u_ic_true, v_ic_true) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        # --- PDE Loss (Navier-Stokes) ---
        x_pde.requires_grad_(True)
        y_pde.requires_grad_(True)
        t_pde.requires_grad_(True)
        
        u, v, p = self.forward(x_pde, y_pde, t_pde, R_pde, U0_pde)
        
        u_t = torch.autograd.grad(u, t_pde, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x_pde, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_y = torch.autograd.grad(u, y_pde, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        
        v_t = torch.autograd.grad(v, t_pde, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_x = torch.autograd.grad(v, x_pde, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_y = torch.autograd.grad(v, y_pde, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        
        p_x = torch.autograd.grad(p, x_pde, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        p_y = torch.autograd.grad(p, y_pde, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        
        u_xx = torch.autograd.grad(u_x, x_pde, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y_pde, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x, x_pde, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
        v_yy = torch.autograd.grad(v_y, y_pde, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]
        
        # Continuity
        f_c = u_x + v_y
        # Momentum
        f_u = u_t + (u * u_x + v * u_y) + (1.0 / self.rho) * p_x - self.nu * (u_xx + u_yy)
        f_v = v_t + (u * v_x + v * v_y) + (1.0 / self.rho) * p_y - self.nu * (v_xx + v_yy)
        
        loss_pde = torch.mean(f_c**2) + torch.mean(f_u**2) + torch.mean(f_v**2)
        
        # --- IC Loss ---
        u_ic_pred, v_ic_pred, _ = self.forward(x_ic, y_ic, t_ic, R_ic, U0_ic)
        loss_ic = torch.mean((u_ic_pred - u_ic_true)**2) + torch.mean((v_ic_pred - v_ic_true)**2)
        
        # Heavily weight the IC to enforce vortex formation
        total_loss = loss_pde + loss_ic * 10.0
        return total_loss, loss_pde, loss_ic

    def train_model(self, device: torch.device, model_path: str, R_bounds: tuple, U_bounds: tuple, epochs: int = 5000):
        from src.data_gen import get_collocation_points, get_initial_conditions
        
        N_pde, N_ic = 60000, 15000  # HPC Scale
        
        print("Generating IC datasets...")
        x_ic, y_ic, t_ic, R_ic, U0_ic, u_ic_true, v_ic_true = get_initial_conditions(N_ic, R_bounds, U_bounds, device)
        
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        
        print("Starting Training (Navier-Stokes requires patience!)...")
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            x_pde, y_pde, t_pde, R_pde, U0_pde = get_collocation_points(N_pde, R_bounds, U_bounds, device)
            
            loss, loss_pde, loss_ic = self.compute_loss(x_pde, y_pde, t_pde, R_pde, U0_pde,
                                                        x_ic, y_ic, t_ic, R_ic, U0_ic, u_ic_true, v_ic_true)
            loss.backward()
            optimizer.step()
            
            if epoch % 500 == 0:
                print(f"Epoch {epoch:4d}/{epochs} | Total Loss: {loss.item():.5f} | PDE: {loss_pde.item():.5f} | IC: {loss_ic.item():.5f}")

        model_to_save = self.module if isinstance(self, nn.DataParallel) else self
        torch.save(model_to_save.state_dict(), model_path)
        print(f"Model weights saved to '{model_path}'")