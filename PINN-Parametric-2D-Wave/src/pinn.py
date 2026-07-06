import torch
import torch.nn as nn
from typing import Tuple

class ParametricWavePINN(nn.Module):
    def __init__(self, layers: list):
        super(ParametricWavePINN, self).__init__()
        
        modules = []
        for i in range(len(layers) - 1):
            modules.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:
                modules.append(nn.Tanh())
        self.net = nn.Sequential(*modules)
        
    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, y, t, rho], dim=1))

    def compute_loss(self, x_pde, y_pde, t_pde, rho_pde, 
                     x_ic, y_ic, t_ic, rho_ic, p_ic_true, 
                     x_bc, y_bc, t_bc, rho_bc, p_bc_true) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # --- PDE Loss ---
        x_pde.requires_grad_(True)
        y_pde.requires_grad_(True)
        t_pde.requires_grad_(True)
        
        p = self.forward(x_pde, y_pde, t_pde, rho_pde)
        
        p_t = torch.autograd.grad(p, t_pde, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        p_x = torch.autograd.grad(p, x_pde, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        p_y = torch.autograd.grad(p, y_pde, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        
        p_tt = torch.autograd.grad(p_t, t_pde, grad_outputs=torch.ones_like(p_t), create_graph=True)[0]
        p_xx = torch.autograd.grad(p_x, x_pde, grad_outputs=torch.ones_like(p_x), create_graph=True)[0]
        p_yy = torch.autograd.grad(p_y, y_pde, grad_outputs=torch.ones_like(p_y), create_graph=True)[0]
        
        c_squared = 1.0 / rho_pde
        pde_residual = p_tt - c_squared * (p_xx + p_yy)
        loss_pde = torch.mean(pde_residual**2)
        
        # --- IC Loss 1: Position ---
        p_ic_pred = self.forward(x_ic, y_ic, t_ic, rho_ic)
        loss_ic_1 = torch.mean((p_ic_pred - p_ic_true)**2)
        
        # --- IC Loss 2: Zero Initial Velocity ---
        t_ic.requires_grad_(True)
        p_ic_vel_pred = self.forward(x_ic, y_ic, t_ic, rho_ic)
        p_t_ic = torch.autograd.grad(p_ic_vel_pred, t_ic, grad_outputs=torch.ones_like(p_ic_vel_pred), create_graph=True)[0]
        loss_ic_2 = torch.mean(p_t_ic**2)
        
        # --- BC Loss ---
        p_bc_pred = self.forward(x_bc, y_bc, t_bc, rho_bc)
        loss_bc = torch.mean((p_bc_pred - p_bc_true)**2)
        
        loss_data = loss_ic_1 + loss_ic_2 + loss_bc
        total_loss = loss_pde + loss_data
        
        return total_loss, loss_pde

    def train_model(self, device: torch.device, rho_min: float, rho_max: float, epochs_adam: int = 3000, epochs_lbfgs: int = 500):
        from src.data_gen import get_collocation_points, get_initial_conditions, get_boundary_conditions
        
        N_pde, N_ic, N_bc = 15000, 3000, 3000
        x_ic, y_ic, t_ic, rho_ic, p_ic_true = get_initial_conditions(N_ic, rho_min, rho_max, device)
        x_bc, y_bc, t_bc, rho_bc, p_bc_true = get_boundary_conditions(N_bc, rho_min, rho_max, device)

        print("Inizio fase 1: Adam Optimizer (Campionamento dinamico LHS)")
        optimizer_adam = torch.optim.Adam(self.parameters(), lr=1e-3)
        
        for epoch in range(epochs_adam):
            optimizer_adam.zero_grad()
            x_pde, y_pde, t_pde, rho_pde = get_collocation_points(N_pde, rho_min, rho_max, device)
            
            loss, loss_pde = self.compute_loss(x_pde, y_pde, t_pde, rho_pde, x_ic, y_ic, t_ic, rho_ic, p_ic_true, x_bc, y_bc, t_bc, rho_bc, p_bc_true)
            loss.backward()
            optimizer_adam.step()
            
            if epoch % 500 == 0:
                print(f"Adam Epoch {epoch:4d} | Total Loss: {loss.item():.6f} | PDE Loss: {loss_pde.item():.6f}")

        print("\nInizio fase 2: L-BFGS Optimizer (Fine-tuning)")
        optimizer_lbfgs = torch.optim.LBFGS(self.parameters(), max_iter=epochs_lbfgs, line_search_fn="strong_wolfe")
        x_pde_f, y_pde_f, t_pde_f, rho_pde_f = get_collocation_points(N_pde, rho_min, rho_max, device)
        
        def closure():
            optimizer_lbfgs.zero_grad()
            loss, _ = self.compute_loss(x_pde_f, y_pde_f, t_pde_f, rho_pde_f, x_ic, y_ic, t_ic, rho_ic, p_ic_true, x_bc, y_bc, t_bc, rho_bc, p_bc_true)
            loss.backward()
            return loss

        optimizer_lbfgs.step(closure)
        final_loss, final_pde = self.compute_loss(x_pde_f, y_pde_f, t_pde_f, rho_pde_f, x_ic, y_ic, t_ic, rho_ic, p_ic_true, x_bc, y_bc, t_bc, rho_bc, p_bc_true)
        print(f"Addestramento Completato! Final Loss: {final_loss.item():.7f} | Final PDE Loss: {final_pde.item():.7f}")