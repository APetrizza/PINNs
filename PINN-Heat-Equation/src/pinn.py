import torch
import torch.nn as nn
from typing import Tuple

class HeatPINN(nn.Module):
    def __init__(self, layers: list, alpha: float):
        super(HeatPINN, self).__init__()
        self.alpha = alpha
        
        modules = []
        for i in range(len(layers) - 1):
            modules.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:
                modules.append(nn.Tanh())
        
        self.net = nn.Sequential(*modules)
        
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, t], dim=1))

    def compute_loss(self, x_pde: torch.Tensor, t_pde: torch.Tensor, 
                     x_ic: torch.Tensor, t_ic: torch.Tensor, u_ic: torch.Tensor, 
                     x_bc: torch.Tensor, t_bc: torch.Tensor, u_bc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # --- Loss PDE (Equazione del Calore) ---
        x_pde.requires_grad_(True)
        t_pde.requires_grad_(True)
        
        u = self.forward(x_pde, t_pde)
        
        u_t = torch.autograd.grad(u, t_pde, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x_pde, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x_pde, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        
        # Calore: u_t - alpha * u_xx = 0
        pde_residual = u_t - self.alpha * u_xx
        loss_pde = torch.mean(pde_residual ** 2)
        
        # --- Loss Dati (IC e BC) ---
        loss_ic = torch.mean((self.forward(x_ic, t_ic) - u_ic) ** 2)
        loss_bc = torch.mean((self.forward(x_bc, t_bc) - u_bc) ** 2)
        
        loss_data = loss_ic + loss_bc
        total_loss = loss_pde + loss_data
        
        return total_loss, loss_pde

    def train_model(self, device: torch.device, epochs_adam: int = 3000, epochs_lbfgs: int = 500):
        from src.data_gen import get_collocation_points, get_initial_conditions, get_boundary_conditions
        
        N_bc, N_ic = 1000, 1000
        x_ic, t_ic, u_ic_true = get_initial_conditions(N_ic, device)
        x_bc, t_bc, u_bc_true = get_boundary_conditions(N_bc, device)

        print("Inizio fase 1: Adam Optimizer (Campionamento dinamico LHS)")
        optimizer_adam = torch.optim.Adam(self.parameters(), lr=1e-3)
        N_pde = 5000
        
        for epoch in range(epochs_adam):
            optimizer_adam.zero_grad()
            x_pde, t_pde = get_collocation_points(N_pde, device)
            
            loss, loss_pde = self.compute_loss(x_pde, t_pde, x_ic, t_ic, u_ic_true, x_bc, t_bc, u_bc_true)
            loss.backward()
            optimizer_adam.step()
            
            if epoch % 500 == 0:
                print(f"Adam Epoch {epoch:4d} | Total Loss: {loss.item():.6f} | PDE Loss: {loss_pde.item():.6f}")

        print("\nInizio fase 2: L-BFGS Optimizer (Fine-tuning)")
        optimizer_lbfgs = torch.optim.LBFGS(self.parameters(), max_iter=epochs_lbfgs, line_search_fn="strong_wolfe")
        
        x_pde_f, t_pde_f = get_collocation_points(N_pde, device)
        
        def closure():
            optimizer_lbfgs.zero_grad()
            loss, _ = self.compute_loss(x_pde_f, t_pde_f, x_ic, t_ic, u_ic_true, x_bc, t_bc, u_bc_true)
            loss.backward()
            return loss

        optimizer_lbfgs.step(closure)
        final_loss, final_pde = self.compute_loss(x_pde_f, t_pde_f, x_ic, t_ic, u_ic_true, x_bc, t_bc, u_bc_true)
        print(f"Addestramento Completato! Final Loss: {final_loss.item():.7f} | Final PDE Loss: {final_pde.item():.7f}")