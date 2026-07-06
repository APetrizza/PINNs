import torch
import numpy as np
from scipy.stats import qmc
from typing import Tuple

def get_collocation_points(N_pde: int, rho_min: float, rho_max: float, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """LHS Sampling for 4D space: x, y, t, rho."""
    sampler = qmc.LatinHypercube(d=4)
    sample = sampler.random(n=N_pde)
    
    x = torch.tensor(sample[:, 0] * 2 - 1, dtype=torch.float32).view(-1, 1).to(device)
    y = torch.tensor(sample[:, 1] * 2 - 1, dtype=torch.float32).view(-1, 1).to(device)
    t = torch.tensor(sample[:, 2], dtype=torch.float32).view(-1, 1).to(device)
    rho = torch.tensor(sample[:, 3] * (rho_max - rho_min) + rho_min, dtype=torch.float32).view(-1, 1).to(device)
    
    return x, y, t, rho

def get_initial_conditions(N_ic: int, rho_min: float, rho_max: float, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generates ICs: Explosion peak at center, zero initial velocity."""
    x_ic = (torch.rand(N_ic, 1) * 2 - 1).to(device)
    y_ic = (torch.rand(N_ic, 1) * 2 - 1).to(device)
    t_ic = torch.zeros(N_ic, 1).to(device)
    rho_ic = (torch.rand(N_ic, 1) * (rho_max - rho_min) + rho_min).to(device)
    
    p_ic_true = torch.exp(-(x_ic**2 + y_ic**2) / 0.02).to(device)
    
    return x_ic, y_ic, t_ic, rho_ic, p_ic_true

def get_boundary_conditions(N_bc: int, rho_min: float, rho_max: float, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generates BCs: Zero pressure at edges."""
    x_bc = torch.cat([-torch.ones(N_bc//4, 1), torch.ones(N_bc//4, 1), torch.rand(N_bc//2, 1)*2-1]).to(device)
    y_bc = torch.cat([torch.rand(N_bc//2, 1)*2-1, -torch.ones(N_bc//4, 1), torch.ones(N_bc//4, 1)]).to(device)
    t_bc = torch.rand(N_bc, 1).to(device)
    rho_bc = (torch.rand(N_bc, 1) * (rho_max - rho_min) + rho_min).to(device)
    
    p_bc_true = torch.zeros(N_bc, 1).to(device)
    
    return x_bc, y_bc, t_bc, rho_bc, p_bc_true