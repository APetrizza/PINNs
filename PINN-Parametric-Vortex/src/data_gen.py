import torch
import numpy as np
from scipy.stats import qmc
from typing import Tuple

def get_collocation_points(N_pde: int, R_bounds: tuple, U_bounds: tuple, device: torch.device) -> Tuple[torch.Tensor, ...]:
    """5D LHS Sampling: x, y, t, R, U0."""
    sampler = qmc.LatinHypercube(d=5)
    sample = sampler.random(n=N_pde)
    
    x = torch.tensor(sample[:, 0] * 4 - 2, dtype=torch.float32).view(-1, 1).to(device)
    y = torch.tensor(sample[:, 1] * 4 - 2, dtype=torch.float32).view(-1, 1).to(device)
    t = torch.tensor(sample[:, 2], dtype=torch.float32).view(-1, 1).to(device)
    R = torch.tensor(sample[:, 3] * (R_bounds[1] - R_bounds[0]) + R_bounds[0], dtype=torch.float32).view(-1, 1).to(device)
    U0 = torch.tensor(sample[:, 4] * (U_bounds[1] - U_bounds[0]) + U_bounds[0], dtype=torch.float32).view(-1, 1).to(device)
    
    return x, y, t, R, U0

def get_initial_conditions(N_ic: int, R_bounds: tuple, U_bounds: tuple, device: torch.device) -> Tuple[torch.Tensor, ...]:
    """Generates ICs: Oseen-Lamb Vortex mathematical definition."""
    x_ic = (torch.rand(N_ic, 1) * 4 - 2).to(device)
    y_ic = (torch.rand(N_ic, 1) * 4 - 2).to(device)
    t_ic = torch.zeros(N_ic, 1).to(device)
    R_ic = (torch.rand(N_ic, 1) * (R_bounds[1] - R_bounds[0]) + R_bounds[0]).to(device)
    U0_ic = (torch.rand(N_ic, 1) * (U_bounds[1] - U_bounds[0]) + U_bounds[0]).to(device)
    
    r2 = x_ic**2 + y_ic**2
    Gamma = 5.0 * U0_ic * R_ic 
    v_theta = (Gamma / (2 * np.pi * torch.sqrt(r2) + 1e-5)) * (1 - torch.exp(-r2 / R_ic**2))
    
    u_ic_true = U0_ic - v_theta * (y_ic / (torch.sqrt(r2) + 1e-5))
    v_ic_true = v_theta * (x_ic / (torch.sqrt(r2) + 1e-5))
    
    return x_ic, y_ic, t_ic, R_ic, U0_ic, u_ic_true, v_ic_true