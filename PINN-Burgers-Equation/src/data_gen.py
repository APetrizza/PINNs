import torch
import numpy as np
from scipy.stats import qmc
from typing import Tuple

def get_collocation_points(N_pde: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Genera i collocation points interni usando Latin Hypercube Sampling (LHS)."""
    sampler = qmc.LatinHypercube(d=2)
    sample = sampler.random(n=N_pde)
    
    x_pde = torch.tensor(sample[:, 0] * 2 - 1, dtype=torch.float32).view(-1, 1).to(device)
    t_pde = torch.tensor(sample[:, 1], dtype=torch.float32).view(-1, 1).to(device)
    
    return x_pde, t_pde

def get_initial_conditions(N_ic: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Genera le condizioni iniziali a t=0."""
    x_ic = (torch.rand(N_ic, 1) * 2 - 1).to(device)
    t_ic = torch.zeros(N_ic, 1).to(device)
    u_ic_true = -torch.sin(np.pi * x_ic)
    return x_ic, t_ic, u_ic_true

def get_boundary_conditions(N_bc: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Genera le condizioni al contorno a x=-1 e x=1."""
    t_bc = torch.rand(N_bc, 1).to(device)
    x_bc = torch.cat([-torch.ones(N_bc//2, 1), torch.ones(N_bc//2, 1)]).to(device)
    u_bc_true = torch.zeros(N_bc, 1).to(device)
    return x_bc, t_bc, u_bc_true