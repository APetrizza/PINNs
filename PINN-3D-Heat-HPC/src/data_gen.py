import torch
import numpy as np
from scipy.stats import qmc
from typing import Tuple

def get_collocation_points(N_pde: int, alpha_min: float, alpha_max: float, device: torch.device) -> Tuple[torch.Tensor, ...]:
    """5D LHS Sampling: x, y, z, t, alpha."""
    sampler = qmc.LatinHypercube(d=5)
    sample = sampler.random(n=N_pde)
    
    x = torch.tensor(sample[:, 0] * 2 - 1, dtype=torch.float32).view(-1, 1).to(device)
    y = torch.tensor(sample[:, 1] * 2 - 1, dtype=torch.float32).view(-1, 1).to(device)
    z = torch.tensor(sample[:, 2] * 2 - 1, dtype=torch.float32).view(-1, 1).to(device)
    t = torch.tensor(sample[:, 3], dtype=torch.float32).view(-1, 1).to(device)
    alpha = torch.tensor(sample[:, 4] * (alpha_max - alpha_min) + alpha_min, dtype=torch.float32).view(-1, 1).to(device)
    
    return x, y, z, t, alpha

def get_initial_conditions(N_ic: int, alpha_min: float, alpha_max: float, device: torch.device) -> Tuple[torch.Tensor, ...]:
    """Generates ICs: 3D Gaussian Hotspot at center."""
    x_ic = (torch.rand(N_ic, 1) * 2 - 1).to(device)
    y_ic = (torch.rand(N_ic, 1) * 2 - 1).to(device)
    z_ic = (torch.rand(N_ic, 1) * 2 - 1).to(device)
    t_ic = torch.zeros(N_ic, 1).to(device)
    alpha_ic = (torch.rand(N_ic, 1) * (alpha_max - alpha_min) + alpha_min).to(device)
    
    r_squared = x_ic**2 + y_ic**2 + z_ic**2
    u_ic_true = torch.exp(-r_squared / 0.05).to(device)
    
    return x_ic, y_ic, z_ic, t_ic, alpha_ic, u_ic_true

def get_boundary_conditions(N_bc: int, alpha_min: float, alpha_max: float, device: torch.device) -> Tuple[torch.Tensor, ...]:
    """Generates BCs on the 6 faces of the 3D spatial cube (x,y,z in [-1, 1])."""
    pts_per_face = N_bc // 6
    x_bc, y_bc, z_bc = [], [], []
    
    # Faccia 1: x = -1
    x_bc.append(-torch.ones(pts_per_face, 1))
    y_bc.append(torch.rand(pts_per_face, 1) * 2 - 1)
    z_bc.append(torch.rand(pts_per_face, 1) * 2 - 1)
    # Faccia 2: x = 1
    x_bc.append(torch.ones(pts_per_face, 1))
    y_bc.append(torch.rand(pts_per_face, 1) * 2 - 1)
    z_bc.append(torch.rand(pts_per_face, 1) * 2 - 1)
    # Faccia 3: y = -1
    x_bc.append(torch.rand(pts_per_face, 1) * 2 - 1)
    y_bc.append(-torch.ones(pts_per_face, 1))
    z_bc.append(torch.rand(pts_per_face, 1) * 2 - 1)
    # Faccia 4: y = 1
    x_bc.append(torch.rand(pts_per_face, 1) * 2 - 1)
    y_bc.append(torch.ones(pts_per_face, 1))
    z_bc.append(torch.rand(pts_per_face, 1) * 2 - 1)
    # Faccia 5: z = -1
    x_bc.append(torch.rand(pts_per_face, 1) * 2 - 1)
    y_bc.append(torch.rand(pts_per_face, 1) * 2 - 1)
    z_bc.append(-torch.ones(pts_per_face, 1))
    # Faccia 6: z = 1
    x_bc.append(torch.rand(pts_per_face, 1) * 2 - 1)
    y_bc.append(torch.rand(pts_per_face, 1) * 2 - 1)
    z_bc.append(torch.ones(pts_per_face, 1))

    # Concatena tutte le facce
    x_bc = torch.cat(x_bc).to(device)
    y_bc = torch.cat(y_bc).to(device)
    z_bc = torch.cat(z_bc).to(device)
    
    # Tempi e Parametri alpha casuali per le boundaries
    N_total_bc = pts_per_face * 6
    t_bc = torch.rand(N_total_bc, 1).to(device)
    alpha_bc = (torch.rand(N_total_bc, 1) * (alpha_max - alpha_min) + alpha_min).to(device)
    
    # Valore vero sui bordi: Temperatura 0
    u_bc_true = torch.zeros(N_total_bc, 1).to(device)
    
    return x_bc, y_bc, z_bc, t_bc, alpha_bc, u_bc_true