# Parametric PINN: 2D Navier-Stokes Oseen-Lamb Vortex 🌪️

This repository contains an HPC-ready, advanced Physics-Informed Neural Network (PINN). It acts as a **Parametric Surrogate Model** to solve the 2D Incompressible Navier-Stokes equations for an Oseen-Lamb vortex immersed in a uniform freestream.

## 📌 Problem Formulation
We simulate the fluid dynamics of a decaying vortex moving with a background wind. The network learns the physics across a continuous spectrum of vortex radii ($R$) and freestream velocities ($U_0$).

**1. Continuity & Navier-Stokes Equations:**
$$ \nabla \cdot \mathbf{u} = 0 $$
$$ \frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} = -\frac{1}{\rho}\nabla p + \nu \nabla^2 \mathbf{u} $$

- **Domain:** $x,y \in [-2, 2]$, $t \in [0, 1]$
- **Parametric Envelope:** $R \in [0.1, 0.5]$ (Vortex Radius), $U_0 \in [1.0, 10.0]$ m/s (Wind Speed)
- **Fluid Properties:** $\rho = 1.225$ kg/m³ (Air density), $\nu = 1.5 \times 10^{-5}$ m²/s (Kinematic viscosity)
- **Initial Condition:** Exact Oseen-Lamb velocity profile parameterized by $R$ and $U_0$.

## 🚀 Features
- **5D Latin Hypercube Sampling (LHS):** Efficient sampling across $x, y, t, R, U_0$ space ($60,000$ points).
- **HPC Multi-GPU Scalability:** Built-in `torch.nn.DataParallel` support for cluster training, with safe state-dict extraction.
- **Autograd Vorticity Computation:** The post-processing script computes the vorticity field $\omega = \frac{\partial v}{\partial x} - \frac{\partial u}{\partial y}$ exactly using PyTorch's automatic differentiation (`autograd`), entirely avoiding numerical grid approximations.

## 🛠️ Usage
Install dependencies and run the orchestrator (it will train if weights are missing, or directly render the animation if `.pth` exists):

    pip install -r requirements.txt
    python main.py

## 🎥 Results
Split-screen animation comparing the exact vorticity field evolution for two different parametric scenarios (Small vortex/Slow wind vs Large vortex/Fast wind).

![Oseen-Lamb Vortex Evolution](vortex_evolution.gif)