# 3D Parametric PINN for Heat Transfer (HPC Ready) ☄️

This repository contains an enterprise-grade, High-Performance Computing (HPC) ready implementation of a Physics-Informed Neural Network. It solves the **3D Transient Heat Equation** while acting as a parametric surrogate model for the thermal diffusivity ($\alpha$).

## 📌 Problem Formulation
We model the diffusion of a 3D spherical heat source (hotspot) over time:

$$ \frac{\partial u}{\partial t} = \alpha \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} + \frac{\partial^2 u}{\partial z^2} \right) $$

- **5D Spatio-Temporal-Parametric Domain:** $x,y,z \in [-1, 1]$, $t \in [0, 1]$, $\alpha \in [0.01, 0.1]$
- **Initial Condition (IC):** $u(x,y,z,0) = e^{-\frac{x^2+y^2+z^2}{0.05}}$ (Central 3D Gaussian Hotspot)
- **Assumption:** Infinite domain approximation (diffusion observed before boundary interactions).

## 🚀 Features
- **HPC Multi-GPU Support:** Uses `torch.nn.DataParallel` to scale training across cluster nodes.
- **5D Latin Hypercube Sampling (LHS):** Efficient sampling of $50,000$ collocation points across 5 dimensions.
- **Decoupled Workflow:** Training script saves model weights (`.pth`), and a separate inference module loads them (handling the `module.` state_dict prefix) to generate visualizations.

## 🛠️ Usage
Install dependencies:

    pip install -r requirements.txt

Run the orchestration script (it will automatically train if weights are missing, or just generate the GIF if weights exist):

    python main.py

## 🎥 Results
The inference module slices the 3D domain at the $Z=0$ plane, generating a split-screen animation that compares heat diffusion in a low-diffusivity material (Ceramic, $\alpha=0.01$) vs a high-diffusivity material (Metal, $\alpha=0.1$).

![3D Heat Transfer Surrogate](pinn_3d_heat_transfer.gif)