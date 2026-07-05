# Physics-Informed Neural Networks (PINN) for 1D Burgers' Equation 🌊

This repository contains a PyTorch implementation of a Physics-Informed Neural Network (PINN) to solve the 1D viscous Burgers' equation, a standard mathematical benchmark for fluid dynamics and non-linear PDEs.

## 📌 Problem Formulation
The equation models the competition between non-linear convective steepening and viscous dissipation:

$$ \frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2} $$

- **Kinematic Viscosity ($\nu$):** $0.01 / \pi$ (kept low to study shockwave formation).
- **Initial Condition (IC):** $u(x,0) = -\sin(\pi x)$
- **Boundary Conditions (BC):** $u(-1,t) = u(1,t) = 0$

## 🚀 Features
- **Dynamic PDE Sampling:** Collocation points are resampled at every epoch to prevent spatial overfitting (using Latin Hypercube Sampling).
- **Hybrid Optimization:** Uses **Adam** for initial convergence and **L-BFGS** for high-precision fine-tuning.
- **Modular Architecture:** Codebase is structured to separate physics formulation, data generation, and the training loop.

## 🛠️ Installation & Usage

1. Clone the repository:
git clone https://github.com/IL_TUO_NOME_UTENTE/PINN-Burgers-Equation.git
cd PINN-Burgers-Equation

2. Install dependencies:
pip install -r requirements.txt

3. Run the training and generate the animation:
python main.py

## 🎥 Results
After training, the model outputs an animated GIF showing the shockwave formation and dissipation:

![Burgers Equation Shockwave](burgers_shock_wave.gif)