# Parametric PINN: 2D Shockwave Surrogate Model 💥

This repository contains an advanced implementation of a Parametric Physics-Informed Neural Network (PINN). Instead of solving a single instance of a PDE, this model acts as a **real-time Surrogate Solver** for a 2D explosion (blast wave propagation) by taking the fluid density ($\rho$) as an explicit input feature.

## 📌 Problem Formulation
We model acoustic shock propagation using the 2D Wave Equation (linearized Euler):

$$ \frac{\partial^2 p}{\partial t^2} = c^2 \left( \frac{\partial^2 p}{\partial x^2} + \frac{\partial^2 p}{\partial y^2} \right) $$

Where the wave speed $c$ depends on the medium's density $\rho$: $c^2 = \frac{1}{\rho}$.
- **Domain:** $x, y \in [-1, 1]$, $t \in [0, 1]$
- **Parametric Envelope:** $\rho \in [1.0, 4.0]$
- **Initial Condition (Explosion):** $p(x,y,0) = e^{-\frac{x^2+y^2}{0.02}}$ (Gaussian pressure peak)
- **Initial Velocity:** $p_t(x,y,0) = 0$
- **Boundary Conditions:** $p(x,y,t) = 0$ at domain edges.

## 🚀 Features
- **Surrogate Modeling:** The network predicts the wave propagation for *any* fluid density within the trained envelope instantly.
- **4D Latin Hypercube Sampling (LHS):** Dynamic resampling across space ($x,y$), time ($t$), and parameters ($\rho$) to prevent overfitting.
- **Hybrid Optimization:** Adam + L-BFGS for rigorous enforcement of 2nd-order PDE residuals.

## 🛠️ Usage
Install dependencies and run the training:

    pip install -r requirements.txt
    python main.py

## 🎥 Results
The model generates a split-screen animation comparing shockwave propagation in low-density (fast) vs high-density (slow) fluids in real-time.

![Parametric 2D Explosion](parametric_2d_explosion.gif)