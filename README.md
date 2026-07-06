# Physics-Informed Neural Networks (PINNs) Portfolio 🧠⚛️

Welcome to my PINN portfolio! This repository contains professional, modular, and production-ready PyTorch implementations of Physics-Informed Neural Networks solving various fundamental partial differential equations (PDEs).

## 📂 Projects included

### 1. [1D Viscous Burgers' Equation](./PINN-Burgers-Equation) 🌊
Models the competition between non-linear convective steepening and viscous dissipation.
- **Features:** Dynamic Latin Hypercube Sampling (LHS), Adam + L-BFGS hybrid optimization.
- **Result:** Captures the formation and dissipation of a sharp shockwave.

### 2. [1D Transient Heat Transfer](./PINN-Heat-Equation) 🔥
Solves the time-dependent heat equation for temperature decay.
- **Features:** LHS spatio-temporal collocation, strictly enforced physical residuals.
- **Result:** High-fidelity heatmap prediction of thermal diffusion over time.

### 3. [Parametric 2D Shockwave Surrogate](./PINN-Parametric-2D-Wave) 💥
Models a 2D blast wave propagation using a Parametric PINN.
- **Features:** 4D LHS Sampling ($x, y, t, \rho$), Real-time inference across physical envelopes.
- **Result:** Split-screen animation comparing shockwaves in low-density vs high-density fluids.

### 4. [2D Incompressible Navier-Stokes](./PINN-Navier-Stokes-2D) 🌪️
Solves the Navier-Stokes equations for fluid dynamics using the Taylor-Green Vortex benchmark.
- **Features:** 3D Collocation ($x,y,t$), enforces Continuity and Momentum conservation.
- **Result:** Captures the accurate decay of the 2D velocity magnitude field.
---
*More physics models coming soon...*