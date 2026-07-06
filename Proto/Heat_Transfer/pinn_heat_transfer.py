import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Fissa il seed per riproducibilità
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 1. DEFINIZIONE DELLA RETE NEURALE (MLP)
# ==========================================
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        # Input: [x, t] (2 features)
        # Output: [u] (Temperatura)
        # Usiamo Tanh perché è infinitamente derivabile (fondamentale per le derivate seconde della PDE)
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        return self.net(inputs)

# ==========================================
# 2. DEFINIZIONE DELLA FISICA E DELLA LOSS
# ==========================================
def pde_loss(model, x, t, alpha=0.01):
    # Abilita il calcolo del gradiente rispetto agli input
    x.requires_grad_(True)
    t.requires_grad_(True)
    
    # Previsione della rete neurale u(x,t)
    u = model(x, t)
    
    # Derivate prime
    # torch.autograd.grad calcola la derivata dell'output (u) rispetto all'input (t o x)
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    
    # Derivata seconda spaziale
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    
    # Residuo fisico dell'equazione del calore: f = u_t - alpha * u_xx (deve essere 0)
    pde_residual = u_t - alpha * u_xx
    
    return torch.mean(pde_residual**2)

# ==========================================
# 3. GENERAZIONE DEI DATI DI COLLOCAZIONE
# ==========================================
N_pde = 5000  # Punti interni al dominio (per la PDE)
N_bc = 500    # Punti sui bordi (Boundary Conditions)
N_ic = 500    # Punti al tempo zero (Initial Conditions)

# Punti interni (x tra -1 e 1, t tra 0 e 1)
x_pde = torch.empty(N_pde, 1).uniform_(-1, 1)
t_pde = torch.empty(N_pde, 1).uniform_(0, 1)

# Condizioni Iniziali (IC): t=0. Temperatura iniziale ad es. una sinusoide: u(x,0) = -sin(pi*x)
x_ic = torch.empty(N_ic, 1).uniform_(-1, 1)
t_ic = torch.zeros(N_ic, 1)
u_ic_true = -torch.sin(np.pi * x_ic)

# Condizioni al Contorno (BC): x = -1 e x = 1. Teniamo i bordi a temperatura fissa u=0.
t_bc = torch.empty(N_bc, 1).uniform_(0, 1)
x_bc_left = -torch.ones(N_bc // 2, 1)
x_bc_right = torch.ones(N_bc // 2, 1)
x_bc = torch.cat([x_bc_left, x_bc_right], dim=0)
u_bc_true = torch.zeros(N_bc, 1)

# ==========================================
# 4. LOOP DI ADDESTRAMENTO
# ==========================================
model = PINN()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 3000
for epoch in range(epochs):
    optimizer.zero_grad()
    
    # 1. Loss della PDE (sui punti interni)
    loss_pde = pde_loss(model, x_pde, t_pde)
    
    # 2. Loss Condizioni Iniziali (IC)
    u_ic_pred = model(x_ic, t_ic)
    loss_ic = torch.mean((u_ic_pred - u_ic_true)**2)
    
    # 3. Loss Condizioni al Contorno (BC)
    u_bc_pred = model(x_bc, t_bc)
    loss_bc = torch.mean((u_bc_pred - u_bc_true)**2)
    
    # TOTAL LOSS
    loss = loss_pde + loss_ic + loss_bc
    loss.backward()
    optimizer.step()
    
    if epoch % 500 == 0:
        print(f"Epoch {epoch:4d} | Total Loss: {loss.item():.5f} (PDE: {loss_pde.item():.5f}, IC: {loss_ic.item():.5f}, BC: {loss_bc.item():.5f})")

print("Training completato!")

# ==========================================
# 5. VISUALIZZAZIONE RISULTATI
# ==========================================
x_test = torch.linspace(-1, 1, 100).view(-1, 1)
t_test = torch.linspace(0, 1, 100).view(-1, 1)
X, T = torch.meshgrid(x_test.squeeze(), t_test.squeeze(), indexing='ij')

X_flat = X.reshape(-1, 1)
T_flat = T.reshape(-1, 1)

with torch.no_grad():
    U_pred = model(X_flat, T_flat).reshape(100, 100)

plt.figure(figsize=(8, 6))
plt.contourf(T, X, U_pred, 50, cmap='inferno')
plt.colorbar(label='Temperatura $u(x,t)$')
plt.xlabel('Tempo $t$')
plt.ylabel('Spazio $x$')
plt.title('PINN: Predizione Trasferimento di Calore 1D Transitorio')
plt.savefig('pinn_heat_transfer.png')
plt.show()