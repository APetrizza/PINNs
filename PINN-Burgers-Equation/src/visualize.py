import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def create_and_save_animation(model: torch.nn.Module, device: torch.device, filename: str = "burgers_shock_wave.gif"):
    print("\nGenerazione dell'animazione in corso...")
    
    x_test = torch.linspace(-1, 1, 200).view(-1, 1).to(device)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    line, = ax.plot([], [], 'r-', lw=3, label=r'PINN Prediction $u(x,t)$')
    
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel('Spazio (x)')
    ax.set_ylabel('Velocità (u)')
    ax.axhline(0, color='gray', linestyle='--')
    ax.axvline(0, color='gray', linestyle=':')
    ax.legend(loc='upper right')
    ax.grid(True)
    
    time_text = ax.text(-0.9, 0.9, '', fontsize=12, fontweight='bold')
    
    def update(frame):
        t_val = frame / 40.0
        t_test = (torch.ones_like(x_test) * t_val).to(device)
        
        with torch.no_grad():
            u_pred = model(x_test, t_test).cpu().numpy()
            
        line.set_data(x_test.cpu().numpy(), u_pred)
        time_text.set_text(f'Tempo (t) = {t_val:.2f} s')
        
        if 0.25 < t_val < 0.45:
            time_text.set_color('red')
            ax.set_title("Formazione dell'Onda d'Urto (Shock steepening)!")
        else:
            time_text.set_color('black')
            ax.set_title("Evoluzione dell'Equazione di Burgers")
            
        return line, time_text

    ani = animation.FuncAnimation(fig, update, frames=40, interval=100, blit=True)
    ani.save(filename, writer='pillow')
    print(f"Animazione salvata con successo: {filename}")
    plt.close()