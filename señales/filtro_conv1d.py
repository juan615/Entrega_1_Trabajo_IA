import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

fs = 500   
t = np.linspace(0, 1, fs, endpoint=False)
clean_signal = np.sin(2*np.pi*10*t)  # señal senoidal de 10 Hz
noisy_signal = clean_signal + 0.4*np.random.randn(len(t))  


x = torch.tensor(noisy_signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

class ConvFilter(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=7, padding=3, bias=False)
        with torch.no_grad():
            self.conv.weight[:] = torch.ones_like(self.conv.weight) / 7.0

    def forward(self, x):
        return self.conv(x)

# Crear el modelo
model = ConvFilter()

y = model(x).detach().numpy().flatten()

plt.figure(figsize=(12,8))

# (a) Todas juntas
plt.subplot(4,1,1)
plt.plot(t, noisy_signal, label="Noisy", alpha=0.7)
plt.plot(t, clean_signal, label="Clean", linewidth=2)
plt.plot(t, y, label="Filtered (Conv1D)", linewidth=2)
plt.legend()
plt.title("Comparación de todas las señales")

# (b) Solo Clean
plt.subplot(4,1,2)
plt.plot(t, clean_signal, color="orange")
plt.title("Señal limpia (Clean)")

# (c) Solo Noisy
plt.subplot(4,1,3)
plt.plot(t, noisy_signal, color="blue")
plt.title("Señal con ruido (Noisy)")

# (d) Solo filtrada
plt.subplot(4,1,4)
plt.plot(t, y, color="green")
plt.title("Señal filtrada (Conv1D)")

plt.tight_layout()
plt.show()
