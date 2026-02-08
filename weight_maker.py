import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

# --- 1. Architecture: 784 -> 128 -> 32 -> 10 (bias=False) ---
class ParallelMLP(nn.Module):
    def __init__(self):
        super(ParallelMLP, self).__init__()
        # bias=False matches the hardware MAC simplicity
        self.fc1 = nn.Linear(784, 128, bias=False)
        self.fc2 = nn.Linear(128, 32, bias=False)
        self.fc3 = nn.Linear(32, 10, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# --- 2. Quantization Helpers ---
def quantize_tensor(tensor, min_val=-8, max_val=7):
    abs_max = torch.max(torch.abs(tensor))
    if abs_max == 0: return tensor.int()
    scale = abs_max / max(abs(min_val), abs(max_val))
    tensor_q = torch.round(tensor / scale)
    return torch.clamp(tensor_q, min_val, max_val).int()

def to_hex(val, bits=4):
    """Converts signed integer to hex using 2's complement."""
    val = int(val)
    if val < 0: val = (1 << bits) + val
    return f"{val:x}"

def save_giant_mem(filename, tensor):
    """Flatten weights and save as a giant hex .mem file."""
    data_flat = tensor.detach().numpy().flatten()
    with open(filename, 'w') as f:
        for val in data_flat:
            f.write(f"{to_hex(val)}\n")
    print(f"Saved {filename}")

# --- 3. Training Logic (Simplified) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ParallelMLP().to(device)
# [Insert Standard Training Loop Here: 6 Epochs, Adam LR=0.001]

# --- 4. Export Quantized Weights ---
print("\n--- Starting Quantization and Export ---")
model.cpu()
w1_q = quantize_tensor(model.fc1.weight)
w2_q = quantize_tensor(model.fc2.weight)
w3_q = quantize_tensor(model.fc3.weight)

save_giant_mem("w1.mem", w1_q)
save_giant_mem("w2.mem", w2_q)
save_giant_mem("w3.mem", w3_q)