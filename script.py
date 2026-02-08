import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import os
import random
# --- Configuration ---
BATCH_SIZE = 64
EPOCHS = 6
LEARNING_RATE = 0.001
HIDDEN_1 = 64
HIDDEN_2 = 32

# --- 1. Data Setup ---
transform = transforms.Compose([transforms.ToTensor()])
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# --- 2. Define the Model (Hardware Optimized) ---
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        # bias=False ensures PyTorch learns to work without biases,
        # exactly matching our simple Verilog MAC unit.
        self.fc1 = nn.Linear(28*28, HIDDEN_1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(HIDDEN_1, HIDDEN_2, bias=False)
        self.fc3 = nn.Linear(HIDDEN_2, 10, bias=False)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = SimpleMLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- 3. Training Loop ---
print(f"Training for {EPOCHS} epochs (bias=False)...")
for epoch in range(EPOCHS):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{EPOCHS} complete. Loss: {loss.item():.4f}")
# --- 3.1 Model Accuracy
print("\n--- Calculating Original (Floating-Point) Model Accuracy ---")

correct = 0
total = 0

# Set model to evaluation mode (good practice, though your simple model behaves same as train)
model.eval()

with torch.no_grad(): # Turn off gradient calculation to save memory/speed
    for data, target in test_loader:
        # Pass the standard floating point data (0.0 to 1.0) through the model
        output = model(data)

        # Get the index of the max log-probability (the predicted digit)
        pred = output.argmax(dim=1, keepdim=True)

        # Check how many matched the target label
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += data.shape[0]

accuracy = 100. * correct / total
print(f"Original Model Accuracy: {accuracy:.2f}%")


# --- 4. Quantization Helper Functions ---
def quantize_tensor(tensor, min_val=-8, max_val=7):
    abs_max = torch.max(torch.abs(tensor))
    if abs_max == 0: return tensor.int(), 1.0

    scale = abs_max / max(abs(min_val), abs(max_val))
    tensor_q = torch.round(tensor / scale)
    tensor_q = torch.clamp(tensor_q, min_val, max_val)
    return tensor_q.int(), scale

def to_hex(val, bits=4):
    val = int(val)
    if val < 0: val = (1 << bits) + val
    return f"{val:0{bits//4}x}"

def save_mem_file(filename, tensor, bits=4):
    data_flat = tensor.detach().numpy().flatten()
    with open(filename, 'w') as f:
        for val in data_flat:
            f.write(f"{to_hex(val, bits)}\n")
    print(f"Saved {filename}")

# --- 5. Perform Quantization and Export ---
print("\n--- Starting Quantization ---")

# A. Quantize Weights Only (No Biases)
w1_q, s1 = quantize_tensor(model.fc1.weight, -8, 7)
w2_q, s2 = quantize_tensor(model.fc2.weight, -8, 7)
w3_q, s3 = quantize_tensor(model.fc3.weight, -8, 7)

# B. Export Weights
# Note: We do NOT transpose (.t()) here so Verilog reads memory linearly
save_mem_file("w1.mem", w1_q)
save_mem_file("w2.mem", w2_q)
save_mem_file("w3.mem", w3_q)

# C. Export Test Image
# Get one image from test set
test_img, test_label = test_data[5]
img_q = torch.round(test_img * 15).int()
save_mem_file("input1.mem", img_q)

print(f"\nExpected Label for input.mem: {test_label}")

# --- 6. Hardware Golden Reference Check ---
# We simulate the hardware math here to know what the FPGA *should* output
def hardware_simulate(x, w):
    # No bias addition here
    res = torch.matmul(x.float(), w.float())
    return res

# Flatten input
x_vec = img_q.flatten().float()

# Layer 1
# Note: .t() is required for Python math, but not for the .mem file export
z1 = torch.matmul(x_vec, w1_q.t().float())
a1 = torch.clamp(torch.floor(z1/64), 0, 15)

# Layer 2
z2 = torch.matmul(a1, w2_q.t().float())
a2 = torch.clamp(torch.floor(z2/64), 0, 15)

# Layer 3
z3 = torch.matmul(a2, w3_q.t().float())

print("\n--- Golden Reference (Integer Simulation) ---")
print("Layer 3 Output (logits):", z3.detach().numpy())
print("Predicted Digit:", torch.argmax(z3).item())

# --- 7. Calculate Full Quantized Accuracy ---
print("\n--- Calculating Quantized Model Accuracy (Hardware Simulation) ---")

correct = 0
total = 0

with torch.no_grad():
    for data, target in test_loader:
        img_q_batch = torch.round(data * 15).int()
        x_batch = img_q_batch.view(-1, 28*28).float()

        # Hardware Simulation Loop
        # Layer 1
        z1 = torch.matmul(x_batch, w1_q.t().float())
        z1 = torch.floor(z1 / 64)   # Bit shift right 6
        z1 = torch.clamp(z1, 0, 15) # ReLU

        # Layer 2
        z2 = torch.matmul(z1, w2_q.t().float())
        z2 = torch.floor(z2 / 64)
        z2 = torch.clamp(z2, 0, 15)

        # Layer 3
        z3 = torch.matmul(z2, w3_q.t().float())

        # Prediction
        pred = z3.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += data.shape[0]

print(f"Quantized Model Accuracy: {100. * correct / total:.2f}%")


TEST_IMAGE_INDEX = random.randint(0,9999)

# Get image and label
test_img, test_label = test_data[TEST_IMAGE_INDEX]
img_q = torch.round(test_img * 15).int()

# Save only the input file
save_mem_file("input1.mem", img_q)
print(f"Saved input1.mem for Image Index {TEST_IMAGE_INDEX} (True Label: {test_label})")

# --- Golden Reference Check for this specific image ---
x_vec = img_q.flatten().float()

# Layer 1
z1 = torch.matmul(x_vec, w1_q.t().float())
a1 = torch.clamp(torch.floor(z1/64), 0, 15)

# Layer 2
z2 = torch.matmul(a1, w2_q.t().float())
a2 = torch.clamp(torch.floor(z2/64), 0, 15)

# Layer 3
z3 = torch.matmul(a2, w3_q.t().float())

print("\n--- Python Golden Prediction ---")
print("Logits:", z3.detach().numpy())
print("Predicted Class:", torch.argmax(z3).item())