import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import os
import random

# --- 1. Data Setup ---
transform = transforms.Compose([transforms.ToTensor()])
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

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


# C. Export Test Image
# Get one image from test set
test_img, test_label = test_data[random.randint(0,len(test_data)-1)]
img_q = torch.round(test_img * 15).int()
save_mem_file("input1.mem", img_q)

print(f"\nExpected Label for input.mem: {test_label}")
