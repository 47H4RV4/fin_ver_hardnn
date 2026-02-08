import torch
from torchvision import datasets, transforms
import random

# Setup MNIST test data
transform = transforms.Compose([transforms.ToTensor()])
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

def save_image_hex(filename, img_tensor):
    """Quantize image to uint4 (0-15) and save as hex."""
    # Scale 0.0-1.0 to 0-15
    img_q = torch.round(img_tensor * 15).int().flatten()
    with open(filename, 'w') as f:
        for val in img_q:
            # Unsigned hex conversion (0 to f)
            f.write(f"{int(val):x}\n")
    print(f"Saved {filename}")

# Select random test image
index = random.randint(0, 9999)
test_img, test_label = test_data[index]

save_image_hex("input1.mem", test_img)
print(f"Image Label: {test_label}")