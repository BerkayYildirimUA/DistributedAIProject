print("1. Importing Numpy...")
import numpy
print("   Success.")

print("2. Importing Torch...")
import torch
# Trigger some math to force it to load the C++ backend
t = torch.tensor([1.0, 2.0]).cuda() if torch.cuda.is_available() else torch.tensor([1.0])
print("   Success.")

print("3. Importing CARLA...")
import carla
print("   Success.")