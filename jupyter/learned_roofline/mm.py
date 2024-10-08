import pandas as pd
import os, sys
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../..')))

from utils.prediction_utils import *

import matplotlib.pyplot as plt
import seaborn as sns

random_seed = 42

# base_dir = "/Users/andrew/Desktop/Harvard/idreos-research/gpu_profiling"
base_dir = "/n/holylabs/LABS/idreos_lab/Users/azhao/gpu_profiling"
X, y = get_data("mm", base_dir, sample_rate=0.2)

# Saving the result somewhere in case.
df = pd.concat([X, y], axis=1)
df.info()

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
def custom_loss_function(predicted_alpha, target_operator_time, memory_accesses, intensity, pi, beta):
    """
    predicted_alpha: Output from the neural network (predicted alpha)
    target_operator_time: Actual operator times (ground truth)
    memory_accesses: Precomputed memory accesses for each sample
    intensity: Precomputed arithmetic intensity for each sample
    pi: Precomputed dtype-specific peak FLOPs/sec (from dtype_to_peak_fp)
    beta: DRAM bandwidth
    """
    estimated_operator_time = torch.max(memory_accesses / pi, memory_accesses / (predicted_alpha * beta * intensity))
    loss = nn.functional.mse_loss(estimated_operator_time, target_operator_time)
    return loss

X = X.astype({"dtype_16": int, "dtype_32": int, "dtype_b16": int})
X = torch.tensor(X.values, dtype=torch.float32)
y = torch.tensor(y)

def get_dtype_bytes(row):
    if row['dtype_32']:
        return 4
    elif row['dtype_16']:
        return 2
    elif row['dtype_b16']:
        return 2
    else:
        raise ValueError("Unknown dtype in row.")

memory_accesses = (df["n"] * df["m"] + df["m"] * df["p"] + df["n"] * df["p"]) * df.apply(get_dtype_bytes, axis=1)
intensity = (df["gflops"] * 1e9) / memory_accesses

memory_accesses = torch.tensor(memory_accesses.values, dtype=torch.float32)
intensity = torch.tensor(intensity.values, dtype=torch.float32)

def get_dtype_peak_fp(row):
    # 156 for tf32.
    if row['dtype_32']:
        return 19.5
    elif row['dtype_16']:
        return 312
    elif row['dtype_b16']:
        return 312
    else:
        raise ValueError("Unknown dtype in row.")

# Note: hard-coded for A100.
beta = 2.03904

pi = df.apply(get_dtype_peak_fp, axis=1)
pi = torch.tensor(pi.values, dtype=torch.float32)

# Initialize the model
input_size = 7  # n, m, p, gflops, dtype_16, dtype_32, dtype_b16
hidden_size = 64
output_size = 1  # Predicting alpha or operator time
model = Net(input_size, hidden_size, output_size)

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    
    # Forward pass: predict alpha
    predicted_alpha = model(X)
    
    # Compute the custom loss based on the predicted alpha
    loss = custom_loss_function(predicted_alpha, y, memory_accesses, intensity, pi, beta)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Print loss every 10 epochs
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')