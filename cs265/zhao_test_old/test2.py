import torch
from torch import nn
import numpy as np
# import os
# import csv

device = "gpu"
assert(device=="gpu")

# file_name = os.getcwd() + "/benchmark.csv"
# print(file_name)

# f = open(file_name, 'w')
# writer = csv.writer(f)

# writer.writerow([1, 2, 3])

# f.close()
# x = torch.zeros(3, 28, 28)

# res = torch.nn.Conv2d(3, 32, (3, 3))(x)

# res = torch.flatten(torch.zeros(1, 28, 28), start_dim=1)
# print(res.shape)
# print(False + 1)
# print(True + 1)
# t = torch.empty(3, 4, 5)
# t_size = t.size()

device = "cuda" if torch.cuda.is_available() else "cpu"
assert(device == "cuda")

operations = [nn.ReLU(), torch.add, torch.mul, torch.rsqrt]

N, M = 800, 800
A = torch.randn(N, M, device=device) + 2

# for f in operations:
#     print(f(A))