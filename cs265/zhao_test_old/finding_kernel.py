import torch
from torch import nn
from torch.profiler import profile, record_function, ProfilerActivity
from triton.testing import do_bench
import torch._dynamo as dynamo
import os, sys
import numpy as np
from pprint import pprint

from torch.utils.flop_counter import operator_set

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

## Testing mult dimensions
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

N = 800
K = 100
in_channel, out_channel = 3, 32
window1 = window2 = 3

A = torch.randn(1, K, device=device)
B = torch.randn(1, N**2, device=device)
C = torch.randn(N**2, K, device=device)
D = torch.randn(1, N, N, device=device) + 2
E = torch.randn(in_channel, N, N, device=device)

## Initialize a simple model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.lin1 = nn.Linear(N*N, K)

    def forward(self, x):
        return self.lin1(self.flatten(x))


## Initialize a simple model
class AddNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu=nn.ReLU()
        self.lin1 = nn.Linear(N*N, K)
        
    def forward(self, x):
        x = torch.add(x, 265)
        x = torch.add(x, -277)
        x = torch.add(x, 2)
        # x = self.relu(x)
        # x = self.relu(x)
        # x = self.relu(x)
        return x


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, (window1, window2))
        
    def forward(self, x):
        return self.conv1(x)

@dynamo.optimize()
def adds(x):
    relu=nn.ReLU()
    x = torch.add(x, 265)
    x = torch.add(x, -277)
    x = torch.add(x, 2)
    # x = relu(x)
    # x = relu(x)
    # x = relu(x)
    # x = relu(x)
    return x
    # return nn.Linear(N*N, K, device=device)(torch.flatten(x, start_dim=1))

@dynamo.optimize()
def mult(a, b, c):
    return torch.addmm(a, b, c)

@dynamo.optimize()
def linear(x):
    return nn.Linear(N*N, K, device=device)(torch.flatten(x, start_dim=1))

@dynamo.optimize()
def conv2d(x):
    return nn.Conv2d(in_channel, out_channel, (window1, window2), device=device)(x)

def main():

    # model = AddNN().to(device)
    # model = NeuralNetwork().to(device)
    # opt_model = dynamo.optimize("inductor", nopython=True)(model)
    # print("Optimized model", opt_model)

    # # with HiddenPrints():
    # with profile(
    #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #     profile_memory=True, 
    #     record_shapes=True,
    #     with_stack=True,
    # ) as prof:
    #     res = opt_model(D)
    #     # print(f"Res: {res}")

    # print("\nNN results:-----------------------")
    # print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total", row_limit=5))
    # pprint(operator_set)


    model = CNN().to(device)
    opt_model = dynamo.optimize("inductor", nopython=True)(model)
    print("Optimized model", opt_model)

    # with HiddenPrints():
    with profile(
        activities=[ProfilerActivity.CUDA],
        profile_memory=True, 
        record_shapes=True,
        with_stack=True,
    ) as prof:
        res = opt_model(E)

    print("\nCNN results:-----------------------")
    print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total", row_limit=10))

    # with HiddenPrints():    
    with profile(
        activities=[ProfilerActivity.CUDA],
        profile_memory=True, 
        record_shapes=True,
        with_stack=True,
    ) as prof:
        # res = mult(A, B, C)
        # res = linear(D) #; res = mult(A, B, C)
        res = conv2d(E)
        # res = adds(D)
        # print(f"Res: {res}")

    print("\nFunction results:-----------------------")
    print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total", row_limit=5))

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # times = []
    # for _ in range(20):
    #     start.record()
    #     # res = mult(A, B, C)
    #     res = adds(D)
    #     end.record()

    #     # Waits for everything to finish running
    #     torch.cuda.synchronize()
    #     times.append(start.elapsed_time(end))

    # print(times)
    # print(np.mean(times[5:]))
    # print(start.elapsed_time(end))  # milliseconds

if __name__ == "__main__":
    main()



# A = torch.randn(N, N, device="cuda", dtype=torch.float16)
# B = torch.randn(N, N, device="cuda", dtype=torch.float16)
# print("randn", do_bench(lambda: torch.mm(A, B))[0])

# A = torch.rand(N, N, device="cuda", dtype=torch.float16)
# B = torch.rand(N, N, device="cuda", dtype=torch.float16)
# print("rand", do_bench(lambda: torch.mm(A, B))[0])

# A = torch.randint(0, 7, (N, N), device="cuda", dtype=torch.float16)
# B = torch.randint(0, 7, (N, N), device="cuda", dtype=torch.float16)
# print("randint", do_bench(lambda: torch.mm(A, B))[0])

# A = torch.randint(0, 7, (N, N), device="cuda", dtype=torch.float16)
# B = torch.randint(0, 7, (N, N), device="cuda", dtype=torch.float16)
# A[: N // 2] = 0
# B[: N // 2] = 0
# print("randint/zeros", do_bench(lambda: torch.mm(A, B))[0])

