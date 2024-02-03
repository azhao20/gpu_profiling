import torch
from torch import nn
from torch.profiler import profile, record_function, ProfilerActivity
import torch._dynamo as dynamo
import os, sys, csv

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

## Testing mult dimensions
device = "cuda" if torch.cuda.is_available() else "cpu"
assert(device == "cuda")

## Write the result
file_name = os.getcwd() + "/benchmark.csv"
f = open(file_name, 'w')
writer = csv.writer(f)

# N = 100
# K = 1000
 
# in_channel, out_channel = 3, 32
# window1 = window2 = 3

# A = torch.randn(1, N, N, device=device) + 2
# B = torch.randn(in_channel, N, N, device=device)

# ## Initialize a simple model
# class AddMM(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.flatten = nn.Flatten()
#         self.lin1 = nn.Linear(N*N, K)

#     def forward(self, x):
#         return self.lin1(self.flatten(x))

# class CNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_channel, out_channel, (window1, window2))
        
#     def forward(self, x):
#         return self.conv1(x)

def main():

    for i in range(15):
        for j in range(15):
            for k in range(15):
                
                N, M, K = 2**i, 2**j, 2**k
                A = torch.randn(1, N, M, device=device) + 2

                class AddMM(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.flatten = nn.Flatten()
                        self.lin1 = nn.Linear(N*M, K)

                    def forward(self, x):
                        return self.lin1(self.flatten(x))
                
                model = AddMM().to(device)
                opt_model = dynamo.optimize("inductor", nopython=True)(model)

                with HiddenPrints():
                    with profile(
                        activities=[ProfilerActivity.CUDA],
                        profile_memory=True, 
                        record_shapes=True,
                        with_stack=True,
                    ) as prof:
                        res = opt_model(A)
                res = prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total", row_limit=0)
                # print(f"{N}, {M}, {K}, {res}")

                # Input(0), input(1), output, FLOPs, GPU Time (ms)
                writer.writerow([N, M, K, N * M * K * 2, res])

                ## Clear some memory
                torch.cuda.synchronize()
                del model
                del opt_model
                del A
                dynamo.reset()


    f.close()

    # model = AddMM().to(device)
    # opt_model = dynamo.optimize("inductor", nopython=True)(model)
    # print("Optimized model", opt_model)

    # for N in range()
    # N = 100
    # K = 1000
    # M = 2 ... # Try powers of two.

    # for N in range(asdkfjaslkdf):
    #     for K in range(100):
            
            # INSERT: torch.cuda.synchronize.

    # with HiddenPrints():
    #     with profile(
    #         activities=[ProfilerActivity.CUDA],
    #         profile_memory=True, 
    #         record_shapes=True,
    #         with_stack=True,
    #     ) as prof:
    #         res = opt_model(A)

    # print("\nNN results:-----------------------")
    # print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total", row_limit=0))
    # print(f"{N}, {K}")

    # res = opt_model(A)

if __name__ == "__main__":
    main()
