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
file_name = os.getcwd() + "/addmm.csv"
f = open(file_name, 'a')
writer = csv.writer(f)

# start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

def main():

    torch.cuda.empty_cache()

    for i in range(1, 10):
        for j in range(1, 10):
            for k in range(1, 10):
                
                N, M, K = i * 100, j * 100, k * 100 #; N, M, K = 2**i, 2**j, 2**k

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

                ## Built-in profiler.
                with HiddenPrints():
                    with profile(
                        activities=[ProfilerActivity.CUDA],
                        profile_memory=True, 
                        record_shapes=True,
                        with_stack=True,
                    ) as prof:
                        res = opt_model(A)

                res = prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total", row_limit=0)

                print(f"{N}, {M}, {K}, {res}")

                # Input(0), input(1), output, FLOPs, GPU Time (ms)
                writer.writerow([N, M, K, N * M * K * 2, res])

                ## Clear some memory
                del model
                del opt_model
                del A
                dynamo.reset()
                torch.cuda.empty_cache()

                # break
            # break
            f.flush()
        # break

    f.close()

if __name__ == "__main__":
    main()
