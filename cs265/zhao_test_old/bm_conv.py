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
file_name = os.getcwd() + "/conv2d.csv"


start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

# @dynamo.optimize()
# def conv2d(x):
#     return nn.Conv2d(in_channel, out_channel, (window, window), device=device)(x)


def main():

    torch.cuda.empty_cache()

    with open(file_name, 'a') as f: # w to overwrite, a to append.
        writer = csv.writer(f)

        for i in range(1, 10): 
            for j in range(1, 10): 
                for k in range(1, 10):
                    for l in range(1, 10):
                        for window in range(1, 8, 2): # Kernel size. Note that we only allow square kernels of odd lengths.

                            # N, M, in_channel, out_channel = 2**i, 2**i, 2**k, 2**l

                            N, M, in_channel, out_channel = i * 100, j * 100, k * 100, l * 100

                            print(N, M, in_channel, out_channel, window)

                            # N, M, in_channel, out_channel, window = 800, 800, 50, 90, 3

                            if N <= window or M <= window:
                                continue

                            A = torch.randn(in_channel, N, M, device=device) + 2

                            class CNN(nn.Module):
                                def __init__(self):
                                    super().__init__()
                                    self.conv1 = nn.Conv2d(in_channel, out_channel, (window, window)) ## Only test windows of the same size.
                                    
                                def forward(self, x):
                                    return self.conv1(x)

                            model = CNN().to(device)
                            opt_model = dynamo.optimize("inductor", nopython=True)(model)

                            ## Testing method 1: built-in profiler.
                            with HiddenPrints():
                                with profile(
                                    activities=[ProfilerActivity.CUDA],
                                    profile_memory=True, 
                                    record_shapes=True,
                                    with_stack=True,
                                ) as prof:
                                    res = opt_model(A)


                            # torch.cuda.synchronize() ## This is unnecessary
                            res = prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total", row_limit=0)
                            # print(f"{N}, {M}, {in_channel}, {out_channel}, {window}, {res}")

                            # Input(0), input(1), output, FLOPs, GPU Time (ms)
                            ## Hmm: need to calculate flops here.
                            flops = 2 * (N - window + 1) * (M - window + 1) * in_channel * out_channel * (window ** 2) 
                            writer.writerow([N, M, in_channel, out_channel, window, flops, res])

                            ## Clear some memory
                            del model
                            del opt_model
                            del A
                            dynamo.reset()
                            torch.cuda.empty_cache()

                            # break
                        # break
                    # break
                # break
                f.flush()
            # break

if __name__ == "__main__":
    main()
