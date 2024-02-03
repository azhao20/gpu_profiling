import torch
from torch import nn
import torch._dynamo as dynamo
import os, sys, csv
import numpy as np

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert(device == "cuda")

    ## Write the result
    file_name = os.getcwd() + "/pw.csv"
    
    ## Generate 10 consistent numbers.
    ## Shouldn't make any difference.
    np.random.seed(265)
    random_floats = np.random.uniform(1, 265, size=(20,))

    with open(file_name, 'a') as f: # w to overwrite, a to append.
        writer = csv.writer(f)

        for i in range(1, 10):
            for j in range(1, 10):
                for K in range(1, 5):
                    
                    # N, M = 2**i, 2**j
                    # N, M = i * (10 ** 4), j * (10 ** 3)
                    N = M = 32768
                    # K = 5
                    # if N == 2 and M < 32:
                    #     continue

                    # N, M, K = 10000, 10000, 10

                    A = torch.randn(N, M, device=device) + 2

                    ## Initialize a simple model
                    class NN(nn.Module):
                        def __init__(self):
                            super().__init__()
                            self.relu=nn.ReLU()
                            
                        def forward(self, x):
                            for k in range(K): # K = 20
                                x = torch.mul(x, random_floats[k])

                            # x = torch.mul(x, random_floats[11])
                            # x = torch.rsqrt(x)
                            # x = torch.mul(x, random_floats[12])
                            # x = torch.mul(x, random_floats[0])
                            # x = torch.rsqrt(x)
                            # x = torch.add(x, random_floats[3])
                            # x = torch.rsqrt(x)
                            # x = self.relu(x)
                            # x = torch.mul(x, random_floats[1])
                            # x = torch.add(x, random_floats[4])
                            # x = torch.sub(x, random_floats[8])
                            # x = torch.add(x, random_floats[5])
                            # x = self.relu(x)
                            # x = torch.sub(x, random_floats[7])
                            # x = torch.mul(x, random_floats[2])
                            # x = torch.sub(x, random_floats[9])
                            # x = self.relu(x)
                            # x = torch.add(x, random_floats[6])
                            # x = torch.rsqrt(x)
                            # x = torch.sub(x, random_floats[10])

                            
                                # for _ in range(5):
                                # for _ in range(5):
                                #     x = torch.mul(x, 265) ## Random number.
                            # x = torch.add(x, 265) ## Random number.
                            # x = torch.rsqrt(x)
                            # x = self.relu(x)
                            # x = torch.add(x, -365)
                            # x = torch.add(x, 465)
                            return x

                    model = NN().to(device)
                    opt_model = dynamo.optimize("inductor", nopython=True)(model)
                    # with HiddenPrints():
                    res = opt_model(A)

                    torch.cuda.synchronize()
                    # assert(torch._inductor.triton_heuristics.runtime_ms is not None)
                    # print(f"{K}, {N}, {M}, {2 * N * M}, {torch._inductor.triton_heuristics.runtime_ms}")
                    # Number of fused, input(0), input(1), flops, GPU time
                    print(N, M, K)
                    # writer.writerow([K, N, M, 2 * N * M])

                    ## Clear some memory
                    del model
                    del opt_model
                    del A
                    dynamo.reset()
                    torch.cuda.empty_cache()

                    break
                break
                f.flush()
            break


if __name__ == "__main__":
    main()
