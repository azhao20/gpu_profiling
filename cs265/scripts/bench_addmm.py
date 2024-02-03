import os, sys, csv
import torch
from torch import nn
import torch._dynamo as dynamo

## Block prints
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def main():
    device = f"cuda:{sys.argv[1]}" if torch.cuda.is_available() else "cpu"
    assert(device == f"cuda:{sys.argv[1]}")

    ## Write the result
    file_name = os.getcwd() + f"/zhao_data/addmm{sys.argv[1]}.csv"

    if int(sys.argv[1]) == 0:
        num_inputs = list(range(1, 5))
        biases = [False]
    elif int(sys.argv[1]) == 1:
        num_inputs = list(range(5, 8))
        biases = [False]
    elif int(sys.argv[1]) == 2:
        num_inputs = list(range(7, 9))
        biases = [True]
    elif int(sys.argv[1]) == 3:
        num_inputs = list(range(9, 11))
        biases = [True]

    # num_inputs = list(range(1, 7)) if int(sys.argv[1]) < 2 else list(range(7, 11)) ## Split the work over 4 GPUs equally.
    # biases = [True] if int(sys.argv[1]) < 2 else [False] # biases = [True, False]
    in_sizes = list(range(1, 10)) + list(range(10, 100, 10)) + list(range(100, 10000, 100)) + list(range(10000, 80001, 1000))
    out_sizes = list(range(1, 10)) + list(range(10, 100, 10)) + list(range(100, 1001, 100))
    torch.cuda.empty_cache()

    with open (file_name, 'a') as f:
        writer = csv.writer(f)
        
        for inputs in num_inputs:
            for bias in biases:
                for in_size in in_sizes:

                    # if sys.argv[1] == "0" and inputs == 1 and in_size <= 29500:
                    #     continue

                    # elif sys.argv[1] == "1" and inputs == 1 and in_size <= 29600:
                    #     continue

                    # elif sys.argv[1] == "2" and inputs == 8 and in_size <= 61000:
                    #     continue

                    # elif sys.argv[1] == "3" and inputs <= 8:
                    #     continue

                    for out_size in out_sizes:

                        A = torch.randn(inputs, in_size, device=device) + 2
                        assert(A.dtype == torch.float32) # Sanity check

                        ## Create the network! Test 2: try a single layer. Should give the same results.
                        class AddMM(nn.Module):
                            def __init__(self):
                                super().__init__()
                                self.flatten = nn.Flatten()
                                self.lin1 = nn.Linear(in_size, out_size, bias=bias)

                            def forward(self, x):
                                return self.lin1(x)
                        
                        model = AddMM().to(device)
                        opt_model = dynamo.optimize("inductor", nopython=True)(model)

                        ## Built-in profiler.
                        with HiddenPrints():
                            _ = opt_model(A)
                        print(f"{sys.argv[1]}, {inputs}, {in_size}, {out_size}, {bias}")

                        writer.writerow([inputs, in_size, out_size, bias])

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
            # break


if __name__ == "__main__":
    main()
