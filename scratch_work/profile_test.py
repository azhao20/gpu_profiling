import os
import logging # Is this necessary?
import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode
import torch._dynamo as dynamo
import torch._inductor as torchinductor
import traceback as tb
from torch.fx import symbolic_trace
import torch._inductor.config as config ## Print code.
from torch.cuda import nvtx
config.debug = True

from torch.profiler import profile, record_function, ProfilerActivity

# class HiddenPrints:
#     def __enter__(self):
#         self._original_stdout = sys.stdout
#         sys.stdout = open(os.devnull, 'w')

#     def __exit__(self, exc_type, exc_val, exc_tb):
#         sys.stdout.close()
#         sys.stdout = self._original_stdout

# Define operation.
# @dynamo.optimize()
# def mult(a, b):
#     return torch.matmul(a, b)

# @dynamo.optimize()
# def addmm(a, b, c):
#     return torch.addmm(a, b, c)

## Initialize a simple model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            # nn.ReLU(),
            # nn.ReLU(),
            # nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.linear_relu_stack(x) # Logits

class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu=nn.ReLU()

    def forward(self, x):
        for k in range(10): # K = 20
            x = torch.mul(x, k + 1)
        return x

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, (3, 3))
        # self.conv2 = nn.Conv2d(32, 32, (3, 3))
        # self.conv3 = nn.Conv2d(32, 32, (3, 3))
        # self.fc = nn.Linear(21632, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        # x = x.relu()
        # x = self.conv2(x)
        # x = x.relu()
        # x = self.conv3(x)
        # x = x.flatten(1)
        # x = self.fc(x)
        return x

def main():

    # Try using logging to print everything
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)
    console_handler = logging.StreamHandler()  # Create a handler that outputs to the console
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    log.addHandler(console_handler)  # Add the console handler to the logger

    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert(device == "cuda")
    
    # First: generate the FX graph for the unoptimized model

    print("Generating FX graph\n")
    # symbolic_traced : torch.fx.GraphModule = symbolic_trace(NeuralNetwork())
    # symbolic_traced : torch.fx.GraphModule = symbolic_trace(CNN())
    # print("Graph", symbolic_traced.graph) # print("Code", symbolic_traced.code)
    # print("Readable", symbolic_traced.print_readable())

    USE_OPT_MODEL = True
    USE_LINEAR = False

    ## For the addmm kernel.
    # input_val = torch.rand(28, 28, device=device)
    if USE_LINEAR:
        model = NeuralNetwork().to(device)
        input_val = torch.rand(1, 28, 28, device=device)
        # model = NN().to(device)
        # input_val = torch.randn(100, 100, device=device) + 2
    else:
        model = CNN().to(device)
        input_val = torch.randn(20, 3, 32, 32, device=device)

    if USE_OPT_MODEL:
        model = dynamo.optimize("inductor", nopython=True)(model)
    # print("Optimized model", opt_model)

    try:
        # print("Compiling optimized model\n")
        ## Note: do we call export to get the optimized value's graph?
        # model_exp = dynamo.export(opt_model, input_val) 
        # print("Graph", model_exp[0].graph) #; print("Code", model_exp[0].code) #; print("Readable", model_exp[0].print_readable())

        print("\n\n\nTesting------------------\n\n\n")

        # with HiddenPrints():

        # torch.cuda.nvtx.range_push("profile_range")
        # addmm(input_val, input_val, input_val)
        # torch.cuda.nvtx.range_pop()

        torch.cuda.cudart().cudaProfilerStart()
        nvtx.range_push("profile_range")
        res = model(input_val)
        torch.cuda.synchronize()
        nvtx.range_pop()
        torch.cuda.cudart().cudaProfilerStop()
        # print(f"res: {res}")
        # print(f"res.sum(): {res.sum()}")
        # res = opt_model(input_val).sum().backward()
        # opt_model(torch.randn(20, 3, 32, 32, device=device)).sum()

    except:
        print("Failed!")
        tb.print_exc()
    

if __name__ == "__main__":
    main()

