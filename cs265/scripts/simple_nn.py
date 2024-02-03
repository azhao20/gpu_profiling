import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader
from torch.utils.flop_counter import FlopCounterMode
from torchvision import datasets, transforms
import torch._dynamo as dynamo
import torch._inductor as torchinductor
from typing import List
import traceback as tb
from torch.fx import symbolic_trace
import torch._inductor.config as config ## Print code.
config.debug = True
import pdb # Debugging

from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.flop_counter import FlopCounterMode
from torch.utils.flop_counter import operator_set
from fvcore.nn import FlopCountAnalysis
from pprint import pprint
from ptflops import get_model_complexity_info

# class HiddenPrints:
#     def __enter__(self):
#         self._original_stdout = sys.stdout
#         sys.stdout = open(os.devnull, 'w')

#     def __exit__(self, exc_type, exc_val, exc_tb):
#         sys.stdout.close()
#         sys.stdout = self._original_stdout

## Initialize a simple model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.linear_relu_stack(x) # Logits

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, (3, 3))
        self.conv2 = nn.Conv2d(32, 32, (3, 3))
        self.conv3 = nn.Conv2d(32, 32, (3, 3))
        self.fc = nn.Linear(21632, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = x.relu()
        x = self.conv2(x)
        x = x.relu()
        x = self.conv3(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert(device == "cuda")
    input_val = torch.rand(1, 28, 28, device=device)
    # input_val = torch.randn(20, 3, 32, 32, device=device)
    
    # First: generate the FX graph for the unoptimized model

    print("Generating FX graph\n")
    # symbolic_traced : torch.fx.GraphModule = symbolic_trace(NeuralNetwork())
    # symbolic_traced : torch.fx.GraphModule = symbolic_trace(CNN())
    # print("Graph", symbolic_traced.graph) # print("Code", symbolic_traced.code)
    # print("Readable", symbolic_traced.print_readable())

    # Create an instance of the model and optimize it
    # Should do the same thing as above!
    model = NeuralNetwork().to(device)
    # model = CNN().to(device)
    opt_model = dynamo.optimize("inductor", nopython=True)(model)
    print("Optimized model", opt_model)

    try:
        # print("Compiling optimized model\n")
        ## Note: do we call export to get the optimized value's graph?
        # model_exp = dynamo.export(opt_model, input_val) 
        # print("Graph", model_exp[0].graph) #; print("Code", model_exp[0].code) #; print("Readable", model_exp[0].print_readable())

        print("\n\n\nTesting------------------\n\n\n")

        # with HiddenPrints():
        with profile(
            activities=[ProfilerActivity.CUDA],
            profile_memory=True, 
            record_shapes=True,
            with_stack=True,
        ) as prof:
        #     with record_function("model_inference"):
            opt_model(input_val).sum()
            # res = opt_model(input_val).sum().backward()
            # opt_model(torch.randn(20, 3, 32, 32, device=device)).sum()

        
        print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total", row_limit=10))

        # print("\n\n\nBase model flops------------------\n\n\n")
        # flop_counter = FlopCounterMode(model, depth=None)
        # with flop_counter:
        #     model(torch.rand(1, 28, 28, device=device)).sum()
        # pprint(operator_set)

        # print("\n\n\nProfiling base model------------------\n\n\n")

        # with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        #     with record_function("model_inference"):
        #         model(torch.rand(1, 28, 28, device=device)).sum()
        #         # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

        # print(prof.key_averages().table())
        
    except:
        print("Failed!")
        tb.print_exc()


    ## Do some flop counting.

    # Half of the flops :(
    # print("\nFlopCountAnalysis---------------------------------")
    # flops = FlopCountAnalysis(model, input_val)
    # pprint(flops.by_module_and_operator()) 
    
    # print("\nPtflops---------------------------------")
    # flops, params = get_model_complexity_info(model, (1, 28, 28), as_strings=True, print_per_layer_stat=True)
    # print('\nFlops: ' + flops)
    # print('Params: ' + params)

    # try:
    #     print("Trying dynamo explain\n")

    #     explanation, out_guards, graphs, ops_per_graph, break_reasons, explanation_verbose = dynamo.explain(
    #         opt_model, torch.rand(1, 28, 28, device=device)
    #         # opt_model, torch.randn(20, 3, 32, 32, device=device)
    #     )
    #     print(explanation_verbose)

    # except:
    #     print("Failed!")
        # tb.print_exc()

    
    

if __name__ == "__main__":
    main()

