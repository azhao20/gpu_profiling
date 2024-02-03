import torch
import torch.nn as nn
import torch._dynamo
from torch._functorch.aot_autograd import aot_module_simplified

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fcl = nn.Linear(32, 64)

    def forward(self, x):
        x = self.fcl(x)
        return torch.nn.functional.gelu(x)

model = MLP()
batch_size = 8
input = torch.randn(batch_size, 32)

def toy_backend(gm, sample_inputs):
    print("Dynamo produced a fx Graph in Torch IR:")
    gm.print_readable()

    print("Notice that sample_inputs is a list of flattened FakeTensor:")
    print(sample_inputs)
    return gm.forward

torch._dynamo.reset()
fn = torch.compile(backend=toy_backend)(model)

# Triggers compilation of forward graph on the first run
out = fn(input)

def toy_backend(gm, sample_inputs):
    def my_compiler(gm, sample_inputs):
        print("AOTAutograd produced a fx Graph in Aten IR:")
        gm.print_readable()
        return gm.forward
    
    # Invoke AOTAutograd
    return aot_module_simplified(
        gm,
        sample_inputs,
        fw_compiler=my_compiler
    )

torch._dynamo.reset()
fn = torch.compile(backend=toy_backend, dynamic=True)(model)

# Triggers compilation of forward graph on the first run
out = fn(input)
# Triggers compilation of backward graph on the first run
out.sum().backward()

## Decompositions!
from torch._inductor.decomposition import decompositions as default_decompositions

decompositions = default_decompositions.copy()
decompositions.update(
    torch._decomp.get_decompositions([
        torch.ops.aten.addmm,
    ])
)

def toy_backend(gm, sample_inputs):
    def my_compiler(gm, sample_inputs):
        # <implement your compiler here>
        print("Decomposed fx Graph in Aten IR:")
        gm.print_readable()
        return gm

    # Invoke AOTAutograd
    return aot_module_simplified(
        gm,
        sample_inputs,
        decompositions=decompositions,
        fw_compiler=my_compiler
    )

torch._dynamo.reset()
fn = torch.compile(backend=toy_backend, dynamic=True)(model)
out = fn(input)
