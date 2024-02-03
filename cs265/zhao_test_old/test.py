import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch._dynamo as dynamo
from typing import List
import traceback as tb
import networkx

def foo(x, y):
    a = torch.sin(x)
    b = torch.cos(x)
    return a + b

def generate_data(b):
    return (
        torch.randn(b, 3, 128, 128).to(torch.float32).cuda(),
        torch.randint(1000, (b,)).cuda(),
    )

from torchvision.models import resnet18
def init_model():
    return resnet18().to(torch.float32).cuda()

def bar(a, b):
    x = a / (torch.abs(a) + 1)
    # if b.sum() < 0:
    #     b = b * -1
    return x * b

def custom_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print("custom backend called with FX graph:")
    gm.graph.print_tabular()
    return gm.forward

def main():
    
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(f"Using {device} device")

    # X = torch.rand(1, 28, 28, device=device)
    # logits = model(X)
    # pred_probab = nn.Softmax(dim=1)(logits)
    # y_pred = pred_probab.argmax(1)
    # print(f"Predicted class: {y_pred}")

    ## Basic optimization testing

    # opt_foo1 = dynamo.optimize("inductor")(foo)
    # print(opt_foo1(torch.randn(10, 10), torch.randn(10, 10)))

    # Reset since we are using a different backend (a custom one).
    dynamo.reset()
    # opt_model = dynamo.optimize(custom_backend)(init_model())
    # opt_model(generate_data(16)[0])

    opt_bar = dynamo.optimize(custom_backend)(bar)
    inp1 = torch.randn(10)
    inp2 = torch.randn(10)
    opt_bar(inp1, inp2)
    opt_bar(inp1, -inp2)

    explanation, out_guards, graphs, ops_per_graph, break_reasons, explanation_verbose = dynamo.explain(
        bar, torch.randn(10), torch.randn(10)
    )
    print(explanation_verbose)

    print("\nPrinted verbose explanation!\n")

    opt_bar = dynamo.optimize("inductor", nopython=True)(bar)
    try:
        opt_bar(torch.randn(10), torch.randn(10))
    except:
        tb.print_exc()

    print("\nMoving onto init_model()!\n")

    ## TODO: this shouldn't give an error, yet it does :(
    opt_model = dynamo.optimize("inductor", nopython=True)(init_model())
    print(opt_model(generate_data(16)[0]))

    try:
        dynamo.export(bar, torch.randn(10), torch.randn(10))
        print("Success")
    except:
        print("Failure")
        tb.print_exc()

    print("\n Exporting the graph\n")
    model_exp = dynamo.export(init_model(), generate_data(16)[0])
    print(model_exp[0](generate_data(16)[0]))

if __name__ == "__main__":
    main()