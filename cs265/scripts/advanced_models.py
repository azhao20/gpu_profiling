import torch
from torch import nn
import torch._dynamo as dynamo
import torchvision.models as models
import torch._inductor.config as config
import logging
from pprint import pprint
# import pdb # Debugging

from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.flop_counter import FlopCounterMode
from torch.utils.flop_counter import operator_set

# https://pytorch.org/docs/stable/dynamo/troubleshooting.html
dynamo.config.log_level = logging.INFO
config.debug = True

# TODO: import BERT and other models :)

def main():
    # Note: we're only interested in GPU work
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert(device == "cuda")
    n, height, width = 3, 224, 224
    image = torch.randn(n, height, width, device=device)
    resnet18 = models.resnet18(pretrained=True).to(device)

    print("Entering trace!")
    # pdb.set_trace()
    opt_model = torch.compile(backend="inductor")(resnet18)
    print("-------------Optimized model-------------------\n", opt_model)
    # with profile(
    #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #     profile_memory=True, 
    #     record_shapes=True,
    #     with_stack=True,
    # ) as prof:
            # with record_function("model_inference"):
    _ = opt_model(image.unsqueeze(0)).sum().backward()
    
    # print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total", row_limit=100))

    # print("\n\n\nBase model flops------------------\n\n\n")
    # flop_counter = FlopCounterMode(resnet18, depth=None)
    # with flop_counter:
    #     resnet18(image.unsqueeze(0)).sum()
    # pprint(operator_set)

if __name__ == "__main__":
    main()