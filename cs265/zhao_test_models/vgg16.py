import torch
from torch import nn
import torch._dynamo as dynamo
import torchvision.models as models
import torch._inductor.config as config
import logging

# https://pytorch.org/docs/stable/dynamo/troubleshooting.html
dynamo.config.log_level = logging.INFO
config.debug = True

def main():
    # Note: we're only interested in GPU work
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert(device == "cuda")
    n, height, width = 3, 224, 224
    image = torch.randn(n, height, width, device=device)
    vgg19 = models.vgg19().to(device)
    vgg16 = models.vgg16().to(device)
    resnet18 = models.resnet18().to(device)
    alexnet = models.alexnet().to(device)
    squeezenet = models.squeezenet1_0().to(device)
    densenet = models.densenet161().to(device)
    inception = models.inception_v3().to(device)
    googlenet = models.googlenet().to(device)
    shufflenet = models.shufflenet_v2_x1_0().to(device)
    mobilenet = models.mobilenet_v2().to(device)
    resnext50_32x4d = models.resnext50_32x4d().to(device)
    wide_resnet50_2 = models.wide_resnet50_2().to(device)
    mnasnet = models.mnasnet1_0().to(device)

    print("Entering trace!")
    # pdb.set_trace()
    opt_model = torch.compile(backend="inductor")(wide_resnet50_2)
    print("-------------Optimized model-------------------\n", opt_model)
    # with profile(
    #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #     profile_memory=True, 
    #     record_shapes=True,
    #     with_stack=True,
    # ) as prof:
            # with record_function("model_inference"):
    # _ = opt_model(image.unsqueeze(0)).sum().backward()
    
    # print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total", row_limit=100))

    # print("\n\n\nBase model flops------------------\n\n\n")
    # flop_counter = FlopCounterMode(resnet18, depth=None)
    # with flop_counter:
    #     resnet18(image.unsqueeze(0)).sum()
    # pprint(operator_set)

if __name__ == "__main__":
    main()