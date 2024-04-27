import os, sys
import pandas as pd
import torch
import torchvision.models as models
import torch._dynamo as dynamo

from pprint import pprint
from collections import defaultdict
from typing import Any
from torch.utils.flop_counter import FlopCounterMode
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))

aten = torch.ops.aten

def process_mm() -> list[str]:
    return ['m', 'k', 'n']

def process_bmm() -> list[str]:
    return ['b', 'm', 'k', 'n']

def process_conv() -> list[str]:
    return ['x_shape', 'w_shape', 'bias', 'stride', 'padding', 'dilation', 'transposed', 'out_shape']

def process_sdpa() -> list[str]:
    return ['query_shape', 'key_shape', 'value_shape', 'shapes']
    
def process_conv_backward() -> list[str]:
    return ['grad_out_shape', 'x_shape', 'w_shape', 'bias', 'stride', 'padding', 'dilation', \
            'transposed', 'output_padding', 'groups', 'output_mask', 'out_shape', 'shapes']

def process_sdpa_backward() -> list[str]:
    return ['grad_out_shape', 'query_shape', 'key_shape', 'value_shape']

op_registry = {
    aten.mm: process_mm,
    aten.addmm: process_mm,
    aten.bmm: process_bmm,
    aten.baddbmm: process_bmm,
    aten.convolution: process_conv,
    aten._convolution: process_conv,
    aten.convolution_backward: process_conv_backward,
    aten._scaled_dot_product_efficient_attention: process_sdpa,
    aten._scaled_dot_product_flash_attention: process_sdpa,
    aten._scaled_dot_product_efficient_attention_backward: process_sdpa_backward,
    aten._scaled_dot_product_flash_attention_backward: process_sdpa_backward,
}

# TODO: could combine
op_names_registry = {
    aten.mm: "mm",
    aten.addmm: "addmm",
    aten.bmm: "bmm",
    aten.baddbmm: "baddmm",
    aten.convolution: "conv",
    aten._convolution: "_conv",
    aten.convolution_backward: "conv_backward",
    aten._scaled_dot_product_efficient_attention: "sdpea",
    aten._scaled_dot_product_flash_attention: "sdpfa",
    aten._scaled_dot_product_efficient_attention_backward: "spdea_backward",
    aten._scaled_dot_product_flash_attention_backward: "spdfa_backward",
}

model_registry = {
    "vgg19": models.vgg19,
    "vgg16": models.vgg16,
    "resnet18": models.resnet18,
    "resnet101": models.resnet101,
    "alexnet": models.alexnet,
    "squeezenet": models.squeezenet1_0,
    "densenet": models.densenet161,
    "inception": models.inception_v3,
    "googlenet": models.googlenet,
    "shufflenet": models.shufflenet_v2_x1_0,
    "mobilenet": models.mobilenet_v2,
    "resnext50_32x4d": models.resnext50_32x4d,
    "wide_resnet50_2": models.wide_resnet50_2,
    "mnasnet": models.mnasnet1_0
}

device = "cuda:0" if torch.cuda.is_available() else "cpu"
assert(device == "cuda:0")

image = torch.randn(1, 3, 224, 224, device=device)
inception_image = torch.randn(2, 3, 299, 299, device=device) # batch norm training requires > 1 sample.

def get_image(name: str) -> torch.Tensor:
    if name == "inception":
        return inception_image
    return image

def main():
    # Note: we don't care about the model name for now.
    # TODO: what about multiple passes, since Inductor changes over multiple calls?
    # TODO: what about backward? Which loss fn? What are the targets?
    op_to_params = defaultdict(list)
    for name, model in tqdm(model_registry.items(), desc="Processing models"):
        model = model().to(device)
        opt_model = torch.compile(backend="inductor")(model)
        # print("-------------Optimized model-------------------\n", opt_model)
        im = get_image(name)

        flop_counter = FlopCounterMode(model) #, depth = None
        with flop_counter:
            opt_model(im)
        model_shapes: dict[Any, list[Any]] = flop_counter.get_shapes()

        # Add model_shapes to op_to_params
        for op, shapes in model_shapes.items():
            op_to_params[op].extend(shapes)

        dynamo.reset()
        torch.cuda.empty_cache()
        del model

    # Must be called in the current directory.
    dir = "/n/holylabs/LABS/idreos_lab/Users/azhao/gpu_profiling/data/models"
    if not os.path.exists(dir):
        os.makedirs(dir)

    for op, params in op_to_params.items():
        columns = op_registry[op]()
        df = pd.DataFrame(params, columns=columns)
        df.to_csv(os.path.join(dir, op_names_registry[op] + ".csv"), index=False)


if __name__ == "__main__":
    main()