import torch
import torch.nn as nn
import torchvision.models as models

# setup
device = 'cuda:0'
model = models.resnet18().to(device)
data = torch.randn(64, 3, 224, 224, device=device)
target = torch.randint(0, 1000, (64,), device=device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

nb_iters = 1
warmup_iters = 0
for i in range(nb_iters):
    optimizer.zero_grad()

    # start profiling after 10 warmup iterations
    if i == warmup_iters: torch.cuda.cudart().cudaProfilerStart()

    # push range for current iteration
    if i >= warmup_iters: torch.cuda.nvtx.range_push("iteration{}".format(i))

    # push range for forward
    if i >= warmup_iters: torch.cuda.nvtx.range_push("profile_range")
    output = model(data)
    # if i >= warmup_iters: torch.cuda.nvtx.range_pop()
    if i >= warmup_iters: torch.cuda.nvtx.range_pop()

    loss = criterion(output, target)

    # if i >= warmup_iters: torch.cuda.nvtx.range_push("backward")
    loss.backward()
    # if i >= warmup_iters: torch.cuda.nvtx.range_pop()

    # if i >= warmup_iters: torch.cuda.nvtx.range_push("opt.step()")
    optimizer.step()
    # if i >= warmup_iters: torch.cuda.nvtx.range_pop()

    # pop iteration range

torch.cuda.cudart().cudaProfilerStop()

# import torch
# from torch.cuda import nvtx

# # Simple GPU operation within an NVTX range
# def gpu_operation():
#     x = torch.rand((1000, 1000), device='cuda')
#     y = torch.rand((1000, 1000), device='cuda')
#     nvtx.range_push("profile_range")  # Start of NVTX range
#     z = x + y  # Example operation on GPU
#     nvtx.range_pop()  # End of NVTX range

# def main():
#     # Check if CUDA is available
#     if torch.cuda.is_available():
#         gpu_operation()
#     else:
#         print("CUDA is not available. GPU operation cannot be performed.")

# if __name__ == "__main__":
#     main()
