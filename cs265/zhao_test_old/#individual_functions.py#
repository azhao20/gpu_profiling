import torch
import torch._dynamo as dynamo
import torch._inductor.config as config
config.debug = True

@dynamo.optimize()
def addrelu(a, b):
    return torch.relu(torch.add(a, b))

@dynamo.optimize()
def mult(a, b):
    return torch.matmul(a, b)

@dynamo.optimize()
def multrelu(a, b):
    return torch.relu(torch.relu(torch.relu(torch.matmul(a, b))))

@dynamo.optimize()
def relumult(a, b):
    return torch.matmul(torch.relu(a), torch.relu(b))

@dynamo.optimize()
def relumult_a(a, b):
    return torch.matmul(torch.relu(a), b)

@dynamo.optimize()
def addmm(a, b, c):
    return torch.addmm(a, b, c)

## Note: torch.compile yields the same results.
def multrelu_noop(a, b):
    return torch.relu(torch.matmul(a, b))

def relumult_noop(a, b):
    return torch.matmul(torch.relu(a), torch.relu(b))

# opt_relumult = torch.compile(relumult_noop)
# opt_mult = torch.compile(multrelu_noop)

@dynamo.optimize()
def relu(a):
    return torch.relu(a)

# def mult(a, b):
#     return torch.matmul(a, b)

'''
Things worth noting: TorchInductor simply calls extern.mm when only called upon mult.
Similarly, extern_kernels.addmm
In comparison, TI optimizes multrelu and addrelu:
* (val > 0)
Optimizes ReLU

Difference between torch.compile and dynamo.optimize? Doesn't seem to be any.
'''

# addrelu(torch.randn(128, 8192), torch.randn(128, 8192))
# print("Addrelu-----------------\n")
# addrelu(torch.randn(128, 8192, device="cuda"), torch.randn(128, 8192, device="cuda"))
# multrelu(torch.randn(128, 128), torch.randn(128, 8192))
# print("Multrelu-----------------\n")
# multrelu(torch.randn(128, 128, device="cuda"), torch.randn(128, 8192, device="cuda"))
# relumult(torch.randn(128, 128, device="cuda"), torch.randn(128, 8192, device="cuda"))
# relumult_a(torch.randn(128, 128, device="cuda"), torch.randn(128, 8192, device="cuda"))
# opt_multrelu(torch.randn(128, 128, device="cuda"), torch.randn(128, 8192, device="cuda"))


# opt_mult(torch.randn(128, 128), torch.randn(128, 8192))
# opt_mult(torch.randn(128, 128, device="cuda"), torch.randn(128, 8192, device="cuda"))
# relu(torch.randn(128, 128))
# print("Relu-----------------\n")
# relu(torch.randn(128, 128, device="cuda"))
# mult(torch.randn(128, 128), torch.randn(128, 8192))
# print("Mult-----------------\n")
# mult(torch.randn(128, 128, device="cuda"), torch.randn(128, 8192, device="cuda"))
# addmm(torch.randn(128, 128), torch.randn(128, 128), torch.randn(128, 128))
# addmm(torch.randn(128, 128, device="cuda"), torch.randn(128, 128, device="cuda"), torch.randn(128, 128, device="cuda"))
