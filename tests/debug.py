import torch

B, C, L = 2, 4, 256
u = torch.randn(B, C, L)
delta = torch.rand(B, C, L)
A = torch.randn(C, 4)
B_tensor = torch.randn(B, 4, L)
C_tensor = torch.randn(B, 4, L)
D = torch.randn(C)
delta_bias = torch.randn(C)

print(delta.shape)
print(delta_bias.unsqueeze(-1).shape)
try:
    res = delta + delta_bias.unsqueeze(-1)
    print(res.shape)
except Exception as e:
    print(e)
