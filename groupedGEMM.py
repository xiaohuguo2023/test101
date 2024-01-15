import torch

# Define batch size and matrix dimensions
batch_size = 5
m, n, p = 10, 20, 15

# Create random matrices
A = torch.randn(batch_size, m, n)
B = torch.randn(batch_size, n, p)

# Perform batched matrix multiplication
C = torch.bmm(A, B)

print(C.shape)  # Should be [5, 10, 15]

