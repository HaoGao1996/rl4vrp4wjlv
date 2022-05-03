import torch


x = torch.rand([10, 9, 590])  # (batch_size, hidden_size, nums_grids)
y = torch.rand([590, 30])  # (nums_grids, num_regions)

z = torch.matmul(x, y)

print(z.shape)