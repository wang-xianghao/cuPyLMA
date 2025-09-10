import torch

# A simple dense model
class SimpleDense(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_tanh_stack = torch.nn.Sequential(
            torch.nn.Linear(1, 256), torch.nn.Tanh(), torch.nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.linear_tanh_stack(x)