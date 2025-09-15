import torch

class SimpleConvolutional(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_stack = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, kernel_size=4, stride=2, padding=0),  # (8, 13, 13)
            torch.nn.ELU(),
            torch.nn.Conv2d(8, 4, kernel_size=4, stride=2, padding=0),  # (4, 5, 5)
            torch.nn.ELU(),
            torch.nn.Conv2d(4, 4, kernel_size=2, stride=1, padding=0),  # (4, 4, 4)
            torch.nn.ELU(),
            torch.nn.Conv2d(4, 4, kernel_size=2, stride=1, padding=0),  # (4, 3, 3)
            torch.nn.ELU(),
            torch.nn.Conv2d(4, 4, kernel_size=2, stride=1, padding=0),  # (4, 2, 2)
            torch.nn.ELU(),
            torch.nn.Flatten(),
            torch.nn.Linear(4 * 2 * 2, 10),  # Fully connected layer for 10 classes
        )

    def forward(self, x):
        return self.layer_stack(x)