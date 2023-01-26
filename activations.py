import torch
from torch import nn

class FlashSigmoid(nn.Module):
    def __init__(self, x_bar:float, k_1:float, k_2:float):
        super().__init__()
        assert x_bar > 0 and x_bar < 1, f"x_bar must be in (0, 1), got {x_bar}"
        assert k_1 > 0, f"k_1 must be positive, got {k_1}"
        assert k_2 > 0, f"k_2 must be positive, got {k_2}"
        self.x_bar = x_bar
        self.k_1 = k_1
        self.k_2 = k_2


    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return 1 - self.k_1/x if x < self.x_bar else self.k_2 / (1 - self.k_1)

class SoftSigmoid(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return 0.5 + 0.5 * nn.functional.softsign(x/2)

class ShiftSigmoid(nn.Module):
    def __init__(self, shift:float):
        super().__init__()
        self.shift = shift
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return x/(1+x) if x > 0.5 else (-0.5)/((-1.5)+x)