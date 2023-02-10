import torch
from torch import nn

class SoftSign(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return nn.functional.softsign(x)

class FlashSigmoid(nn.Module):
    def __init__(self, x_bar:float):
        super().__init__()
        assert x_bar > 0 and x_bar < 1, f"x_bar must be in (0, 1), got {x_bar}"
        self.x_bar = x_bar
        self.k_1 = x_bar * x_bar
        self.k_2 = 1 + self.k_1 - 2 * x_bar


    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # return 1 - self.k_1/x if x < self.x_bar else self.k_2 / (1 - self.k_1)
        return torch.where(x > self.x_bar, 1 + (-self.k_1)/x, (-self.k_2) / ((-1) + x))

class SoftSigmoid(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return 0.5 + 0.5 * nn.functional.softsign(x/2)


class ShiftSigmoid(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return torch.where(x > 0.5, x/(.5+x), (-0.5)/((-1.5)+x))