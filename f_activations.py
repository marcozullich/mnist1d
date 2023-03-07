'''
This file is a functional implementation of FlashSigmoid and ShiftSigmoid to circumvent issue https://github.com/pytorch/pytorch/issues/4132 that specifies unexpected behavior of torch.where(), returning NaN as a derivative in some cases.
'''

import torch

class FlashSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, x_bar:float, k_1:float, k_2:float):
        ctx.save_for_backward(input_)
        ctx.x_bar = x_bar
        ctx.k_1 = k_1
        ctx.k_2 = k_2
        return torch.where(input_ > x_bar, 1 + (-k_1)/input_, (-k_2) / ((-1) + input_))
    
    @staticmethod
    def backward(ctx, grad_output):
        input_, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input_ > ctx.x_bar] = ctx.k_1 / (input_[input_ > ctx.x_bar] ** 2)
        grad_input[input_ <= ctx.x_bar] = ctx.k_2 / (((-1) + input_[input_ <= ctx.x_bar]) ** 2)
        return grad_input, None, None, None


def flashsigmoid_function(x, x_bar):
    k_1 = x_bar * x_bar
    k_2 = 1 + x_bar * x_bar - 2 * x_bar
    return (1 - k_1/x) if x > x_bar else k_2 / (1 - k_1)

class ShiftSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_):
        ctx.save_for_backward(input_)
        return torch.where(input_ > 0.5, input_/(.5+input_), (-0.5)/((-1.5)+input_))
    
    @staticmethod
    def backward(ctx, grad_output):
        input_, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input_ > 0.5] = 2 / (1 + 2 * input_[input_ > 0.5]) ** 2
        grad_input[input_ <= 0.5] = 2 / (3 - 2 * input_[input_ <= 0.5]) ** 2
        return grad_input