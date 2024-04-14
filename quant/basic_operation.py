import torch

def quant(x, scaling_factor, max_range=127):
    return (x/scaling_factor).round_().clamp_(-max_range ,max_range)

def dequant(x, scaling_factor):
    return x.mul_(scaling_factor)

def quant_weight(weight, w_scaling_factor, max_range_w=127):
    #channel-wise quantization
    if weight.dim() == 4:  # conv2d 
        shape = (-1, 1, 1, 1)
    elif weight.dim() == 2:  # linear 
        shape = (-1, 1)
    else:
        raise ValueError("Unsupported weight dimensions")

    w_scaling_factor = w_scaling_factor.view(shape)
    quantized_weight = (weight / w_scaling_factor).round().clamp(-max_range_w, max_range_w)
    return quantized_weight

def dequant_weight(quantized_weight, w_scaling_factor):
    #channel-wise dequantization
    if quantized_weight.dim() == 4:  # conv2d 
        shape = (-1, 1, 1, 1)
    elif quantized_weight.dim() == 2:  # linear 
        shape = (-1, 1)
    else:
        raise ValueError("Unsupported weight dimensions")

    w_scaling_factor = w_scaling_factor.view(shape)
    dequantized_weight = quantized_weight * w_scaling_factor
    return dequantized_weight


class Act_Fake_quantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scaling_factor, max_range):
        quantized_input = quant(input, scaling_factor, max_range)
        return dequant(quantized_input, scaling_factor)
    #STE
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None
    
class Weight_Fake_quantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scaling_factor, max_range):
        input = quant_weight(input, scaling_factor, max_range)
        return dequant_weight(input, scaling_factor)
    #STE
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


