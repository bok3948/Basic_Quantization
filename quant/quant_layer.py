import torch
import torch.nn as nn
import torch.nn.functional as F

from quant.basic_operation import *

"""
    Currently two modes:
    (1) fp mode: preform operation in FP32. This is for PTQ

    (2) fake quant mode: fake quantization(simulated quantization) mode. This is for QAT

        computation graph: 
        [Input](FP32)
            |
            v
        [Fake Quant]
            |
            v
        [FakeQuantized Input](FP32)=======================|
                                                          |
                                                          v
                                                [Convolution or linear]=====> [Output](FP32)
                                                        ^       \                   |
                                                        |        \                  |
                                                        |         \                 V
                                                        |     [Bias](FP32)     [FakeQuantized Output](FP32)
        [Weight](FP32)                                  |                           |
            |                                           |                           |   
            v                                           |                           V
        [Fake Quant]                                    |                        [Output](FP32)
            |                                           |
            v                                           |
        [FakeQuantized Weight](FP32)====================|


    Future work:

    simple inference mode
    for future work, i will build simple inference mode

    if software(ex.torch) and hardware support integer conv, linear functional operation. the following mode can be used. Currently not supported.

    perform quantization inference. bias is not quantized as additcion is relatively small. unlike onnx quantization, there is no need to quantize and dequantize to conv, linear output. So, in this mode, output is not quantized.

    computation graph: 
        [Input](FP32)
            |
            v
        [Quant]
            |
            v
        [Quantized Input](Int8)============== |
                                              |
                                              v
                                        [Int8 Convolution or linear]=====> [Output](Int32)
                                        ^        \                                  |
                                        |         \                                 |
                                        |          \                                V
                                        |           [Bias](FP32)           [Output * (w_sclae*x_scale)](FP32) 
        ([Weight](FP32))                |                                           |
            |                           |                                           |
            v                           |                                           V
        ([Quant])                       |                                    [Output](FP32)
            |                           |
            v                           |
        [Quantized Weight](Int8)========|
                                  
"""
class QuantConv2d(nn.Conv2d):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    """

    def __init__(self, args=None, **kwargs):
        super(QuantConv2d, self).__init__(
            in_channels=kwargs["in_channels"],
            out_channels=kwargs["out_channels"],
            kernel_size=kwargs["kernel_size"],
            stride=kwargs["stride"],
            padding=kwargs["padding"],
            dilation=kwargs["dilation"],
            groups=kwargs["groups"],
            bias=kwargs["bias"],
            padding_mode=kwargs["padding_mode"]
        )
        
        self.register_buffer("n_bits_w", torch.tensor(args.n_bits_w))
        self.register_buffer("n_bits_a", torch.tensor(args.n_bits_a))

        self.max_range_w = int(2**(self.n_bits_w - 1) - 1)
        self.max_range_a = int(2**(self.n_bits_a - 1) - 1)

        self.register_buffer("w_scale", torch.zeros(kwargs["out_channels"]))
        self.register_buffer("x_scale", torch.tensor(0.0))
        self.register_buffer("y_scale", torch.tensor(0.0))
        self.register_buffer("qweight", None)

        self.mode = 'fp'
        self.args = args 

    def forward(self, x):
        if self.mode == 'fp':
            return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        elif self.mode == 'fake_quant':
            return self.fake_quant_forward(x)
        
        else:
            raise ValueError("Unknown mode: {}".format(self.mode))

    def fake_quant_forward(self, x):
        x = Act_Fake_quantizer.apply(x, self.x_scale.to(x.device), self.max_range_a)
        dequant_weight = Weight_Fake_quantizer.apply(self.weight.data.clone(), self.w_scale.to(x.device), self.max_range_w)
        x = F.conv2d(x, dequant_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        x = Act_Fake_quantizer.apply(x, self.y_scale.to(x.device), self.max_range_a)
        return x
    
    
class QuantLinear(nn.Linear):
    def __init__(self,  args=None, **kwargs):
        super(QuantLinear, self).__init__(
            in_features=kwargs["in_features"],
            out_features=kwargs["out_features"],
            bias=(kwargs["bias"] is not None)
        )

        self.register_buffer("n_bits_w", torch.tensor(args.n_bits_w))
        self.register_buffer("n_bits_a", torch.tensor(args.n_bits_a))

        self.max_range_w = int(2**(self.n_bits_w - 1) - 1)
        self.max_range_a = int(2**(self.n_bits_a - 1) - 1)

        self.register_buffer("w_scale", torch.zeros(kwargs["out_features"]))
        self.register_buffer("x_scale", torch.tensor(0.0))
        self.register_buffer("y_scale", torch.tensor(0.0))
        self.register_buffer("qweight", None)

        self.mode = 'fp'
        self.args = args

    def forward(self, x):
        if self.mode == 'fp':
            out = F.linear(x, self.weight, self.bias)
            return out
        elif self.mode == 'fake_quant':
            return self.fake_quant_forward(x)
        
        else:
            raise ValueError("Unknown mode: {}".format(self.mode))
        
    def fake_quant_forward(self, x):
        x = Act_Fake_quantizer.apply(x, self.x_scale.to(x.device), self.max_range_a)
        dequant_weight = Weight_Fake_quantizer.apply(self.weight.data.clone(), self.w_scale.to(x.device), self.max_range_w)
        x = F.linear(x, dequant_weight, self.bias)
        x = Act_Fake_quantizer.apply(x, self.y_scale.to(x.device), self.max_range_a)
        return  x
    

    
