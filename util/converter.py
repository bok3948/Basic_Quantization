import torch
import torch.nn as nn

from quant.quant_layer import *

def replace_module(module, args):
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            quant_conv2d = QuantConv2d(
                in_channels=child.in_channels, 
                out_channels=child.out_channels, 
                kernel_size=child.kernel_size, 
                stride=child.stride, 
                padding=child.padding, 
                dilation=child.dilation, 
                groups=child.groups, 
                bias=child.bias is not None, 
                padding_mode=child.padding_mode, 
                args=args
            )
            quant_conv2d.load_state_dict(child.state_dict(), strict=False)
            setattr(module, name, quant_conv2d)

            handle = quant_conv2d.register_forward_hook(forward_hook)
            setattr(module, name + '_hook_handle', handle)

        elif isinstance(child, nn.Linear):
            quant_linear = QuantLinear(
                in_features=child.in_features, 
                out_features=child.out_features, 
                bias=child.bias is not None, 
                args=args
            )
            quant_linear.load_state_dict(child.state_dict(), strict=False)
            setattr(module, name, quant_linear)
            handle = quant_linear.register_forward_hook(forward_hook)
            setattr(module, name + '_hook_handle', handle)

        else:
            replace_module(child, args)
    return module


def remove_hooks(module):
    for name, child in module.named_children():
        if hasattr(child, 'mode'):  # Hook이 걸려 있는 경우
            # Hook 핸들을 가져옵니다
            handle = getattr(module, name + '_hook_handle', None)
            if handle:
                handle.remove()  # Hook 제거

        remove_hooks(child)

def forward_hook(module, input, output):
    # fp_model_input, quant_model_input, fp_model_out, quant_model_out 속성 초기화
    if not hasattr(module, 'input'):
        module.input = []

    if not hasattr(module, 'out'):
        module.out = []

    module.input.append(input[0].cpu().detach())
    module.out.append(output.cpu().detach())



