import subprocess
from prettytable import PrettyTable
import numpy as np

import torch

import onnx
import onnxruntime
from onnxruntime.quantization import QuantType, quantize_static, calibrate, CalibrationDataReader, QuantFormat

from quant.quant_layer import *


def onnx_convert(model, model_name="convnext_small", dummy_size=(1, 3, 224, 224), save_dir="./" ):
    model.eval()
    x = torch.randn(dummy_size, requires_grad=True)

    file_name =  model_name + ".onnx"
    torch.onnx.export(model,              
                    x,                         
                    save_dir + "/"  + file_name,
                    export_params=True,        
                    opset_version=17,          
                    do_constant_folding=True,  
                    input_names = ['input'],   
                    output_names = ['output'], 
                    dynamic_axes={'input' : {0 : 'batch_size'},    
                                    'output' : {0 : 'batch_size'}},
                    )
    print(f"Model {model_name} is converted to ONNX format as {save_dir}/{file_name}")
    return save_dir + "/"  + file_name

def onnx_prepro(onnx_model_path):
    
    #preprocess the model. fusion folding etc
    command = [
        "python", "-m", "onnxruntime.quantization.preprocess",
        "--input", onnx_model_path,
        "--output", onnx_model_path
    ]

    result = subprocess.run(command, check=False)
    print(f"Model {onnx_model_path} is preprocessed")
    return onnx_model_path


class ONNX_calib_loader(CalibrationDataReader):
    def __init__(self, calib_loader):
        self.enum_data = None

        self.nhwc_data_list = []
        for i, (images, _) in enumerate(calib_loader):
            images = images.numpy()
            self.nhwc_data_list.append(images)
            break

        self.input_name = "input"
        self.datasize = len(self.nhwc_data_list)

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(
                [{self.input_name: nhwc_data} for nhwc_data in self.nhwc_data_list]
            )
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None


def onnx_quantize(model_path, onnx_calib_loader):

    model = onnx.load(model_path)
    graph = model.graph

    nodes_to_quantize = []
    for node in graph.node:
        if node.op_type in ["Conv", "Gemm"]: 
            nodes_to_quantize.append(node.name)
    # print(f"node to quantize {nodes_to_quantize}")
    
    quantize_static(
        model_input=model_path,
        model_output=model_path.replace(".onnx", "_quant.onnx"),
        per_channel=True,
        reduce_range=True,
        nodes_to_quantize=nodes_to_quantize,
        nodes_to_exclude=None,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        extra_options={"ActivationSymmetric": True,
                        "WeightSymmetric": True},
        calibration_data_reader=onnx_calib_loader,
        calibrate_method=calibrate.CalibrationMethod.MinMax,
        )
    print(f"Model {model_path} is quantized")

def replace_quant_params(quant_model_path, pytorch_model):
    model = onnx.load(quant_model_path)

    module_params = []
    for name, module in pytorch_model.named_modules():
        sub_dict = {"x_scale": None, "y_scale": None, "w_scale": None}
        if isinstance(module, (QuantConv2d, QuantLinear)):
            sub_dict["x_scale"] = module.x_scale
            sub_dict["y_scale"] = module.y_scale
            sub_dict["w_scale"] = module.w_scale
            if module.bias is not None:
                bias = module.w_scale * module.x_scale
                sub_dict["b_scale"] = bias
            else:
                sub_dict["b_scale"] = None
            sub_dict["name"] = name
            module_params.append(sub_dict) 
    
    module_params.reverse()

    idxs = []
    for i, initializer in enumerate(model.graph.initializer):
        sub_dict = {"input": None, "output": None, "weight": None, "bias": None, "name": None}

        if "scale" in initializer.name:

            if ("Conv" in initializer.name or "weight" in initializer.name) and "input" not in initializer.name and "output" not in initializer.name and "quantized" not in initializer.name:
            
                sub_dict["input"] = i - 3
                sub_dict["output"] = i - 1
                sub_dict["weight"] =  i
                sub_dict["name"] = initializer.name
                idxs.append(sub_dict)

            elif "conv_dw.weight" in initializer.name and "input" not in initializer.name and "output" not in initializer.name and "quantized" not in initializer.name:
                sub_dict["input"] = i - 3
                sub_dict["output"] = i - 1
                sub_dict["weight"] =  i
                sub_dict["name"] = initializer.name
                idxs.append(sub_dict)
                
                
            elif "fc.weight" in initializer.name:
                sub_dict["input"] = i - 3
                sub_dict["output"] = i - 1
                sub_dict["weight"] =  i
                sub_dict["name"] = initializer.name
                idxs.append(sub_dict)
        
    for i, initializer in enumerate(model.graph.initializer):
        if "scale" in initializer.name:
            for sub_dict in idxs:
                if sub_dict["name"].replace("scale", "quantized_scale") in initializer.name and "Conv" in sub_dict["name"]:
                    sub_dict["bias"] = i

                elif "conv_dw.bias" in initializer.name and sub_dict["name"].replace("weight_scale", "bias_quantized_scale") in sub_dict["name"]:
                    sub_dict["bias"] = i

                elif "fc.weight" in sub_dict["name"] and "fc.bias" in initializer.name:
                    sub_dict["bias"] = i
    idxs.reverse()
    assert len(idxs) == len(module_params)

    results = {}
    for _ in range(len(module_params)):
        torch_dict = module_params.pop()
        onnx_dict = idxs.pop()
        for index, initializer in enumerate(model.graph.initializer):
            if index == onnx_dict["input"]:
                if len(initializer.float_data) != 0:
                    del initializer.float_data[:]
                    initializer.float_data.append(float(torch_dict["x_scale"]))
                elif hasattr(initializer, 'raw_data'):
                    initializer.raw_data = torch_dict["x_scale"].numpy().tobytes()
                key = torch_dict["name"] + "_x_scale"
                results[key] = initializer.name

            elif index == onnx_dict["output"]:
                if len(initializer.float_data) != 0:
                    del initializer.float_data[:]
                    initializer.float_data.append(float(torch_dict["y_scale"]))
                elif hasattr(initializer, 'raw_data'):
                    initializer.raw_data = torch_dict["y_scale"].numpy().tobytes()
                key = torch_dict["name"] + "_y_scale"
                results[key] = initializer.name

            elif index == onnx_dict["weight"]:
                if len(initializer.float_data) != 0:
                    del initializer.float_data[:]
                    w_scale_list = torch_dict["w_scale"].numpy().astype(np.float32).tolist()
                    initializer.float_data.extend(w_scale_list)
                elif hasattr(initializer, 'raw_data'):
                    initializer.raw_data = torch_dict["w_scale"].numpy().tobytes()
                key = torch_dict["name"] + "_w_scale"
                results[key] = initializer.name

            elif index == onnx_dict["bias"]:
                if len(initializer.float_data) != 0:
                    if torch_dict["b_scale"] is not None:
                        del initializer.float_data[:]
                        initializer.float_data.append(torch_dict["b_scale"].numpy().astype(np.float32).tolist())
                        
                elif hasattr(initializer, 'raw_data'):
                    if torch_dict["b_scale"] is not None:
                        initializer.raw_data = torch_dict["b_scale"].numpy().tobytes()

                if torch_dict["b_scale"] is not None:
                    key = torch_dict["name"] + "_b_scale"
                    results[key] = initializer.name

    table = PrettyTable()
    table.field_names = ["Torch module name", "ONNX Initializer Name"]

    for key, value in results.items():
        table.add_row([key, value])

    print(table)
  



            




