"""
    onnx inference mode:   
    inference method for onnx model. 
    NOTE: onnx differs with fake_quant mode. (1)fake_quant do not perform quantization on bias. 
                                             (2) onnx inference mode perform real integer arthimatic computation.
                                             (3) onnx inference mode perform batch folding on preprocessing step.
 
    computation graph: 
    [Input](FP32)
        |
        v
    [Quant]
        |
        v
    [Quantized Input](Int8)================ |
                                            |
                                            v
                                    [Int8 Convolution or linear]=====> [Output](Int32)==|
                                    ^             ^                                     |
                                    |             |                                     |
                                    |             |                                     V
                                    |[int32(Bias /(x_scale * w_scale)))] [int8(Output * (w_sclae*x_scale)/ y_scale)))] 
    ([Weight](FP32))                |             ^                                     |
        |                           |             |                                     |
        v                           |             |                                     V
    ([Quant])                       |       [Bias](FP32)                            [Output](Int8) 
        |                           |                                                   |
        v                           |                                                   V
    [Quantized Weight](Int8)========|                                    
                                                                                [dequantized output](FP32)
"""
import torch

import onnxruntime 

from timm.utils import accuracy

from util.torch_to_onnx import onnx_convert, onnx_prepro, onnx_quantize, ONNX_calib_loader, replace_quant_params
import util.misc as misc


def ONNX_inference(calib_loader, fp_model, quant_model, val_loader, output_dir, args):
    onnx_calib_loader = ONNX_calib_loader(calib_loader)
    onnx_path = onnx_convert(fp_model, model_name=args.model, dummy_size=(1, 3, args.input_size, args.input_size), save_dir=output_dir)
    onnx_path = onnx_prepro(onnx_path) 
    onnx_quantize(onnx_path, onnx_calib_loader)
    onnx_path = onnx_path.replace(".onnx", "_quant.onnx")
    replace_quant_params(onnx_path, quant_model)

    ort_session = onnxruntime.InferenceSession(onnx_path)
    metric_logger = misc.MetricLogger(delimiter="\t")
    header = 'ONNX inference:'

    for inputs, labels in metric_logger.log_every(val_loader, 100, header):

        inputs_np = inputs.numpy()

        ort_inputs = {ort_session.get_inputs()[0].name: inputs_np}
        ort_outs = ort_session.run(None, ort_inputs)

        logits_np = ort_outs[0]
        logits_tensor = torch.from_numpy(logits_np)

        acc1, acc5 = accuracy(logits_tensor, labels, topk=(1, 5))

        batch_size = inputs.shape[0]
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()} , onnx_path