import os
import json
import argparse
import time
import copy
import numpy as np
from datetime import timedelta

import torch

from timm import create_model

from util.datasets import build_dataset, build_calib_loader
from util.converter import replace_module, remove_hooks
from util.calibrator import max_calibrator, entropy_calibrator, max_calibrator_tensor_wise
from util.profiler import profiler
from util import misc
from quant.quant_layer import QuantConv2d, QuantLinear
from quant.basic_operation import *
from engine import evaluate
from onnx_inference import ONNX_inference

def get_args_parser():
    parser = argparse.ArgumentParser(description='post-training quantization', add_help=False)

    #setting
    parser.add_argument('--device', default='cuda', help='cpu vs cuda')

    #data load
    #/mnt/d/data/image/ILSVRC/Data/CLS-LOC
    parser.add_argument('--data-set', default='CIFAR', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'])
    parser.add_argument('--data_path', default='/mnt/d/data/image', type=str, help='path to ImageNet data')
    parser.add_argument('--eval-crop-ratio', default=0.875, type=float, help="Crop ratio for evaluation")
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    #model
    parser.add_argument('--model', default="resnet18", type=str, help='model name',
                        choices=['resnet18.a1_in1k', 'resnet18'])
    parser.add_argument('--pretrained', default='', help='get pretrained weights from checkpoint')
    
    # quantization parameters
    parser.add_argument('--n_bits_w', default=8, type=int, help='bitwidth for weight quantization')
    parser.add_argument('--n_bits_a', default=8, type=int, help='bitwidth for activation quantization')

    # calibrator
    parser.add_argument('--calibrator', default='max', choices=['entropy', "max"], help='calibrator for activation quantization')
    parser.add_argument('--num_samples', default=50, type=int, help='size of the calibration dataset')
    parser.add_argument('--calib_batch_size', default=10, type=int, help='number of iterations for calibration')

    #save
    parser.add_argument('--output_dir', default='./output_dir', type=str, help='path where to save scale, empty for no saving')

    #evaluate
    parser.add_argument('--no_evaluate', action='store_true', help='whether evaluate quantized model')

    #onnx inference
    parser.add_argument('--onnx_inference', default=True, type=bool, help='perform onnx inference')

    return parser

def main(args):
    device = torch.device(args.device) 
    #load validation dataset for evaluation
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)

    #get calibraition dataloader
    calib_loader = build_calib_loader(dataset_train, num_samples=args.num_samples, seed=seed, args=args)

    dataset_val, _ = build_dataset(is_train=False, args=args)

    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=10, 
        num_workers=3,
        drop_last=False
    )

    # load model
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=True,
        num_classes=args.nb_classes,
    ) 
    # model = torch.hub.load("chenyaofo/pytorch-cifar-models", args.model, pretrained=True)
          
    if args.pretrained:
        if args.pretrained.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.pretrained, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.pretrained, map_location='cpu')

        msg = model.load_state_dict(checkpoint['model'], strict=False)
        print(msg)

    fp_model = copy.deepcopy(model)

    quant_model = replace_module(model, args)
    quant_model.to(device)
    quant_model.eval()
    print(quant_model)

    print(f"Start calibration")
    start_time = time.time()
    # step 1: get output for each layer  this will gonna save in module.out
    for i, (images, _) in enumerate(calib_loader):
        images = images.to(device)
        with torch.no_grad():  
            quant_model(images)

    quant_data = {}
    # activation calibration    
    for idx, (name, module) in enumerate(quant_model.named_modules()):
        if isinstance(module, QuantConv2d) or isinstance(module, QuantLinear):
            # weights calibration
            w_scale = max_calibrator(module.weight.data, module.max_range_w)
            module.w_scale = w_scale

            output = torch.cat(module.out, dim=0)
            input = torch.cat(module.input, dim=0)

            if args.calibrator == "entropy":
                x_scale = entropy_calibrator(input, module.max_range_a, 2048, args)
                y_scale = entropy_calibrator(output, module.max_range_a, 2048, args)
            elif args.calibrator == "max":
                x_scale = max_calibrator_tensor_wise(input, module.max_range_a)
                y_scale = max_calibrator_tensor_wise(output, module.max_range_a)
            else:
                raise ValueError(f"Calibrator {args.calibrator} is not supported")

            module.x_scale= x_scale
            module.y_scale= y_scale

    if args.output_dir:
        output_dir = args.output_dir + f"/{args.model}"
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving quant model to {output_dir}/{args.model}_quant.pth")
        quant_model = quant_model.to("cpu")
        save_dict = {"model": quant_model.state_dict()}
        torch.save(save_dict, f"{output_dir}/{args.model}_quant.pth")

    # remove hook for save memory
    remove_hooks(quant_model)

    if "cuda" in args.device:
        torch.cuda.empty_cache()

    total_time = time.time() - start_time
    total_time_str = str(timedelta(seconds=int(total_time)))
    print(f"Calibration time: {total_time_str}")

    if args.no_evaluate is not True:
        #make sure all module mode is quant
        for name, module in quant_model.named_modules():
            if hasattr(module, 'mode'):
                module.mode = "fake_quant"

        print(f"Start evaluation for quant model")
        quant_test_stats = evaluate(data_loader_val, quant_model.to(device), device)
        print(f"Quant model test stats: {quant_test_stats}")

        # evaltuate original model
        fp_test_stats = evaluate(data_loader_val, fp_model.to(device), device)
        pf = profiler(dummy_size=(1, 3, args.input_size, args.input_size))
        
        fp_test_stats["name"] = f"pytorch_{args.model}_fp"
        fp_test_stats["latency(ms)"] = pf.torch_model_latency(fp_model)
        fp_test_stats["size(mb)"] = pf.torch_model_size(fp_model)

        print(f"FP model test stats: {fp_test_stats}")

        #normalize acc change
        acc_change = ((quant_test_stats["acc1"] - fp_test_stats["acc1"]) / fp_test_stats["acc1"]) * 100 
        print(f"Relative Acc change(acc(quant) - acc(fp) / acc(fp)): {acc_change:.2f} %")

        summary = {"fp model": fp_test_stats, "fake quant model": quant_test_stats}

        # onnx inference
        for name, module in quant_model.named_modules():
            if hasattr(module, 'mode'):
                module.mode = "fp"
        if args.onnx_inference:
            onnx_quant_val_stats, onnx_path = ONNX_inference(calib_loader, fp_model.to("cpu"), quant_model.to("cpu"), data_loader_val, output_dir, args)
            onnx_quant_val_stats["name"] = f"onnx_{args.model}_quant"
            onnx_quant_val_stats["latency(ms)"] = pf.onnx_model_latency(onnx_path)
            onnx_quant_val_stats["size(mb)"] = pf.onnx_model_size(onnx_path)
            print(f"ONNX inference test stats: {onnx_quant_val_stats}")

            misc.print_table([fp_test_stats, onnx_quant_val_stats])
            summary["onnx quant model"] = onnx_quant_val_stats

        with open(f"{output_dir}/PTQ_summary.json", "w") as f:
            json.dump(summary, f)
        

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
