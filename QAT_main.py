import os
import json
import copy
import argparse
import time
import numpy as np
from datetime import timedelta

import torch
import torch.nn as nn

from timm import utils
from timm import create_model
#from timm.scheduler import create_scheduler_v2
from torch.utils.tensorboard import SummaryWriter

from util.datasets import build_dataset
from util.converter import replace_module, remove_hooks
from util.profiler import profiler
from util import misc as misc
from quant.quant_layer import QuantConv2d, QuantLinear
from quant.basic_operation import *
from engine import train_one_epoch, evaluate
from onnx_inference import ONNX_inference

def get_args_parser():
    parser = argparse.ArgumentParser(description='Quantization-Aware Training', add_help=False)

    #setting
    parser.add_argument("--distributed", action="store_true", help="Use distributed training")
    parser.add_argument('--dist_url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--dist-bn', type=str, default='reduce',
                    help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
    parser.add_argument('--device', default='cuda', help='cpu vs cuda')

    #data load
    #/mnt/d/data/image/ILSVRC/Data/CLS-LOC
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'])
    parser.add_argument('--data_path', default='/mnt/d/data/image/ILSVRC/Data/CLS-LOC', type=str, help='path to data')
    parser.add_argument('--eval-crop-ratio', default=0.875, type=float, help="Crop ratio for evaluation")
    parser.add_argument('--input-size', default=224, type=int, help='images input size')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size for training')

    #model
    parser.add_argument('--model', default='resnet18.a1_in1k', type=str, help='model name',
                        choices=['resnet18.a1_in1k', 'resnet50', 'resnet18'])
    parser.add_argument('--pretrained', default='', help='get pretrained weights from checkpoint')
    parser.add_argument("--quantized_model", default="/mnt/c/Users/tang3/OneDrive/바탕 화면/code/my_code/quantization/output_dir/resnet18.a1_in1k/resnet18.a1_in1k_quant.pth")
  
    # quantization parameters
    parser.add_argument('--n_bits_w', default=8, type=int, help='bitwidth for weight quantization')
    parser.add_argument('--n_bits_a', default=8, type=int, help='bitwidth for activation quantization')

    #optimizer
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)

    #run
    parser.add_argument('--epochs', default=4, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    
    #save and log
    parser.add_argument('--print_freq', default=500, type=int)
    parser.add_argument("--save_freq", default=1, type=int)
    parser.add_argument('--output_dir', default='./output_dir', type=str, help='path where to save scale, empty for no saving')

    #onnx inference
    parser.add_argument('--onnx_inference', default=True, help='perform onnx inference')
    return parser

def main(args):

    if args.distributed:
        misc.init_distributed_mode(args)
    device = torch.device(args.device) 
    torch.cuda.empty_cache()

    #load validation dataset for evaluation
    seed = 777
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    if args.output_dir:
        output_dir = f"{args.output_dir}/{args.model}"

        log_dir = output_dir +"/" + "log"
        log_writer = SummaryWriter(log_dir=log_dir)

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    if args.distributed:
        #Sampler
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        train_sampler = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset_train)
    
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size, 
        num_workers=3,
        drop_last=True, sampler=train_sampler
    )
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

    fp_model = copy.deepcopy(model)
    fp_model.eval()

    if args.pretrained:
        if args.pretrained.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.pretrained, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.pretrained, map_location='cpu')

        msg = fp_model.load_state_dict(checkpoint['model'], strict=False)
        print(msg)

    quant_model = replace_module(model, args)
    quant_model.to(device)
    quant_model.eval()

    if args.quantized_model:
        checkpoint = torch.load(args.quantized_model, map_location='cpu')
        msg = quant_model.load_state_dict(checkpoint['model'], strict=False)
        print(msg)
    else:
        raise ValueError("You need to provide pretrained model with quantization parameters, do PTQ first.")

    if args.distributed:
        if args.dist_bn is not None:
            quant_model = nn.SyncBatchNorm.convert_sync_batchnorm(quant_model)
        quant_model = nn.parallel.DistributedDataParallel(quant_model, device_ids=[args.gpu])
        model_without_ddp = quant_model.module
    else:
        model_without_ddp = quant_model

    torch.compile(quant_model)
        
    # remove hook for save memory
    remove_hooks(quant_model)
    print(quant_model)

    # make sure all module mode is quant
    for name, module in quant_model.named_modules():
        if hasattr(module, 'mode'):
            module.mode = "fake_quant"

    # optimizer
    optimizer = torch.optim.SGD(quant_model.parameters(),
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay)
    
    # scheduler         
    #lr_scheduler = create_scheduler_v2(optimizer, args)
    lr_scheduler = None
    
    #loss
    criterion = torch.nn.CrossEntropyLoss().to(args.device)

    #resume
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.start_epoch = checkpoint['epoch'] + 1

    best_acc, best_epoch, max_accuracy = 0.0, 0, 0
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(data_loader_train, model, 
            optimizer, criterion, None, None, device,
            epoch, None, args
        )
        if lr_scheduler is not None:
            lr_scheduler.step(epoch)

        if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                if args.local_rank == 0:
                    print("Distributing BatchNorm running means and vars")
                utils.distribute_bn(model, args.world_size, args.dist_bn == 'reduce')
        
        val_stats = evaluate(data_loader_val, quant_model, device)

        if log_writer is not None:
            log_writer.add_scalar('perf/acc1', val_stats['acc1'], epoch)

        if max_accuracy < val_stats["acc1"]:
            max_accuracy = val_stats["acc1"]
            best_quant_model = quant_model
            best_epoch = epoch
            if args.output_dir:
                checkpoint_path = output_dir + "/" +  'QAT_best_checkpoint.pth'
                misc.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'args': args,
                        "max_accuracy": max_accuracy,
                    }, checkpoint_path)

        #Save
        if args.output_dir and (epoch % args.save_freq == 0 or epoch + 1 ==args.epochs):
            misc.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': args,

            }, os.path.join(output_dir, f'QAT_checkpoint_{epoch}.pth')
            )

        log_stats = {**{f'val_{k}': v for k, v in val_stats.items()},
                        'epoch': epoch, 'best_acc': max_accuracy}
        
        if misc.is_main_process():
            with open(f"{output_dir}/QAT_val_log.txt", mode='a', encoding='utf-8') as f:
                f.write(json.dumps(log_stats) + '\n')
    
    total_time = time.time() - start_time
    total_time_str = str(timedelta(seconds=int(total_time)))
    log_stats["total_time"] = total_time_str
    print(f'Training time {total_time_str}')
    print(f'Best acc: {max_accuracy} at epoch {best_epoch}')

    #get fp model test stats
    print(f"Start evaluation for fp model")
    fp_test_stats = evaluate(data_loader_val, fp_model.to(device), device)

    pf = profiler(dummy_size=(1, 3, args.input_size, args.input_size))
    fp_test_stats["name"] = f"pytorch_{args.model}_fp"
    fp_test_stats["latency(ms)"] = pf.torch_model_latency(fp_model)
    fp_test_stats["size(mb)"] = pf.torch_model_size(fp_model)

    quant_test_stats = {"acc1": max_accuracy}

    print(f"FP model test stats: {fp_test_stats}")
    print(f"Quant model test stats: {quant_test_stats}")

    #normalize acc change
    acc_change = ((quant_test_stats["acc1"] - fp_test_stats["acc1"]) / fp_test_stats["acc1"]) * 100 
    print(f"Relative Acc change(acc(quant) - acc(fp) / acc(fp)): {acc_change:.2f} %")

    summary = {"fp model": fp_test_stats, "fake quant model": quant_test_stats}
    
    # onnx inference
    if args.onnx_inference:
        onnx_quant_val_stats, onnx_path = ONNX_inference(data_loader_train, fp_model.to("cpu"), best_quant_model.to("cpu"), data_loader_val, output_dir, args)
        onnx_quant_val_stats["name"] = f"onnx_{args.model}_quant"
        onnx_quant_val_stats["latency(ms)"] = pf.onnx_model_latency(onnx_path)
        onnx_quant_val_stats["size(mb)"] = pf.onnx_model_size(onnx_path)
        print(f"ONNX inference test stats: {onnx_quant_val_stats}")
    
        misc.print_table([fp_test_stats, onnx_quant_val_stats])

        summary["onnx quant model"] = onnx_quant_val_stats

    with open(f"{output_dir}/QAT_summary.json", "w") as f:
        json.dump(summary, f)

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)