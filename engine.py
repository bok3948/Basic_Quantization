import math
import sys
from typing import Iterable

import torch

from timm.utils import accuracy
import util.misc as misc


def train_one_epoch(data_loader: Iterable, model: torch.nn.Module, optimizer: torch.optim.Optimizer, criterion: torch.nn.Module, loss_scaler,
                lr_scheduler=None, 
                device=None, epoch=None,
                log_writer=None, args=None):
    model.train()
    metric_logger = misc.MetricLogger(delimiter=" ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch [{}]'.format(epoch)
    
    for it, (inputs, labels) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        logits = model(inputs)
        loss = criterion(logits, labels)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        
        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        
        loss_value_reduce = misc.all_reduce_mean(loss_value)
        
    #gather the stats from all processes
    metric_logger.synchronize_between_processes()
    if it % args.print_freq == 0:
        print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



def evaluate(data_loader, model, device):
    model.eval()

    metric_logger = misc.MetricLogger(delimiter="\t")
    header = 'Test:'

    for inputs, labels in metric_logger.log_every(data_loader, 100, header):
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        #with torch.cuda.amp.autocast():
        logits = model(inputs)

        acc1, acc5 = accuracy(logits, labels, topk=(1, 5))

        batch_size = inputs.shape[0]
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


