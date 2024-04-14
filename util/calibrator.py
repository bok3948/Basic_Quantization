import copy

import torch
import torch.distributions as distributions

from quant.basic_operation import *

def max_calibrator_tensor_wise(input, max_range_a):
    max = input.abs().max()
    return  torch.tensor(max /  (max_range_a))


def max_calibrator(input, max_range):
    """
    abs.max() / max_range  = scale_factor, this is for channel wise quantziation, the channel dim is first dim.
    """
    w_scale_factor = torch.zeros(input.size(0)).to(input.device)
    for i in range(input.size(0)):
        w_scale_factor[i] = input[i].abs().max() / (max_range)
    return torch.tensor(w_scale_factor)


def entropy_calibrator(input, max_range_a=127, num_bins=2048, args=None):
    min_divergence = float('inf')
    optimal_m = None
    input = torch.abs(input) 
    x = input.view(-1)

    # 히스토그램과 빈 경계 계산
    bins, bins_edge = torch.histogram(x, bins=num_bins, range=(0, x.max()))
    bin_width = bins_edge[1] - bins_edge[0]
    num_quantized_bins = max_range_a + 1

    for i in range(num_quantized_bins, num_bins):
        start_index = 0
        end_index = i if i <= num_bins else num_bins
        sliced_bins = copy.deepcopy(bins[start_index:end_index])
        nonzeros = (bins != 0).float()

        # 후보 분포
        quantized_Q = torch.zeros((num_quantized_bins), dtype=torch.float32)
        num_merged_bins = len(sliced_bins) // num_quantized_bins

        # Quantize the histogram
        for index in range(num_quantized_bins):
            start = index * num_merged_bins
            end = start + num_merged_bins
            quantized_Q[index] = bins[start:end].sum()
        quantized_Q[-1] += sliced_bins[num_quantized_bins * num_merged_bins:].sum()
        
        # expand quantized_Q to match size the same as bins
        expand_Q = torch.zeros_like(bins, dtype=torch.float32)
        for index in range(num_quantized_bins):
            start = index * num_merged_bins
            end = start + num_merged_bins

            norm = nonzeros[start:end].sum()
            if norm != 0:
                expand_Q[start:end] = quantized_Q[index] / norm

        # add clamping error
        outliers_count = bins[i:].sum()
        expand_Q[-1] += outliers_count

        # 원본 분포 P에서 비어 있는 부분 유지
        expand_Q[bins == 0] = 0

        p_distribution = distributions.Categorical(probs=bins)
        q_distribution = distributions.Categorical(probs=expand_Q)
        divergence = distributions.kl_divergence(p_distribution, q_distribution)

        if divergence <= min_divergence:
            min_divergence = divergence
            optimal_m = i

    threshold = (optimal_m + 0.5) * bin_width
    scale = threshold / (max_range_a)  # alpa / (2**b -1)  = s
    return torch.tensor(scale)
