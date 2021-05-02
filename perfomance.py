import time
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.autograd.profiler import profile


MB = 1024 * 1024
ms = 1000


def profile_net(model: nn.Module, *model_input) -> Dict[str, Any]:
    """A tool that allows you to collect data on the performance of a neural network.

    Args:
        model: PyTorch model to profile.
        model_input: Input to the model (data, parameters, etc.)

    Returns:
        Information about shape, time, memory and model size.
    """
    torch.cuda.synchronize()
    with profile(use_cuda=True, profile_memory=True) as prof:
        start_time = time.perf_counter()
        output = model(*model_input)
        end_time = time.perf_counter()

    return {
        'output_shape': output.shape,
        'elapsed_time, ms': np.round((end_time - start_time) / ms, 2),
        'cpu_memory, Mb': np.round(prof.total_average().cpu_memory_usage / MB, 2),
        'gpu_memory, Mb': np.round(prof.total_average().cuda_memory_usage / MB, 2),
        'model_size, Mb': np.round(
            sum(p.numel() * p.element_size() for p in model.parameters()) / MB, 2
        ),
    }
