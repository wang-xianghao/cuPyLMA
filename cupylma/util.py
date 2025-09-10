import torch

from legate.core import get_machine, TaskTarget
from typing import List

# Get available GPUs for torch
def get_available_gpus() -> List[torch.device]:
    num_legate_gpus = get_machine().count(TaskTarget.GPU)
    num_total_gpus = torch.cuda.device_count()
    gpu_id_start = num_legate_gpus
    gpu_devices = []
    
    for gpu_id in range(gpu_id_start, num_total_gpus):
        gpu_devices.append(torch.device(gpu_id))

    return gpu_devices