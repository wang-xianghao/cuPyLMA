import torch
from ..config import configuration
from typing import List
from multiprocessing.dummy import Pool

class ParallelPytorchAdapter:
    def _get_model_device(self, model):
        return next(model.parameters()).device
    
    def __init__(self, model: torch.nn.Module, devices: List[torch.device]):
        main_model_device = self._get_model_device(model)
        if main_model_device not in devices:
            raise ValueError(f"Cannot find main model's device {main_model_device} in {devices}")

        # Map device to model and the corresponding buffer
        self.device_to_model = {}
        self.device_to_buffer = {}
        self.devices = devices

        # Spawn models on multiple devices
        for device in devices:
            if device == main_model_device:
                self.device_to_model[device] = model
            else:
                # Clone the model to the new device
                cloned_model = model.to(device)
                self.device_to_model[device] = cloned_model

        # Initialize buffer
        for device in devices:
            model = self.device_to_model[device]
            buffer = {}
            # Store parameters for stateless computation
            buffer['params'] = { n: p for n, p in model.named_parameters() if p.requires_grad }
            buffer['params_shape'] = [p.numel() for p in buffer['params'].values()]
            buffer['flat_params'] = torch.cat([p.detach().flatten() for p in buffer['params'].values()])
            # Bind model parameters with the detached one
            start = 0
            for _, p in buffer['params'].items():
                size = p.numel()
                p.data = buffer['flat_params'][start : start + size].view_as(p)
                start += size
            # Make argument for stateless model evaluation
            buffer['params_and_buffers'] = {
                **dict(model.named_parameters()),
                **dict(model.named_buffers()),
            }
            # Backup parameters
            buffer['flat_params_backup'] = None
            self.device_to_buffer[device] = buffer

        # Backup model parameters
        self.save_model()

        # Synchronize all devices
        # XXX: this prevent interference on timing
        # Could be removed for better performance (maybe)
        for device in self.devices:
            torch.cuda.synchronize(device)

    def save_model(self):
        if configuration.VERBOSE_MODEL:
            print("[model] save_model()")

        for device in self.device_to_buffer.keys():
            buffer = self.device_to_buffer[device]
            buffer['flat_params_backup'] = buffer['flat_params'].clone()