import torch
import copy
from typing import List
from ..config import configuration

class ParallelTensor:
    def __init__(self, tensors):
        self.tensors = tensors
        self.devices = list(map(lambda x : x.device, tensors))

    def __iter__(self):
        for tensor in self.tensors:
            yield tensor.device, tensor

    def __getitem__(self, index):
        return self.devices[index], self.tensors[index]

    def __len__(self):
        return len(self.tensors)

class ParallelPytorchAdapter:
    def _get_device(self, model):
        return next(model.parameters()).device
    
    def __init__(self, main_model: torch.nn.Module, devices: List[torch.device]):
        main_model_device = self._get_device(main_model)
        if main_model_device not in devices:
            raise ValueError(f"Cannot find main model's device {main_model_device} in {devices}")

        if configuration.VERBOSE_MODEL:
            print(f'[model] devices: {devices}')

        # Map device to model and the corresponding buffer
        self.device_to_model = {}
        self.device_to_buffer = {}
        self.devices = devices

        # Spawn models on multiple devices
        for device in devices:
            if device == main_model_device:
                self.device_to_model[device] = main_model
            else:
                # XXX: need a deepcopy
                self.device_to_model[device] = copy.deepcopy(main_model).to(device)

        # Initialize buffer
        for device, model in self.device_to_model.items():
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

    def save_model(self):
        if configuration.VERBOSE_MODEL:
            print("[model] save_model()")

        for device, buffer in self.device_to_buffer.items():
            buffer['flat_params_backup'] = buffer['flat_params'].clone()

    @torch.no_grad()
    def preprocess(self, batch: torch.Tensor):
        if configuration.VERBOSE_MODEL:
            print(f'[model] preprocess({batch.data_ptr():x})')
        
        batch_size = batch.shape[0]
        mini_batch_num = len(self.devices)
        mini_batch_size = (batch_size + mini_batch_num - 1) // mini_batch_num
        mini_batches_host = torch.split(batch, mini_batch_size)
        mini_batches_device = [None] * mini_batch_num

        # Start parallel transfer
        for device_i, mini_batch_host in enumerate(mini_batches_host):
            device = self.devices[device_i]
            # Transfer to the target device
            mini_batches_device[device_i] = mini_batch_host.to(device, non_blocking=True)
        
        return ParallelTensor(mini_batches_device)

    def _forward_task(self, tensor):
        device = tensor.device
        model = self.device_to_model[device]
        buffer = self.device_to_buffer[device]
        tensor = torch.func.functional_call(model, buffer['params_and_buffers'], tensor)
        return tensor

    @torch.no_grad()
    def forward(self, X: ParallelTensor):
        if configuration.VERBOSE_MODEL:
            print(f'[model] forward')

        # Start forward pass on each device
        output_tensors = []
        for device, tensor in X:
            device = tensor.device
            model = self.device_to_model[device]
            buffer = self.device_to_buffer[device]
            tensor = torch.func.functional_call(model, buffer['params_and_buffers'], tensor)
            output_tensors.append(tensor)
        
        return ParallelTensor(output_tensors)

    def get_model_size(self):
        pass