import torch
import copy
import cupynumeric as np
from typing import List, Tuple, Callable
from torch.cuda import nvtx
from multiprocessing.dummy import Pool
from ..config import configuration

class ParallelTensor:
    def __init__(self, tensors):
        self.tensors = tensors
        shape_0 = sum(map(lambda t : t.shape[0], self.tensors))
        self._shape = (shape_0, ) + self.tensors[0].shape[1:]

    @property
    def shape(self) -> Tuple[int]:
        return self._shape

    def __iter__(self):
        for tensor in self.tensors:
            yield tensor.device, tensor

    def __getitem__(self, index):
        return self.tensors[index].device, self.tensors[index]

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
        self.num_devices = len(self.devices)

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

    @torch.no_grad()
    def save_model(self):
        if configuration.VERBOSE_MODEL:
            print("[model] save_model()")

        for _, buffer in self.device_to_buffer.items():
            buffer['flat_params_backup'] = buffer['flat_params'].clone()

    @torch.no_grad()
    def preprocess(self, batch: torch.Tensor, slice_size: int) -> Tuple[ParallelTensor, int]:
        if configuration.VERBOSE_MODEL:
            print(f'[model] preprocess({batch.data_ptr():x})')
        
        # Determine slice size
        batch_size = batch.shape[0]
        if slice_size is None or slice_size > batch_size:
            slice_size = batch_size

        # Distribute each slice to multiple GPU
        result_tensors = []
        for slice_start in range(0, batch_size, slice_size):
            slice_end = min(slice_start + slice_size, batch_size)
            slice = batch[slice_start:slice_end]

            # Divide the slice into mini-slices
            mini_slice_size = (slice_end - slice_start + self.num_devices - 1) // self.num_devices
            mini_slices = torch.split(slice, mini_slice_size)
            # Move each mini-slice to the corresponding device
            for device_i, mini_slice in enumerate(mini_slices):
                device = self.devices[device_i]
                mini_slice_device = mini_slice.to(device, non_blocking=True)
                result_tensors.append(mini_slice_device)
        
        return ParallelTensor(result_tensors), slice_size

    @torch.no_grad()
    def forward(self, X: ParallelTensor):
        if configuration.VERBOSE_MODEL:
            print(f'[model] forward')

        # Start forward pass on each device
        output_tensors = []
        for device, tensor in X:
            with nvtx.range('forward_slice'):
                device = tensor.device
                model = self.device_to_model[device]
                buffer = self.device_to_buffer[device]
                tensor = torch.func.functional_call(model, buffer['params_and_buffers'], tensor)
                output_tensors.append(tensor)
        
        return ParallelTensor(output_tensors)

    def _compute_residuals(self, flat_params, inputs, targets, residual_fn):
        device = flat_params.device
        buffer = self.device_to_buffer[device]
        model = self.device_to_model[device]
        
        buffers = dict(model.named_buffers())
        param_list = torch.split(flat_params, buffer['params_shape'])
        params = {
            name: tensor.view_as(param)
            for (name, param), tensor in zip(buffer['params'].items(), param_list)
        }
        params_and_buffers = {**params, **buffers}
        outputs = torch.func.functional_call(model, params_and_buffers, inputs)
        residual = residual_fn(ParallelTensor([outputs]), ParallelTensor([targets]))[0][1]
        return residual

    @torch.no_grad()
    def jacobian(self, X: ParallelTensor, y: ParallelTensor, residual_fn: Callable) -> ParallelTensor:
        # Get Jacobian slice size and mini-slice size
        model_size = self.get_model_size()
        mini_slice_size = X.tensors[0].shape[0]
        
        # Determine Jacobian evaluation method
        if mini_slice_size > model_size:
            jac_build = torch.func.jacfwd
        else:
            jac_build = torch.func.jacrev

        jacobian_mini_slices = []
        
        # Compute each Jacobian mini-slice
        for (device, input_tensor), (_, target_tensor) in zip(X, y):
            jac_func = jac_build(
                lambda p: self._compute_residuals(
                    p,
                    input_tensor,
                    target_tensor,
                    residual_fn,
                )
            )
            buffer = self.device_to_buffer[device]

            jacobian_mini_slice = jac_func(buffer['flat_params'])

            jacobian_mini_slices.append(jacobian_mini_slice)

        return ParallelTensor(jacobian_mini_slices)
        

    def get_model_size(self) -> int:
        buffer = self.device_to_buffer[self.devices[0]]
        return buffer['flat_params'].shape[0]

    def get_tensor_shape(self, X: ParallelTensor) -> Tuple[int]:
        return X.shape

    def get_tensor_range(self, X: ParallelTensor, idx_start: int, idx_end: int) -> ParallelTensor:
        result_tensors = []

        idx = 0
        for _, tensor in X:
            if idx >= idx_start:
                result_tensors.append(tensor)

            idx += tensor.shape[0]

            if idx == idx_end:
                break
            elif idx > idx_end:
                raise ValueError('Unaligned slice size')

        return ParallelTensor(result_tensors)

    def synchronize(self):
        for device in self.devices:
            torch.cuda.synchronize(device)

    def tensor_to_cupynumeric(self, X: ParallelTensor, dtype: type) -> np.ndarray:
        result = np.empty(X.shape, dtype=dtype)

        # with Pool(len(X.tensors)) as p:
        #     result_arrays = p.map(self._tensor_to_cupynumeric_task, X.tensors)
        
        # idx = 0
        # for arr in result_arrays:
        #     idx_end = idx + arr.shape[0]
        #     result[idx:idx_end] = arr
        #     idx = idx_end

        tensors_host = []
        for device, tensor in X:
            tensors_host.append(tensor.detach().to('cpu', non_blocking=True))

        self.synchronize()

        idx = 0
        for tensor_host in tensors_host:
            idx_end = idx + tensor_host.shape[0]
            result[idx:idx_end] = tensor_host.numpy()
            idx = idx_end

        return result