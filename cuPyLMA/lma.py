import cupynumeric as np
import torch
import copy

from legate.timing import time
from torch.cuda import nvtx
from typing import Any, Callable, Tuple, List

from .config import configuration
from .sliced_tensor import SlicedTensor


class LMA:
    @torch.no_grad()
    def _save_parameters(self):
        for buffer in self.device_buffer_map.values():
            buffer['flat_params_backup'] = buffer['flat_params'].clone()

    def _replicate_model(self, source_model, devices):
        '''Replicate model on each device and initialize the corresponing buffer'''
        device_model_map = {}
        device_buffer_map = {}

        # Replicate model
        for device in devices:
            if device == source_model:
                device_model_map[device] = source_model
            else:
                device_model_map[device] = copy.deepcopy(source_model).to(device)

        # Initialize buffers
        for device, model in device_model_map.items():
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
            device_buffer_map[device] = buffer

        return device_model_map, device_buffer_map

    @torch.no_grad()
    def _preprocess(self, h_batch: torch.Tensor, slice_size: int) -> Tuple[SlicedTensor, int]:
        '''Slice batch into mini-slices and distribute them to devices'''
        batch_size = h_batch.shape[0]
        if slice_size is None or slice_size > batch_size:
            slice_size = batch_size

        d_mini_slice_list = []
        for slice_start in range(0, batch_size, slice_size): # Split batch into slices
            slice_end = min(slice_start + slice_size, batch_size)
            h_slice = h_batch[slice_start:slice_end]

            mini_slice_size = (slice_end - slice_start + self.num_devices - 1) // self.num_devices
            h_mini_slice_list = torch.split(h_slice, mini_slice_size)
            # XXX: figure out whether how async memcopy works
            for device_i, mini_slice in enumerate(h_mini_slice_list): # Split slice into mini-slices
                device = self.devices[device_i]
                d_mini_slice = mini_slice.to(device, non_blocking=True)
                d_mini_slice_list.append(d_mini_slice)

        return SlicedTensor(d_mini_slice_list), slice_size
    
    @torch.no_grad()
    def _forward(self, sliced_inputs: SlicedTensor) -> SlicedTensor:
        '''Perform forward pass on sliced input tensors'''
        outputs_list = []

        for tensor in sliced_inputs:
            device = tensor.device
            with nvtx.range('forward_slice'):
                model = self.device_model_map[device]
                buffer = self.device_buffer_map[device]
                outputs_list.append(torch.func.functional_call(model, buffer['params_and_buffers'], tensor))

        return SlicedTensor(outputs_list)

    def _compute_residuals(self, flat_params, input_tensor, target_tensor, residual_fn):
        '''Compute residuals with model parameters as the input'''
        device = flat_params.device
        buffer = self.device_buffer_map[device]
        model = self.device_model_map[device]
        
        buffers = dict(model.named_buffers())
        param_list = torch.split(flat_params, buffer['params_shape'])
        params = {
            name: tensor.view_as(param)
            for (name, param), tensor in zip(buffer['params'].items(), param_list)
        }
        params_and_buffers = {**params, **buffers}
        output_tensor = torch.func.functional_call(model, params_and_buffers, input_tensor)
        return residual_fn(output_tensor, target_tensor)

    def _synchronize(self):
        '''Synchronize all devices'''
        for device in self.devices:
            torch.cuda.synchronize(device)

    @torch.no_grad
    def _jacobian(self, input_tensor: torch.Tensor, target_tensor: torch.Tensor, residual_fn: Callable) -> torch.Tensor:
        '''Compute one mini-slice of Jacobian'''
        model_size = self.model_size
        batch_size = input_tensor.shape[0]

        # Determine Jacobian evaluation method
        if batch_size > model_size:
            jac_build = torch.func.jacfwd
        else:
            jac_build = torch.func.jacrev

        # Compute Jacobian
        jac_func = jac_build(
            lambda p: self._compute_residuals(
                p,
                input_tensor,
                target_tensor,
                residual_fn,
            )
        )

        device = input_tensor.device
        buffer = self.device_buffer_map[device]
        J = jac_func(buffer['flat_params'])

        return J

    def _build_equation_no_overlap(self, sliced_inputs, sliced_targets, sliced_residuals, model_size, batch_size):
        overlap_transfer = configuration.OPTIM_OVERLAP_TRANDFER
        
        # Pre-allocated memory for distributed Jacobian
        J = np.empty((batch_size, model_size), dtype=self.dtype)

        h_mini_slice_list = []

        # Compute Jacobian
        idx = 0
        for input_tensor, target_tensor in zip(sliced_inputs, sliced_targets):
            idx_end = idx + input_tensor.shape[0]
            with nvtx.range(f'jacobian_mini_slice_compute[{idx}:{idx_end}]'):
                d_mini_slice = self._jacobian(input_tensor, target_tensor, self.residual_fn)
            with nvtx.range(f'jacobian_mini_slice_transfer[{idx}:{idx_end}]'):
                if overlap_transfer:
                    # Overlap Jacobian computation with transfer
                    stream = torch.cuda.Stream(input_tensor.device)
                    with torch.cuda.stream(stream):
                        stream.wait_stream(torch.cuda.default_stream(input_tensor.device))
                        h_mini_slice = d_mini_slice.detach().to('cpu', non_blocking=True)
                        h_mini_slice_list.append((h_mini_slice, idx, idx_end))
                else:
                    # No overlap
                    J[idx:idx_end] = d_mini_slice.detach().cpu()
            idx = idx_end

        if overlap_transfer:
            # Wait all async transfers
            self._synchronize()
            for h_mini_slice, idx, idx_end in h_mini_slice_list:
                J[idx:idx_end] = h_mini_slice

        return J, None, None
        

    def __init__(
        self,
        # TODO: add adapter type
        model: torch.nn.Module,
        devices: List[torch.device],
        loss_fn: Callable,
        residual_fn: Callable,
        solver: Callable = np.linalg.solve,
        learning_rate: float = 0.1,
        damping_start: float = 1e-3,
        damping_up: float = 10.0,
        damping_down: float = 0.1,
        dtype=np.float32,
    ):
        self.loss_fn = loss_fn
        self.residual_fn = residual_fn
        self.solver = solver
        self.learning_rate = learning_rate
        self.damping_start = damping_start
        self.damping_up = damping_up
        self.damping_down = damping_down
        self.damping_factor = damping_start
        self.dtype = dtype

        # Replicate model on each device
        self.device_model_map, self.device_buffer_map = self._replicate_model(model, devices)
        self.devices = devices
        self.num_devices = len(self.devices)
        self.model_size = next(iter(self.device_buffer_map.values()))['flat_params'].shape[0]

        # Backup parameters
        self._save_parameters()

    def step(self, h_inputs: Any, h_targets: Any, slice_size: int = None) -> bool:

        # Preprocess tensors
        with nvtx.range("preprocess"):
            sliced_inputs, slice_size = self._preprocess(h_inputs, slice_size)
            sliced_targets, _ = self._preprocess(h_targets, slice_size)

        # Compute outputs
        with nvtx.range("forward"):
            sliced_outputs = self._forward(sliced_inputs)
        
        # Compute residuals
        with nvtx.range("residual_fn"):
            residual_list = []
            for output_tensor, target_tensor in zip(sliced_outputs, sliced_targets):
                residual_list.append(self.residual_fn(output_tensor, target_tensor))
            sliced_residuals = SlicedTensor(residual_list)

        # Get Jacobian size
        model_size = self.model_size
        batch_size = sliced_inputs.shape[0]

        J, JJ, rhs = self._build_equation_no_overlap(sliced_inputs, sliced_targets, sliced_residuals, model_size, batch_size)

        return J