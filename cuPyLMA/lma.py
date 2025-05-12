import cupynumeric as np
import torch
import copy
import warnings

from queue import Queue
from torch.cuda import nvtx
from typing import Callable, Tuple, List

from .config import configuration
from .sliced_tensor import SlicedTensor

NP_TORCH_TYPE_MAP = {np.float32 : torch.float32}
TORCH_NP_TYPE_MAP = {torch.float32 : np.float32}

class LMA:
    @torch.no_grad()
    def _save_parameters(self):
        for buffer in self.device_buffer_map.values():
            buffer['flat_params_backup'] = buffer['flat_params'].clone()

    @torch.no_grad()
    def _update_parameters(self, h_update: torch.Tensor):
        '''Update each model'''
        for device, buffer in self.device_buffer_map.items():
            buffer['flat_params'].add_(-h_update.to(device))

    @torch.no_grad()
    def _restore_parameters(self):
        for buffer in self.device_buffer_map.values():
            buffer['flat_params'].copy_(buffer['flat_params_backup'])


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

    def _async_copy_to_host(self, d_tensor: torch.Tensor, stream: torch.cuda.Stream = None) -> Tuple[torch.Tensor, torch.cuda.Event]:
        '''Asynchronously copying tensor to the host within a stream'''
        if stream is None:
            stream = torch.cuda.current_stream(d_tensor.device)
        with torch.cuda.stream(stream):
            h_tensor = d_tensor.to('cpu', non_blocking=True)
            copy_event = stream.record_event()
            return h_tensor, copy_event

    def _sync_copy_to_host(self, d_tensor: torch.Tensor) -> torch.Tensor:
        h_tensor = d_tensor.detach().to('cpu', non_blocking=True)
        torch.cuda.synchronize(d_tensor.device)
        return h_tensor

    def _build_equation_no_overlap(self, sliced_inputs, sliced_targets, sliced_residuals, model_size, batch_size):
        overlap_h2d = configuration.OPTIM_OVERLAP_H2D
        overlap_d2h_h2d = configuration.OPTIM_OVERLAP_D2H_H2D

        if overlap_d2h_h2d and not overlap_h2d:
            raise ValueError('Enabling bi-direction overlap require host-to-device overlap')
        
        # Distributed jacobian matrix and residuals on multiple devices
        J = np.empty((batch_size, model_size), dtype=self.dtype)
        r = np.empty(batch_size, dtype=self.dtype)

        '''Compute Jacobian matrix'''
        # Buffer for temporary mini-slices on the host
        host_buffer = Queue()

        mini_slice_start = 0
        for input_tensor, target_tensor, residual_tensor in zip(sliced_inputs, sliced_targets, sliced_residuals):
            mini_slice_end = mini_slice_start + input_tensor.shape[0]
            device = input_tensor.device

            # Transfer residual
            with nvtx.range(f'residual_d2h[{mini_slice_start}:{mini_slice_end}]'):
                if overlap_h2d:
                    d2h_stream = torch.cuda.Stream(device) if overlap_d2h_h2d else torch.cuda.current_stream(device)
                    h_target_tensor, copy_event = self._async_copy_to_host(residual_tensor, d2h_stream)
                    host_buffer.put((r, h_target_tensor, mini_slice_start, mini_slice_end, copy_event))
                else:
                    r[mini_slice_start:mini_slice_end] = self._sync_copy_to_host(residual_tensor)
            
            # Compute mini-slice
            with nvtx.range(f'jacobian[{mini_slice_start}:{mini_slice_end}]'):
                d_mini_slice = self._jacobian(input_tensor, target_tensor, self.residual_fn)
                
            # Transfer mini-slice
            with nvtx.range(f'jacobian_d2h[{mini_slice_start}:{mini_slice_end}]'):
                if overlap_h2d:
                    d2h_stream = torch.cuda.Stream(device) if overlap_d2h_h2d else torch.cuda.current_stream(device)
                    d2h_stream.wait_stream(torch.cuda.current_stream(device)) # Wait for completing Jacobian computation
                    h_mini_slice, copy_event = self._async_copy_to_host(d_mini_slice.detach(), d2h_stream)
                    host_buffer.put((J, h_mini_slice, mini_slice_start, mini_slice_end, copy_event))
                else:
                    J[mini_slice_start:mini_slice_end] = self._sync_copy_to_host(d_mini_slice.detach())
            
            # Next mini-slice
            mini_slice_start = mini_slice_end

        if overlap_h2d:
            while not host_buffer.empty():
                x, s, i, j, e = host_buffer.get()
                if e.query():
                    with nvtx.range(f'h2d[{i}:{j}]'):
                        x[i:j] = s
                else:
                    host_buffer.put((x, s, i, j, e))

        '''Compute approximate Hessian and equation RHS'''
        # Determine equation
        with nvtx.range(f'equation_build'):
            if batch_size > model_size:
                JJ = J.T @ J
                rhs = J.T @ r
            else:
                JJ = J @ J.T
                rhs = r

        # Normalization
        normalization_factor = 1.0 / batch_size
        np.multiply(normalization_factor, JJ, out=JJ)
        np.multiply(normalization_factor, rhs, out=rhs)
                
        return J, JJ, rhs

    def _compute_loss(self, sliced_outputs, sliced_targets) -> float:
        '''Compute the loss
            XXX: this assumes using a MSE loss
        '''
        overall_loss = 0.0
        total_items = 0
        for output_tensor, target_tensor in zip(sliced_outputs, sliced_targets):
            loss_val = self.loss_fn(output_tensor, target_tensor).item()
            overall_loss += loss_val * output_tensor.shape[0]
            total_items += output_tensor.shape[0]

        return overall_loss / total_items

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
        self.model_dtype = next(iter(self.device_buffer_map.values()))['flat_params'].dtype

        # Backup parameters
        self._save_parameters()

    def step(self, h_inputs: torch.Tensor, h_targets: torch.Tensor, slice_size: int = None) -> bool:

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

        # Build equation
        J, JJ, rhs = self._build_equation_no_overlap(sliced_inputs, sliced_targets, sliced_residuals, model_size, batch_size)
        overdetermined = batch_size > model_size

        loss_val = self._compute_loss(sliced_outputs, sliced_targets)
        terminating = False
        for i in range(configuration.UPDATE_ATTEMPTS):
            # Computed damped LHS
            JJ_damped = JJ + self.damping_factor * np.eye(JJ.shape[0], dtype=self.dtype)

            # Try to solve equation of updates
            solved = False
            try:
                updates = self.solver(JJ_damped, rhs)
                solved = True
            except Exception as e:
                warnings.warn(f'Singular matrix occurs: damping_factor={self.damping_factor}')

            if solved:
                if not overdetermined:
                    assert J is not None
                    updates = J.T @ updates
                
                # Pinned host updates
                h_updates = torch.tensor(updates, dtype=self.model_dtype, device='cpu', pin_memory=True)
                self._update_parameters(h_updates)

                # Update criteria check
                sliced_outputs = self._forward(sliced_inputs)
                new_loss_val = self._compute_loss(sliced_outputs, sliced_targets)
                if new_loss_val < loss_val:
                    loss_val = new_loss_val

                    # Success in updating, then damp down and save the model
                    self.damping_factor = max(self.damping_factor * self.damping_down, configuration.DAMPING_MIN)
                    self._save_parameters()
                    break
                else:
                    warnings.warn('Failed attempt due to increasing loss')
                    self._restore_parameters()
            
            # Fail in updating, damp up
            self.damping_factor *= self.damping_up

            # Termination criteria check
            terminating = self.damping_factor >= configuration.DAMPING_MAX
            if terminating:
                # Warn and reset damping factor
                warnings.warn('Terminated due to large damping factor')
                self.damping_factor = self.damping_start
                break

        return loss_val, terminating