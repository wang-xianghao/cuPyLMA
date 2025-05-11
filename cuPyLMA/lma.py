import cupynumeric as np
from legate.timing import time
from torch.cuda import nvtx

from typing import Any, Callable, Tuple
from .config import configuration


class LMA:
    def __init__(
        self,
        # TODO: add adapter type
        adapter,
        loss_fn: Callable,
        residual_fn: Callable,
        solver: Callable = np.linalg.solve,
        learning_rate: float = 0.1,
        damping_start: float = 1e-3,
        damping_up: float = 10.0,
        damping_down: float = 0.1,
        dtype=np.float32,
    ):
        self.adapter = adapter
        self.loss_fn = loss_fn
        self.residual_fn = residual_fn
        self.solver = solver
        self.learning_rate = learning_rate
        self.damping_start = damping_start
        self.damping_up = damping_up
        self.damping_down = damping_down
        self.damping_factor = damping_start
        self.dtype = dtype

    def _build_equation_no_overlap(
        self, inputs_ten, targets_ten, residuals_ten, model_size, batch_size, slice_size
    ):
        adpt = self.adapter
        # Pre-allocated memory for distributed Jacobian
        J = np.empty((batch_size, model_size), dtype=self.dtype)

        # Compute Jacobian
        for idx in range(0, batch_size, slice_size):
            idx_end = min(batch_size, idx + slice_size)
            inputs_slice = adpt.get_tensor_range(inputs_ten, idx, idx_end)
            targets_slice = adpt.get_tensor_range(targets_ten, idx, idx_end)

            with nvtx.range(f'jacobian_slice_compute[{idx}:{idx_end}]'):
                J_slice_ten = adpt.jacobian(
                    inputs_slice, targets_slice, self.residual_fn
                )

            with nvtx.range(f'jacobian_slice_transfer[{idx}:{idx_end}]'):
                J[idx:idx_end] = adpt.tensor_to_cupynumeric(J_slice_ten, self.dtype)

        return J, None, None

    def step(self, inputs_ten: Any, targets_ten: Any, slice_size: int = None) -> bool:
        enable_overlap = configuration.OPTIM_OVERLAP_HESSIAN
        adpt = self.adapter

        # Preprocess tensors
        with nvtx.range("preprocess"):
            inputs_ten, slice_size = adpt.preprocess(inputs_ten, slice_size)
            targets_ten, _ = adpt.preprocess(targets_ten, slice_size)

        # Compute outputs
        with nvtx.range("forward"):
            outputs_ten = adpt.forward(inputs_ten)
        # Compute residuals
        with nvtx.range("residual_fn"):
            residuals_ten = self.residual_fn(outputs_ten, targets_ten)

        # Get Jacobian size
        model_size = adpt.get_model_size()
        batch_size = adpt.get_tensor_shape(inputs_ten)[0]

        if enable_overlap:
            # Overlap communication with approximated Hessian
            raise NotImplementedError("No overlap implementation")
        else:
            # Compute approximated Hessian after receiving the whole Jacobian
            J, JJ, rhs = self._build_equation_no_overlap(
                inputs_ten,
                targets_ten,
                residuals_ten,
                model_size,
                batch_size,
                slice_size
            )

        return J
