import cupynumeric as np
from legate.timing import time
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
        dtype=np.float32
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
        self.dtype = np.float32

    def step(self, inputs_ten, targets_ten, slice_size=None):
        adpt = self.adapter
        inputs_ten = adpt.preprocess(inputs_ten)
        targets_ten = adpt.preprocess(targets_ten)

        # Compute outputs
        outputs_ten = adpt.forward(inputs_ten)
        
        