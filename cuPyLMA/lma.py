import cupynumeric as np
from typing import Any, Callable, Tuple

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

    def step(self, X, y, slice_size=None):
        adpt = self.adapter
        X = adpt.preprocess(X)
        y = adpt.preprocess(y)

        outputs_ten = adpt.forward(X)

        pass