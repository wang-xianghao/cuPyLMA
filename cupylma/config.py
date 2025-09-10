class Configuration:
    def __init__(self):
        # Damping
        self.DAMPING_MAX: float = 1e10
        self.DAMPING_MIN: float = 1e-10
        self.UPDATE_ATTEMPTS: int = 10

        # Optimization
        self.OPTIM_OVERLAP_H2D: bool = True         # Overlap Jacobian computation with host-to-device communication
        self.OPTIM_OVERLAP_D2H_H2D: bool = True     # Overlap Jacobian computation with both device-to-host
                                                    # and host-to-device communication
        self.OPTIM_CYCLIC_DISPATCH: bool = True     # Allocate sliced task to each device in a cyclic way

        # Debug
        self.VERBOSE_MODEL: bool = False
        self.VERBOSE_OPTIM: bool = False

        self.FORCED_JACOBIAN_MODE: bool = False
        self.JACOBIAN_MODE: str = 'forward'

configuration = Configuration()