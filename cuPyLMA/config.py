class Configuration:
    def __init__(self):
        # Damping
        self.DAMPING_MAX: float = 1e10
        self.DAMPING_MIN: float = 1e-10
        self.UPDATE_ATTEMPTS: int = 10

        # Optimization
        self.OPTIM_OVERLAP: bool = False

        # Debug
        self.VERBOSE_MODEL: bool = False
        self.VERBOSE_OPTIM: bool = False

configuration = Configuration()