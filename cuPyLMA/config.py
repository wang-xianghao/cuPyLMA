class Configuration:
    def __init__(self):
        # Damping
        self.DAMPING_MAX: float = 1e10
        self.DAMPING_MIN: float = 1e-10
        self.UPDATE_ATTEMPTS: int = 10

        # Optimization
        self.OPTIM_OVERLAP_HESSIAN: bool = False    # Overlap Jacobian slice computation with approximating Hessian
        self.OPTIM_OVERLAP_TRANDFER: bool = True    # Overlap Jacobian slice computationw with transfer

        # Debug
        self.VERBOSE_MODEL: bool = False
        self.VERBOSE_OPTIM: bool = False

configuration = Configuration()