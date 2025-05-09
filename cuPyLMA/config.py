class Configuration:
    def __init__(self):
        # Damping
        self.DAMPING_MAX: float = 1e10
        self.DAMPING_MIN: float = 1e-10
        self.UPDATE_ATTEMPTS: int = 10

        # Debug
        self.VERBOSE_MODEL: bool = False

configuration = Configuration()