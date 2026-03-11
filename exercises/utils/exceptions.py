class ProfilingEarlyStop(Exception):
    """Custom exception to signal early stopping during profiling."""

    def __init__(self, message: str = "Planned early stop triggered after reaching the profiling target step."):
        super().__init__(message)