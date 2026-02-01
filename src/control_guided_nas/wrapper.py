import numpy as np

from .linear_reach import get_max_diam as linear_get_max_diam
from .linear_reach import models

LINEAR_SYS = ["F1", "CC"]
NON_LINEAR_SYS = ["CAR"]


def get_max_diam(latency: float, errors: float | list[float], sysname: str = "F1"):
    if sysname in LINEAR_SYS:
        system = models()[sysname]
        nx = system.A.shape[0]
        if isinstance(errors, float):
            errors = [errors] * nx
        x0center = np.asarray([1.0] * nx)
        x0size = np.asarray([0.1] * nx)
        return linear_get_max_diam(
            system,
            int(latency * 1000),
            np.asarray(errors),
            x0center,
            x0size,
            return_pipe=False,
        )[0]
    elif sysname in NON_LINEAR_SYS:
        from noisyreach.deviation import AVAIL_SYSTEMS, deviation

        if isinstance(errors, float):
            errors = [errors] * AVAIL_SYSTEMS[sysname]["dims"]
        return np.max(deviation(latency, [1 - e for e in errors], system=sysname))
    else:
        raise ValueError(
            f"`sysname` value {sysname} not recognized."
            + f"Valid systems include {LINEAR_SYS} for linear systems, and"
            + f"{NON_LINEAR_SYS} for non-linear systems."
        )
