import pathlib

import numpy as np
from juliacall import Main as jl
from noisyreach.deviation import AVAIL_SYSTEMS, deviation

LINEAR_SYS = ["F1", "CC"]
NON_LINEAR_SYS = ["CAR"]

julia_initialized = False


def jl_init():
    global julia_initialized
    if not julia_initialized:
        jl.include(str(pathlib.Path(__file__).parent.resolve()) + "/get_max_diam.jl")
        julia_initialized = True


def get_max_diam(latency: float, errors: float | list[float], sysname: str = "F1"):
    if sysname in LINEAR_SYS:
        jl_init()
        s = jl.seval(f"benchmarks[:{sysname}]")
        if isinstance(errors, float):
            errors = [errors] * s.nx
        x0center = np.asarray([1.0] * s.nx)
        x0size = np.asarray([0.1] * s.nx)
        return jl.get_max_diam(
            s,
            int(latency * 1000),
            np.asarray(errors),
            x0center,
            x0size,
            return_pipe=False,
        )[0]
    elif sysname in NON_LINEAR_SYS:
        if isinstance(errors, float):
            errors = [errors] * AVAIL_SYSTEMS[sysname]["dims"]
        return np.max(deviation(latency, [1 - e for e in errors], system=sysname))
    else:
        raise ValueError(
            f"`sysname` value {sysname} not recognized."
            + f"Valid systems include {LINEAR_SYS} for linear systems, and"
            + f"{NON_LINEAR_SYS} for non-linear systems."
        )
