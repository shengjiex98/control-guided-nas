"""Control-guided neural architecture search wrapper module.

This module provides the main Python interface for computing maximum diameters
of reachable sets for control systems under various conditions.
"""

import pathlib
from typing import List, Union

import numpy as np
from juliacall import Main as jl
from noisyreach.deviation import AVAIL_SYSTEMS, deviation

LINEAR_SYS = ["F1", "CC"]
MULTI_DIM_LINEAR_SYS = ["ACCLK"]
NON_LINEAR_SYS = ["CAR"]

jl.include(str(pathlib.Path(__file__).parent.resolve()) + "/get_max_diam.jl")


def get_max_diam(
    latency: float,
    errors: Union[float, List[float]],
    sysname: str = "ACCLK",
    relative_error: bool = True,
) -> float:
    """Compute maximum diameter of reachable set for a control system.

    Args:
        latency: Control latency in seconds
        errors: Sensor/actuator errors. Can be float or list of floats.
               For multi-sensing-error case studies, multi-dimensional
               linear systems (MULTI_DIM_LINEAR_SYS) are the only appropriate ones.
        sysname: System name. Options:
                - Linear systems: "F1", "CC"
                - Multi-dimensional linear systems: "ACCLK"
                - Non-linear systems: "CAR"
        relative_error: If True, errors are interpreted as relative (percentage)
                       of state values; if False, errors are absolute values.
                       Currently only applies to linear systems (LINEAR_SYS and
                       MULTI_DIM_LINEAR_SYS). Has no effect on non-linear systems.
                       Default: True

    Returns:
        Maximum diameter of reachable set as float

    Raises:
        ValueError: If sysname is not recognized or errors type is invalid
    """
    if sysname in LINEAR_SYS:
        s = jl.seval(f"benchmarks[:{sysname}]")
        if isinstance(errors, float):
            errors_a = np.asarray([errors] * s.nx)
        elif isinstance(errors, list):
            errors_a = np.asarray(errors)
        else:
            raise ValueError(
                f"`errors` must be a float or a list of floats, got {type(errors)}."
            )
        x0center = np.asarray([1.0] * s.nx)
        x0size = np.asarray([0.1] * s.nx)
        return jl.get_max_diam(
            s,
            round(latency * 1000),
            errors_a,
            x0center,
            x0size,
            return_pipe=False,
            relative_error=relative_error,
        )[0]
    elif sysname in MULTI_DIM_LINEAR_SYS:
        if isinstance(errors, list):
            errors_a = np.asarray(errors)
        else:
            raise ValueError(
                f"`errors` must be a list of floats for MULTI_DIM_LINEAR_SYS, got {type(errors)}."
            )
        return jl.get_max_diam_multi_dim(
            sysname,
            round(latency * 1000),
            errors_a,
            return_pipe=False,
        )[0]
    elif sysname in NON_LINEAR_SYS:
        if isinstance(errors, float):
            errors_a = [errors] * AVAIL_SYSTEMS[sysname]["dims"]
        elif isinstance(errors, list):
            errors_a = np.asarray(errors)
        else:
            raise ValueError(
                f"`errors` must be a float or a list of floats, got {type(errors)}."
            )
        return np.max(deviation(latency, [1 - e for e in errors_a], system=sysname))
    else:
        raise ValueError(
            f"`sysname` value {sysname} not recognized."
            + f"Valid systems include {LINEAR_SYS} for linear systems, and"
            + f"{NON_LINEAR_SYS} for non-linear systems."
        )
