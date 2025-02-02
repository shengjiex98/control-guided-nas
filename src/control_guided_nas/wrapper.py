import numpy as np
from juliacall import Main as jl
from juliacall import Pkg as jlpkg
import pathlib

jl.include(str(pathlib.Path(__file__).parent.resolve()) + "/get_max_diam.jl")

def get_max_diam(latency: float, errors: float|list[float], s: str="F1"):
    s = jl.seval(f"benchmarks[:{s}]")
    if isinstance(errors, float):
        errors = [errors] * s.nx
    x0center = np.asarray([10.] * s.nx)
    x0size = np.asarray([1.] * s.nx)
    return jl.get_max_diam(s, int(latency * 1000), np.asarray(errors), x0center, x0size)[1]
