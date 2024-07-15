import numpy as np
from juliacall import Main as jl
from juliacall import Pkg as jlpkg
import pathlib

jlpkg.add(url="https://github.com/shengjiex98/ControlBenchmarks.jl.git")
jlpkg.add(url="https://github.com/shengjiex98/NoisyReach.jl.git")

jl.include(str(pathlib.Path(__file__).parent.resolve()) + "/get_max_diam.jl")
jl.seval("using ControlBenchmarks")

def get_max_diam(latency: float, errors: float|list[float], sys: str="F1"):
    sys = jl.seval(f"benchmarks[:{sys}]")
    if isinstance(errors, float):
        errors = [errors] * sys.nx
    x0center = np.asarray([10.] * sys.nx)
    x0size = np.asarray([1.] * sys.nx)
    return jl.get_max_diam(sys, latency, np.asarray(errors), x0center, x0size)
