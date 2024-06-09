import numpy as np
from juliacall import Main as jl
from juliacall import Pkg as jlpkg
import pathlib

jlpkg.add(url="https://github.com/shengjiex98/ControlBenchmarks.jl.git")
jlpkg.add(url="https://github.com/shengjiex98/NoisyReach.jl.git")

jl.include(str(pathlib.Path(__file__).parent.resolve()) + "/get_max_diam.jl")

jl.seval("using ControlBenchmarks")
sys = jl.seval("benchmarks[:F1]")
x0center = np.asarray([10., 10.])
x0size = np.asarray([1., 1.])

def get_max_diam(latency: float, errors: list[float]):
    return jl.get_max_diam(sys, latency, np.asarray(errors), x0center, x0size)
