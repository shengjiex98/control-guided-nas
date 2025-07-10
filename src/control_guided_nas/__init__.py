"""Control-guided neural architecture search library.

This library provides tools for analyzing reachability and performance of control
systems under various conditions. It combines Python and Julia code to compute
maximum diameters of reachable sets for different control system types.

System Types:
- Linear systems: F1, CC (handled by Julia backend)
- Multi-dimensional linear systems: ACCLK (handled by Julia backend, 
  appropriate for multi-sensing-error case studies)
- Non-linear systems: CAR (handled by noisyreach Python library)

Main Function:
- get_max_diam(): Computes maximum diameter of reachable set given latency,
  errors, and system name

Constants:
- LINEAR_SYS: Available linear system names
- MULTI_DIM_LINEAR_SYS: Available multi-dimensional linear system names  
- NON_LINEAR_SYS: Available non-linear system names
"""

from .wrapper import LINEAR_SYS, MULTI_DIM_LINEAR_SYS, NON_LINEAR_SYS, get_max_diam

__all__ = ["LINEAR_SYS", "NON_LINEAR_SYS", "MULTI_DIM_LINEAR_SYS", "get_max_diam"]
