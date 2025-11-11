"""Differentiable PyTorch wrapper for control-guided NAS.

This module provides a PyTorch-compatible, differentiable version of get_max_diam
that can be used in neural architecture search loss functions.
"""

import pathlib

import numpy as np
import torch
from juliacall import Main as jl

# Load the differentiable Julia code
jl.include(str(pathlib.Path(__file__).parent.resolve()) + "/get_max_diam_ad.jl")


class DifferentiableMaxDiam(torch.autograd.Function):
    """PyTorch autograd Function for differentiable maximum diameter computation.

    This enables gradient flow through the reachability analysis for neural
    architecture search optimization.
    """

    @staticmethod
    def forward(ctx, latency, errors, sysname="F1", temperature=0.01,
                x0center=None, x0size=None, K=None, dims=None):
        """Forward pass: compute maximum diameter.

        Args:
            ctx: Context object for saving information for backward pass
            latency: Control latency (scalar tensor or float)
            errors: Sensor/actuator errors (1D tensor or list)
            sysname: System name (str)
            temperature: Softmax temperature for smooth maximum (smaller = closer to true max)
            x0center: Initial state center (optional)
            x0size: Initial state size (optional)
            K: Controller gain matrix (optional)
            dims: Dimensions to compute diameter over (optional)

        Returns:
            Maximum diameter as a scalar tensor
        """
        # Convert PyTorch tensors to numpy/scalars for Julia
        latency_val = float(latency.detach().numpy() if isinstance(latency, torch.Tensor) else latency)
        errors_val = errors.detach().numpy() if isinstance(errors, torch.Tensor) else np.array(errors)

        # Call Julia function that returns (diameter, grad_latency, grad_errors)
        result = jl.get_max_diam_with_grad(
            sysname,
            latency_val,
            errors_val,
            temperature=temperature,
            x0center=x0center,
            x0size=x0size,
            K=K,
            dims=dims
        )

        diameter, grad_latency, grad_errors = result

        # Save gradients for backward pass
        ctx.save_for_backward(
            torch.tensor(grad_latency, dtype=torch.float32),
            torch.tensor(grad_errors, dtype=torch.float32)
        )
        ctx.sysname = sysname

        return torch.tensor(diameter, dtype=torch.float32)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass: return gradients w.r.t. inputs.

        Args:
            ctx: Context object with saved tensors
            grad_output: Gradient of loss w.r.t. output (∂L/∂diameter)

        Returns:
            Tuple of gradients (∂L/∂latency, ∂L/∂errors, None, None, ...)
        """
        grad_latency, grad_errors = ctx.saved_tensors

        # Chain rule: ∂L/∂latency = ∂L/∂diameter * ∂diameter/∂latency
        grad_latency_out = grad_output * grad_latency
        grad_errors_out = grad_output * grad_errors

        # Return gradients for all inputs (None for non-differentiable args)
        return grad_latency_out, grad_errors_out, None, None, None, None, None, None


def get_max_diam_differentiable(
    latency: torch.Tensor | float,
    errors: torch.Tensor | list[float],
    sysname: str = "F1",
    temperature: float = 0.01,
    x0center: list[float] | None = None,
    x0size: list[float] | None = None,
    K: np.ndarray | None = None,
    dims: list[int] | None = None,
) -> torch.Tensor:
    """Compute maximum diameter of reachable set (differentiable version).

    This function can be used in PyTorch computation graphs and supports
    automatic differentiation through backpropagation.

    Args:
        latency: Control latency in seconds (scalar, requires_grad=True for gradients)
        errors: Sensor/actuator errors (vector, requires_grad=True for gradients)
        sysname: System name. Currently supports linear systems: "F1", "CC"
        temperature: Softmax temperature for smooth maximum approximation.
                    Smaller values (e.g., 0.001) give better approximation but
                    may have numerical issues. Default 0.01 is a good balance.
        x0center: Initial state center (optional, uses system defaults if None)
        x0size: Initial state size (optional, uses system defaults if None)
        K: Controller gain matrix (optional, computed via LQR if None)
        dims: Dimensions to compute diameter over (optional, uses all dims if None)

    Returns:
        Maximum diameter as a scalar tensor with gradient information

    Example:
        >>> # In a NAS context
        >>> latency = torch.tensor(0.1, requires_grad=True)
        >>> errors = torch.tensor([0.05, 0.05], requires_grad=True)
        >>> diameter = get_max_diam_differentiable(latency, errors, sysname="F1")
        >>> loss = task_loss + 0.1 * diameter
        >>> loss.backward()
        >>> print(latency.grad)  # Gradient of loss w.r.t. latency
        >>> print(errors.grad)   # Gradient of loss w.r.t. errors

    Note:
        - The diameter computation uses a smooth maximum approximation (softmax)
          instead of the true maximum to enable differentiation. The temperature
          parameter controls the accuracy of this approximation.
        - For best results in optimization, start with temperature=0.01 and
          decrease if needed.
        - This function currently only supports linear systems (F1, CC).
          Non-linear systems (CAR) are not yet supported.
    """
    # Convert inputs to tensors if needed
    if not isinstance(latency, torch.Tensor):
        latency = torch.tensor(latency, dtype=torch.float32)
    if not isinstance(errors, torch.Tensor):
        errors = torch.tensor(errors, dtype=torch.float32)

    # Apply the custom autograd function
    return DifferentiableMaxDiam.apply(
        latency, errors, sysname, temperature, x0center, x0size, K, dims
    )
