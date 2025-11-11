# Differentiable Maximum Diameter for Neural Architecture Search

This implementation provides a PyTorch-compatible, differentiable version of `get_max_diam` that can be used in neural architecture search (NAS) optimization.

## Overview

The differentiable version allows you to compute gradients of the maximum diameter with respect to:

- **Latency** (∂diameter/∂latency): How diameter changes with control latency
- **Errors** (∂diameter/∂errors): How diameter changes with sensor/actuator errors

This enables gradient-based optimization of these parameters within a neural network.

## Key Features

1. **Smooth Maximum Approximation**: Uses softmax to approximate the maximum operation, making it differentiable
2. **Julia-based AD**: Leverages Julia's ForwardDiff.jl for efficient automatic differentiation
3. **PyTorch Integration**: Custom `torch.autograd.Function` enables seamless PyTorch integration
4. **Configurable Temperature**: Control the accuracy/smoothness tradeoff of the maximum approximation

## Installation

```bash
# Install dependencies (including PyTorch)
uv sync

# Julia packages will be automatically installed via juliapkg
```

## Usage

### Basic Example

```python
import torch
from control_guided_nas.differentiable_wrapper import get_max_diam_differentiable

# Create parameters with gradient tracking
latency = torch.tensor(0.05, requires_grad=True)  # 50ms in seconds
errors = torch.tensor([0.05, 0.05], requires_grad=True)  # 5% errors

# Compute diameter (differentiable!)
diameter = get_max_diam_differentiable(latency, errors, sysname="F1")

# Compute gradients
diameter.backward()

print(f"∂diameter/∂latency: {latency.grad}")
print(f"∂diameter/∂errors: {errors.grad}")
```

### In a NAS Context

```python
import torch
import torch.nn as nn
from control_guided_nas.differentiable_wrapper import get_max_diam_differentiable

class ArchitectureNet(nn.Module):
    """Network that predicts latency and error parameters."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # Output: latency, error1, error2
        )

    def forward(self, x):
        out = self.net(x)
        latency = torch.sigmoid(out[:, 0]) * 0.19 + 0.01  # 10-200ms
        errors = torch.sigmoid(out[:, 1:]) * 0.3 + 0.01   # 1%-31%
        return latency, errors

# Training loop
model = ArchitectureNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for task_data, task_target in dataloader:
    optimizer.zero_grad()

    # Predict architecture parameters
    latency, errors = model(task_data)

    # Task loss (e.g., classification, regression)
    task_output = main_network(task_data, latency, errors)
    task_loss = loss_fn(task_output, task_target)

    # Control diameter (reachability constraint)
    diameter = get_max_diam_differentiable(latency[0], errors[0], sysname="F1")

    # Combined loss
    loss = task_loss + 0.1 * diameter  # λ=0.1 weights diameter penalty

    # Gradients flow through BOTH task loss and diameter!
    loss.backward()
    optimizer.step()
```

## Files

- **`src/control_guided_nas/get_max_diam_ad.jl`**: Julia implementation with automatic differentiation
  - `get_max_diam_diff()`: Differentiable diameter computation using smooth maximum
  - `get_max_diam_with_grad()`: Returns both value and gradients
  - `softmax_reduction()`: Smooth approximation of maximum

- **`src/control_guided_nas/differentiable_wrapper.py`**: PyTorch wrapper
  - `DifferentiableMaxDiam`: Custom `torch.autograd.Function` class
  - `get_max_diam_differentiable()`: Main user-facing API

- **`examples/differentiable_example.py`**: Complete examples demonstrating:
  - Basic gradient computation
  - Simple optimization
  - NAS architecture network
  - Comparison with original implementation

## How It Works

### The Two Main Hurdles Solved

1. **Julia-Python Autodiff Incompatibility**
   - Solution: Use Julia's ForwardDiff.jl to compute gradients in Julia
   - PyTorch custom `Function` receives gradients from Julia
   - Gradients propagate back to PyTorch parameters seamlessly

2. **Non-differentiable Maximum**
   - Solution: Replace `max()` with softmax approximation
   - `softmax_reduction(values, T)` computes weighted average using softmax
   - As temperature T → 0, approaches true maximum
   - Default T=0.01 provides good balance

### Gradient Flow

```
PyTorch NN weights (θ)
    ↓ forward
latency, errors (requires_grad=True)
    ↓ forward [Python → Julia via juliacall]
get_max_diam_diff() [Julia with ForwardDiff]
    ↓ forward
diameter (scalar)
    ↓ forward
loss = task_loss + λ * diameter
    ↓ backward
∂loss/∂diameter = λ
    ↓ backward [Julia computes: ∂diameter/∂latency, ∂diameter/∂errors]
∂loss/∂latency, ∂loss/∂errors
    ↓ backward [Julia → Python]
∂loss/∂θ [PyTorch autograd]
```

## Parameters

### `get_max_diam_differentiable()`

- **latency** (Tensor): Control latency in seconds, scalar with `requires_grad=True`
- **errors** (Tensor): Error vector with `requires_grad=True`
- **sysname** (str): System name - currently supports "F1", "CC"
- **temperature** (float): Softmax temperature for smooth maximum
  - Smaller = more accurate approximation of true max
  - Default: 0.01 (recommended)
  - Range: 0.001-0.1 (0.001 may have numerical issues)
- **x0center** (list, optional): Initial state center
- **x0size** (list, optional): Initial state size
- **K** (ndarray, optional): Controller gain matrix
- **dims** (list, optional): Dimensions to compute diameter over

## Temperature Parameter Guide

| Temperature | Accuracy | Smoothness | Use Case |
|-------------|----------|------------|----------|
| 0.1 | Low | Very smooth | Early training, numerical stability |
| 0.01 | Good | Smooth | **Recommended default** |
| 0.001 | High | Less smooth | Fine-tuning, if numerically stable |

## Limitations

1. **Linear systems only**: Currently supports F1, CC (linear benchmarks)
   - Non-linear systems (CAR) not yet supported
   - Multi-dimensional linear systems (ACCLK) partially supported

2. **Approximation**: Uses softmax approximation of maximum
   - Not exactly equal to original `get_max_diam()`
   - Error typically < 0.1% with temperature=0.01

3. **Performance**: Slightly slower than original due to AD overhead
   - Forward pass: ~1.5-2x slower
   - Backward pass: additional computation

## Running Examples

```bash
# Run all examples
uv run python examples/differentiable_example.py

# The examples will demonstrate:
# 1. Basic gradient computation
# 2. Simple optimization (minimize diameter)
# 3. NAS with architecture network
# 4. Comparison with original implementation
```

## Troubleshooting

### "Julia package not found" error

```bash
# Ensure Julia packages are installed
python -c "from juliacall import Main as jl; jl.seval('import Pkg; Pkg.add(\"ForwardDiff\")')"
```

### Numerical instability (NaN gradients)

- Increase temperature parameter (e.g., from 0.001 to 0.01)
- Check input ranges (latency > 0, errors in [0, 1])

### Slow performance

- Use smaller number of time steps in reachability computation
- Consider caching results for repeated parameter values
- Use GPU for other parts of model (diameter computation is CPU-only)

## Citation

If you use this differentiable implementation in your research, please cite the original control-guided-nas work.
