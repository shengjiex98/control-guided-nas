"""Example demonstrating differentiable maximum diameter computation for NAS.

This script shows how to use the differentiable version of get_max_diam in a
PyTorch optimization context, which is useful for neural architecture search.
"""

import torch
import torch.nn as nn
import torch.optim as optim

from control_guided_nas.differentiable_wrapper import get_max_diam_differentiable


def example_gradient_computation():
    """Demonstrate basic gradient computation through get_max_diam."""
    print("=" * 70)
    print("Example 1: Basic Gradient Computation")
    print("=" * 70)

    # Create parameters with gradient tracking
    latency = torch.tensor(50.0, requires_grad=True)  # 50ms
    errors = torch.tensor([0.05, 0.05], requires_grad=True)  # 5% errors

    print(f"Initial latency: {latency.item():.4f} ms")
    print(f"Initial errors: {errors.detach().numpy()}")

    # Compute diameter (differentiable)
    diameter = get_max_diam_differentiable(
        latency / 1000,  # Convert to seconds
        errors,
        sysname="F1",
        temperature=0.01
    )

    print(f"\nComputed diameter: {diameter.item():.6f}")

    # Compute gradients
    diameter.backward()

    print(f"\nGradient w.r.t. latency (ms): {latency.grad.item():.6e}")
    print(f"Gradient w.r.t. errors: {errors.grad.numpy()}")

    print("\nInterpretation:")
    if latency.grad.item() > 0:
        print("  - Increasing latency → increases diameter (reachable set grows)")
    else:
        print("  - Increasing latency → decreases diameter (reachable set shrinks)")

    if errors.grad[0].item() > 0:
        print("  - Increasing errors → increases diameter (reachable set grows)")
    else:
        print("  - Increasing errors → decreases diameter (reachable set shrinks)")


def example_simple_optimization():
    """Demonstrate using diameter in a simple optimization problem."""
    print("\n" + "=" * 70)
    print("Example 2: Simple Optimization (Minimize Diameter)")
    print("=" * 70)

    # Start with suboptimal parameters
    latency = torch.tensor(100.0, requires_grad=True)  # 100ms
    errors = torch.tensor([0.1, 0.1], requires_grad=True)  # 10% errors

    optimizer = optim.Adam([latency, errors], lr=1.0)

    print(f"Initial latency: {latency.item():.2f} ms")
    print(f"Initial errors: {errors.detach().numpy()}")

    # Optimize for 10 steps
    print("\nOptimizing to minimize diameter...")
    for step in range(10):
        optimizer.zero_grad()

        # Compute diameter (this is our "loss")
        diameter = get_max_diam_differentiable(
            latency / 1000,  # Convert to seconds
            errors,
            sysname="F1",
            temperature=0.01
        )

        # Backward pass
        diameter.backward()

        # Update parameters
        optimizer.step()

        # Clamp to reasonable ranges
        with torch.no_grad():
            latency.clamp_(10.0, 200.0)  # 10-200ms
            errors.clamp_(0.001, 0.5)    # 0.1%-50%

        if step % 2 == 0:
            print(f"Step {step:2d}: diameter={diameter.item():.6f}, "
                  f"latency={latency.item():.2f}ms, "
                  f"errors={errors.detach().numpy()}")

    print(f"\nFinal latency: {latency.item():.2f} ms")
    print(f"Final errors: {errors.detach().numpy()}")


def example_nas_architecture_net():
    """Demonstrate using diameter in a NAS context with a neural network."""
    print("\n" + "=" * 70)
    print("Example 3: Neural Architecture Search Context")
    print("=" * 70)

    class SimpleArchitectureNet(nn.Module):
        """Simple network that predicts latency and error parameters."""

        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(4, 16),
                nn.ReLU(),
                nn.Linear(16, 3),  # Output: latency, error1, error2
            )

        def forward(self, x):
            out = self.layers(x)
            # Map outputs to valid ranges
            latency = torch.sigmoid(out[:, 0]) * 190 + 10  # 10-200ms
            errors = torch.sigmoid(out[:, 1:]) * 0.3 + 0.01  # 1%-31%
            return latency, errors

    # Create network and optimizer
    net = SimpleArchitectureNet()
    optimizer = optim.Adam(net.parameters(), lr=0.01)

    # Dummy input (in real NAS, this might be task features)
    task_input = torch.randn(1, 4)

    print("Training architecture network to minimize control diameter...")

    for step in range(20):
        optimizer.zero_grad()

        # Network predicts architecture parameters
        latency, errors = net(task_input)

        # Compute diameter using predicted parameters
        diameter = get_max_diam_differentiable(
            latency[0] / 1000,  # Convert to seconds
            errors[0],
            sysname="F1",
            temperature=0.01
        )

        # In real NAS, you'd combine this with a task loss:
        # loss = task_loss + lambda_diameter * diameter
        loss = diameter

        # Backward through both diameter computation AND network
        loss.backward()
        optimizer.step()

        if step % 5 == 0:
            print(f"Step {step:2d}: diameter={diameter.item():.6f}, "
                  f"latency={latency[0].item():.2f}ms, "
                  f"errors={errors[0].detach().numpy()}")

    print("\nFinal architecture parameters:")
    latency, errors = net(task_input)
    print(f"  Latency: {latency[0].item():.2f} ms")
    print(f"  Errors: {errors[0].detach().numpy()}")


def example_compare_with_original():
    """Compare differentiable version with original implementation."""
    print("\n" + "=" * 70)
    print("Example 4: Comparison with Original Implementation")
    print("=" * 70)

    from control_guided_nas.wrapper import get_max_diam

    latency = 0.05  # 50ms
    errors = [0.05, 0.05]
    sysname = "F1"

    # Original (non-differentiable) version
    original_diameter = get_max_diam(latency, errors, sysname)

    # Differentiable version with different temperatures
    temps = [0.1, 0.01, 0.001]

    print(f"Original (true max):     {original_diameter:.6f}")
    for temp in temps:
        diff_diameter = get_max_diam_differentiable(
            torch.tensor(latency),
            torch.tensor(errors),
            sysname=sysname,
            temperature=temp
        )
        error = abs(diff_diameter.item() - original_diameter)
        print(f"Differentiable (T={temp:5.3f}): {diff_diameter.item():.6f} "
              f"(error: {error:.6e})")

    print("\nNote: Smaller temperature = better approximation of true maximum")
    print("      But too small may cause numerical instability")


if __name__ == "__main__":
    print("\nDifferentiable Maximum Diameter Examples for NAS")
    print("=" * 70)

    example_gradient_computation()
    example_simple_optimization()
    example_nas_architecture_net()
    example_compare_with_original()

    print("\n" + "=" * 70)
    print("Examples completed!")
    print("=" * 70)
