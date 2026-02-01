from __future__ import annotations

import numpy as np
import control as ct

from .zonotope import (
    Zonotope,
    diameter,
    linear_map,
    minkowski_sum,
    reduce_order,
    remove_redundant_generators,
)


def _phi_matrices(system, k_gain: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Augmented dynamics matches Julia layout:
    # state = [x; u_hold; u_ctrl], where u_hold is last applied control
    # and u_ctrl is the freshly computed control at control instants.
    a, b = system.A, system.B
    nx, nu = a.shape[0], b.shape[1]
    phi_control = np.block(
        [
            [a, b, np.zeros((nx, nu))],
            [np.zeros((nu, nx + nu)), np.eye(nu)],
            [-k_gain, np.zeros((nu, 2 * nu))],
        ]
    )
    phi_hold = np.block(
        [
            [a, b, np.zeros((nx, nu))],
            [np.zeros((nu, nx)), np.eye(nu), np.zeros((nu, nu))],
            [np.zeros((nu, nx + nu)), np.eye(nu)],
        ]
    )
    return phi_control, phi_hold


def _error_bound(
    k_gain: np.ndarray,
    errors: np.ndarray,
    zonotope: Zonotope,
    nx: int,
    nu: int,
    relative_error: bool,
) -> Zonotope:
    # Error zonotope is injected into the u_ctrl dimensions only.
    # If relative_error is True, errors scale with current state bounds.
    if relative_error:
        generators = zonotope.generators[:nx, :]
        max_states = np.sum(np.abs(generators), axis=1) + np.abs(zonotope.center[:nx])
        control_error = -k_gain @ (errors * max_states)
    else:
        control_error = -k_gain @ errors
    diag = np.concatenate([np.zeros(nx), np.zeros(nu), control_error])
    return Zonotope(np.zeros(nx + 2 * nu), np.diag(diag))


def reach(
    phi_fn,
    x0: Zonotope,
    w_fn,
    horizon: int,
    nx: int,
    *,
    max_order: float = np.inf,
    reduced_order: float = 2.0,
    remove_redundant: bool = True,
    return_pipe: bool = False,
    dims: list[int] | None = None,
):
    # Forward reachability:
    # x_{k+1} = Phi(k) * x_k ⊕ W(k, x_k) over a fixed horizon.
    if return_pipe:
        pipe = [x0]

    current = x0
    max_diam = 0.0
    for k in range(1, horizon + 1):
        current = minkowski_sum(linear_map(phi_fn(k - 1), current), w_fn(k - 1, current))
        if remove_redundant:
            current = remove_redundant_generators(current)
        if current.order > max_order:
            current = reduce_order(current, reduced_order)
        if return_pipe:
            pipe.append(current)
        max_diam = max(max_diam, diameter(current, nx, dims))

    return (max_diam, pipe) if return_pipe else (max_diam,)


def get_max_diam(
    system,
    latency_ms: int,
    errors: np.ndarray,
    x0center: np.ndarray,
    x0size: np.ndarray,
    *,
    return_pipe: bool = True,
    relative_error: bool = True,
    dims: list[int] | None = None,
):
    # Main Python entry for linear systems (F1, CC):
    # computes max diameter over horizon, optionally returning the pipe.
    errors = np.asarray(errors, dtype=float).reshape(-1)
    nx = system.A.shape[0]
    nu = system.B.shape[1]
    if errors.shape[0] != nx:
        raise ValueError(
            f"Error ({errors.shape[0]}) must have same dimensions as system ({nx})"
        )

    base_dt = 0.001
    # Discretize at base_dt and compute LQR gain at control period (latency).
    discrete = ct.c2d(system, base_dt, method="zoh")
    k_system = ct.c2d(system, base_dt * latency_ms, method="zoh")
    k_gain, _, _ = ct.dlqr(k_system.A, k_system.B, np.eye(nx), np.eye(nu))

    phi_control, phi_hold = _phi_matrices(discrete, k_gain)
    phi = lambda k: phi_control if k % latency_ms == 0 else phi_hold

    x0 = Zonotope(
        np.concatenate([x0center, np.zeros(2 * nu)]),
        np.diag(np.concatenate([x0size, np.zeros(2 * nu)])),
    )

    def w_fn(k, x):
        if k % latency_ms == 0:
            return _error_bound(k_gain, errors, x, nx, nu, relative_error)
        return Zonotope(np.zeros(nx + 2 * nu), np.zeros((nx + 2 * nu, nx + 2 * nu)))

    return reach(phi, x0, w_fn, 1000, nx, return_pipe=return_pipe, dims=dims)


def models() -> dict[str, ct.StateSpace]:
    sys_rcn = ct.ss(
        np.array(
            [
                [-1 / 2e-6 * (1 / 100000 + 1 / 500000), 1 / (500000 * 2e-6)],
                [1 / (500000 * 1e-5), -1 / 1e-5 * (1 / 500000 + 1 / 200000)],
            ],
            dtype=float,
        ),
        np.array([[1 / (100000 * 2e-6)], [1 / (200000 * 1e-5)]], dtype=float),
        np.eye(2),
        np.zeros((2, 1)),
    )

    sys_f1t = ct.ss(
        np.array([[0.0, 6.5], [0.0, 0.0]]),
        np.array([[0.0], [6.5 / 0.3302]]),
        np.array([[1.0, 0.0]]),
        np.zeros((1, 1)),
    )

    sys_dcm = ct.ss(
        np.array([[-10.0, 1.0], [-0.02, -2.0]]),
        np.array([[0.0], [2.0]]),
        np.array([[1.0, 0.0]]),
        np.zeros((1, 1)),
    )

    sys_css = ct.ss(
        np.array(
            [[0.0, 1.0, 0.0, 0.0], [-8.0, -4.0, 8.0, 4.0], [0.0, 0.0, 0.0, 1.0], [80.0, 40.0, -160.0, -60.0]],
            dtype=float,
        ),
        np.array([[0.0], [80.0], [20.0], [-1120.0]], dtype=float),
        np.array([[1.0, 0.0, 0.0, 0.0]]),
        np.zeros((1, 1)),
    )

    sys_ewb = ct.ss(
        np.array([[0.0, 1.0], [8.3951e3, 0.0]]),
        np.array([[0.0], [4.0451]], dtype=float),
        np.array([[7.9920e3, 0.0]]),
        np.zeros((1, 1)),
    )

    sys_cc1 = ct.ss(
        np.array([[-0.05]], dtype=float),
        np.array([[0.01]], dtype=float),
        np.array([[1.0]], dtype=float),
        np.array([[0.0]], dtype=float),
    )

    sys_cc2 = ct.ss(
        np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-6.0476, -5.2856, -0.238]]),
        np.array([[0.0], [0.0], [2.4767]]),
        np.array([[1.0, 0.0, 0.0]]),
        np.zeros((1, 1)),
    )

    sys_mpc = ct.ss(ct.tf([3.0, 1.0], [1.0, 0.6, 1.0]))

    return {
        "RC": sys_rcn,
        "F1": sys_f1t,
        "DC": sys_dcm,
        "CS": sys_css,
        "EW": sys_ewb,
        "C1": sys_cc1,
        "CC": sys_cc2,
        "MPC": sys_mpc,
    }
