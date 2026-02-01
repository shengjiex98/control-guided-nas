from __future__ import annotations

import control as ct
import numpy as np

from .zonotope import (
    Zonotope,
    diameter,
    linear_map,
    minkowski_sum,
    reduce_order,
    remove_redundant_generators,
)


def _phi_matrices(
    system, k_gain: np.ndarray, let: bool
) -> tuple[np.ndarray, np.ndarray]:
    a, b = system.A, system.B
    nx, nu = a.shape[0], b.shape[1]
    if let:
        # Augmented dynamics for logical execution time (LET):
        # state = [x; u_hold; u_ctrl], where u_hold is last applied control
        # and u_ctrl is the freshly computed control at control instants.
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
    else:
        # Immediate-apply dynamics (legacy behavior):
        # state = [x; u], where u is applied in the same sampling period.
        phi_control = np.block([[a, b], [-k_gain, np.zeros((nu, nu))]])
        phi_hold = np.block([[a, b], [np.zeros((nu, nx)), np.eye(nu)]])
    return phi_control, phi_hold


def _error_bound(
    k_gain: np.ndarray,
    errors: np.ndarray,
    zonotope: Zonotope,
    nx: int,
    nu: int,
    relative_error: bool,
    let: bool,
) -> Zonotope:
    # Error zonotope is injected into the u_ctrl dimensions only.
    # If relative_error is True, errors scale with current state bounds.
    if relative_error:
        generators = zonotope.generators[:nx, :]
        max_states = np.sum(np.abs(generators), axis=1) + np.abs(zonotope.center[:nx])
        control_error = -k_gain @ (errors * max_states)
    else:
        control_error = -k_gain @ errors
    if let:
        diag = np.concatenate([np.zeros(nx), np.zeros(nu), control_error])
        return Zonotope(np.zeros(nx + 2 * nu), np.diag(diag))
    diag = np.concatenate([np.zeros(nx), control_error])
    return Zonotope(np.zeros(nx + nu), np.diag(diag))


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
        current = minkowski_sum(
            linear_map(phi_fn(k - 1), current), w_fn(k - 1, current)
        )
        # If the reachable set becomes non-finite (inf/nan), the system has
        # blown up numerically for this horizon or parameter set. At that point
        # further propagation is meaningless, so we return +inf as the diameter
        # to signal "unbounded/overflow" rather than raising or producing NaNs.
        # This makes long-running scans robust and keeps results interpretable.
        if (
            not np.isfinite(current.center).all()
            or not np.isfinite(current.generators).all()
        ):
            max_diam = np.inf
            break
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
    k_gain_override: np.ndarray | None = None,
    let: bool = False,
):
    # Main Python entry for linear systems (F1, CC):
    # computes max diameter over horizon, optionally returning the pipe.
    # Set let=True to use logical execution time dynamics (delayed control).
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
    if k_gain_override is None:
        k_system = ct.c2d(system, base_dt * latency_ms, method="zoh")
        k_gain, _, _ = ct.dlqr(k_system.A, k_system.B, np.eye(nx), np.eye(nu))
    else:
        k_gain = np.asarray(k_gain_override, dtype=float)

    phi_control, phi_hold = _phi_matrices(discrete, k_gain, let)
    phi = lambda k: phi_control if k % latency_ms == 0 else phi_hold

    aug_nu = 2 * nu if let else nu
    x0 = Zonotope(
        np.concatenate([x0center, np.zeros(aug_nu)]),
        np.diag(np.concatenate([x0size, np.zeros(aug_nu)])),
    )

    def w_fn(k, x):
        if k % latency_ms == 0:
            return _error_bound(k_gain, errors, x, nx, nu, relative_error, let)
        size = nx + aug_nu
        return Zonotope(np.zeros(size), np.zeros((size, size)))

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
            [
                [0.0, 1.0, 0.0, 0.0],
                [-8.0, -4.0, 8.0, 4.0],
                [0.0, 0.0, 0.0, 1.0],
                [80.0, 40.0, -160.0, -60.0],
            ],
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

    # Adaptive cruise control with lane keeping (circular lane)
    m = 1500.0
    iz = 2500.0
    lf = 1.2
    lr = 1.6
    cf = 80000.0
    cr = 100000.0
    vx = 20.0
    sys_acc_lk = ct.ss(
        np.array(
            [
                [0.0, vx, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [
                    0.0,
                    0.0,
                    -2 * (cf + cr) / (m * vx),
                    -vx - 2 * (lf * cf - lr * cr) / (m * vx),
                    0.0,
                    2 * (cf + cr) / m,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    -2 * (lf * cf - lr * cr) / (iz * vx),
                    -2 * (lf**2 * cf + lr**2 * cr) / (iz * vx),
                    0.0,
                    2 * (lf * cf - lr * cr) / iz,
                    0.0,
                    0.0,
                ],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=float,
        ),
        np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [2 * cf / m, 0.0, 2 * cr / m, 0.0],
                [2 * lf * cf / iz, 0.0, -2 * lr * cr / iz, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
            ],
            dtype=float,
        ),
        np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            ],
            dtype=float,
        ),
        np.zeros((2, 4)),
    )

    return {
        "RC": sys_rcn,
        "F1": sys_f1t,
        "DC": sys_dcm,
        "CS": sys_css,
        "EW": sys_ewb,
        "C1": sys_cc1,
        "CC": sys_cc2,
        "MPC": sys_mpc,
        "ACCLK": sys_acc_lk,
    }


def acc_lk_presets():
    # Constants ported from get_max_diam.jl for ACCLK.
    x0center = np.array([0.0, 0.0, -0.5, 0.1, 0.0, 15.0, 12.0, -2.0])
    x0size = np.full(8, 0.1)
    k_gain = np.array(
        [
            [
                3.366556150380166,
                7.1418732745720455,
                0.24940568614738273,
                0.23628994658198366,
                -0.00022746013762489062,
                0.9231605957617468,
                0.22743044093745474,
                0.08424723254105806,
            ],
            [
                0.2735664453493172,
                0.9947250357709307,
                0.042809548360928075,
                -0.009121350587273293,
                0.01900520234953239,
                7.183070543182434,
                -19.00451337413501,
                -7.171535440590174,
            ],
            [
                0.16119798742357008,
                -1.7779779689402742,
                0.12300913051753758,
                -0.20377705493439624,
                -0.00020267791720359867,
                0.8898153758821936,
                0.20264992849760982,
                0.07527943846844243,
            ],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    error_map = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    dims = [0, 6]
    return x0center, x0size, k_gain, error_map, dims
