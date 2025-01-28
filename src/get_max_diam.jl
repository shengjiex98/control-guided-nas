
using ControlSystemsBase
using LinearAlgebra
using ReachabilityAnalysis

using .reachability: reach, max_diam
using .models: benchmarks

function get_max_diam(sys::StateSpace, latency::Real, errors::AbstractVector{<:Real}, 
        x0center::AbstractVector{<:Real}, x0size::AbstractVector{<:Real})
    @boundscheck length(errors) == sys.nx || 
        raise(ArgumentError("Error must have same dimensions as the system model"))
    A = c2d(sys, 1).A
    B = c2d(sys, 1).B
    K = lqr(ControlSystemsBase.Discrete, A, B, I, I)

    # Perception error bound zonotope
    E = Zonotope(zeros(sys.nx + sys.nu), Diagonal([errors; zeros(sys.nu)]))

    # Closed-loop dynamics -- control
    Φc = [A B; -K zeros(sys.nu)]
    # Closed-loop dynamics -- hold
    Φh = [A B; zeros(sys.nx) I]
    ϕ(k) = k % latency == 0 ? Φc : Φh

    # Overall error bound zonotope
    Wc = linear_map(
        [B * (-K) zeros(sys.nx, sys.nu); 
         zeros(sys.nu, sys.nx + sys.nu)],
        E
    ) |> remove_redundant_generators
    Wh = Zonotope(zeros(sys.nx + sys.nu), Diagonal(zeros(sys.nx + sys.nu)))
    W(k) = k % latency == 0 ? Wh : Wc

    x0 = Zonotope([x0center; zeros(sys.nu)], Diagonal([x0size; zeros(sys.nu)]))

    r = reach(Φ, x0, W, 100)
    max_diam(r)
end
