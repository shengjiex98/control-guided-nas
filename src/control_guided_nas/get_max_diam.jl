
using ControlSystemsBase
using LinearAlgebra
using ReachabilityAnalysis

push!(LOAD_PATH, @__DIR__)
using Reachability
using Models

function get_max_diam(sys::StateSpace, latency::Real, errors::AbstractVector{<:Real}, 
        x0center::AbstractVector{<:Real}, x0size::AbstractVector{<:Real})
    @boundscheck length(errors) == sys.nx || 
        raise(ArgumentError("Error must have same dimensions as the system model"))
    A = c2d(sys, 0.001).A
    B = c2d(sys, 0.001).B
    K = lqr(c2d(sys, latency), I, I)

    # Perception error bound zonotope
    E = Zonotope(zeros(sys.nx + sys.nu), Diagonal([errors; zeros(sys.nu)]))

    # Closed-loop dynamics -- control
    Φc = [A B; -K zeros(sys.nu, sys.nu)]
    # Closed-loop dynamics -- hold
    Φh = [A B; zeros(sys.nu, sys.nx) I]
    Φ(k) = k % latency == 0 ? Φc : Φh

    # Overall error bound zonotope
    Wc = linear_map(
        [B * (-K) zeros(sys.nx, sys.nu); 
         zeros(sys.nu, sys.nx + sys.nu)],
        E
    ) |> remove_redundant_generators
    Wh = Zonotope(zeros(sys.nx + sys.nu), Diagonal(zeros(sys.nx + sys.nu)))
    W = (k, _) -> k % latency == 0 ? Wh : Wc

    x0 = Zonotope([x0center; zeros(sys.nu)], Diagonal([x0size; zeros(sys.nu)]))

    r = reach(Φ, x0, W, 2000)
    max_diam(r)
end

# sys = benchmarks[:F1]
# println(get_max_diam(sys, 0.001, [0.5, 0.5], [10, 10], [1, 1]))
