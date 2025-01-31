
using ControlSystemsBase
using LinearAlgebra
using ReachabilityAnalysis
using Polyhedra

push!(LOAD_PATH, @__DIR__)
using Reachability
using Models

function get_max_diam(sys::StateSpace, latency_ms::Integer, errors::AbstractVector{<:Real}, 
        x0center::AbstractVector{<:Real}, x0size::AbstractVector{<:Real})
    @boundscheck length(errors) == sys.nx || 
        throw(ArgumentError("Error ($(length(errors))) must have same dimensions as the system model ($(sys.nx))"))
    A = c2d(sys, 0.001).A
    B = c2d(sys, 0.001).B
    K = lqr(c2d(sys, 0.001 * latency_ms), I, I)

    # Closed-loop dynamics -- control
    Φc = [A B; -K zeros(sys.nu, sys.nu)]
    # Closed-loop dynamics -- hold
    Φh = [A B; zeros(sys.nu, sys.nx) I]
    Φ(k) = k % latency_ms == 0 ? Φc : Φh

    # Error bound Zonotope when calculating control input. It adds reachable
    # regions to the control input dimensions. The error bound is calculated
    # by taking the maximum absolute value of each state dimension, and 
    # multiplying the error rate with it.
    Wc(k, x) = let
        x = reduce_order(x, 2 * sys.nx)
        @info k x
        # Find vertices of the zonotope. Each col is a vertex.
        vertices_matrix = vertices_list(x) |> stack
        # Find the maximum absolute value for each state dimension
        max_states = maximum(abs.(vertices_matrix[1:sys.nx,:]), dims=2) |> vec
        Zonotope(
            zeros(sys.nx + sys.nu), 
            Diagonal([zeros(sys.nx); K * (errors .* max_states)])
        )
    end
    Wh = Zonotope(zeros(sys.nx + sys.nu), Diagonal(zeros(sys.nx + sys.nu)))
    W = (k, x) -> k % latency_ms == 0 ? Wc(k, x) : Wh

    x0 = Zonotope([x0center; zeros(sys.nu)], Diagonal([x0size; zeros(sys.nu)]))

    r = reach(Φ, x0, W, 1000)
    r, max_diam(r)
end

# sys = benchmarks[:F1]
# println(get_max_diam(sys, 0.001, [0.5, 0.5], [10, 10], [1, 1]))
