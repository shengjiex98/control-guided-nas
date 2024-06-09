
using ControlSystemsBase
using LinearAlgebra
using ReachabilityAnalysis
using NoisyReach

function get_max_diam(sys::StateSpace, latency::Real, errors::AbstractVector{<:Real}, 
        x0center::AbstractVector{<:Real}, x0size::AbstractVector{<:Real})
    @boundscheck length(errors) == sys.nx || 
        raise(ArgumentError("Error must have same dimensions as the system model"))
    A = c2d(sys, latency).A
    B = c2d(sys, latency).B
    K = lqr(ControlSystemsBase.Discrete, A, B, I, I)

    # Perception error bound zonotope
    E = Zonotope(zeros(Float64, sys.nx), Diagonal(errors))

    # Closed-loop dynamics
    Φ = A - B * K

    # Overall error bound zonotope
    W = get_error_bound(B, K, E)

    x0 = Zonotope(x0center, Diagonal(x0size))

    r = reach(Φ, x0, W, 100)
    max_diam(r)
end
