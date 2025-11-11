"""
Differentiable version of get_max_diam for neural architecture search.

This module provides automatic differentiation support for the maximum diameter
computation, enabling gradient-based optimization in PyTorch.
"""

using ControlSystemsBase
using LinearAlgebra
using ReachabilityAnalysis
using Polyhedra
using OffsetArrays: OffsetArray, Origin
using ForwardDiff

push!(LOAD_PATH, @__DIR__)
using Models

# Import constants from the original module
include("get_max_diam.jl")

"""
    softmax_reduction(values::AbstractVector{<:Real}, temperature::Real=0.01)

Smooth approximation of maximum using softmax with temperature parameter.
As temperature → 0, this approaches the true maximum.
"""
function softmax_reduction(values::AbstractVector{<:Real}, temperature::Real=0.01)
    exp_values = exp.(values ./ temperature)
    weighted_sum = sum(values .* exp_values)
    return weighted_sum / sum(exp_values)
end

"""
    smooth_max(a::Real, b::Real, temperature::Real=0.01)

Smooth approximation of max(a, b) using LogSumExp trick.
"""
function smooth_max(a::Real, b::Real, temperature::Real=0.01)
    return temperature * log(exp(a / temperature) + exp(b / temperature))
end

"""
    diam_smooth(x::LazySet, nx::Integer; dims::Union{Nothing,AbstractVector{<:Integer}}=nothing)

Differentiable diameter computation (same as original, but works with ForwardDiff).
"""
function diam_smooth(x::LazySet, nx::Integer; dims::Union{Nothing,AbstractVector{<:Integer}}=nothing)
    selected_dims = dims === nothing ? (1:nx) : dims
    2 * sum(abs.(x.generators[selected_dims,:]), dims=2) |> maximum
end

"""
    reach_smooth(Φ::Function, x0::LazySet, W::Function, H::Integer, nx::Integer;
                 max_order::Real=Inf, reduced_order::Real=2, remove_redundant::Bool=true,
                 dims::Union{Nothing,AbstractVector{<:Integer}}=nothing, temperature::Real=0.01)

Differentiable reachability analysis using smooth maximum approximation.
Returns the smooth maximum diameter over the time horizon.
"""
function reach_smooth(Φ::Function, x0::LazySet, W::Function, H::Integer, nx::Integer;
        max_order::Real=Inf, reduced_order::Real=2, remove_redundant::Bool=true,
        dims::Union{Nothing,AbstractVector{<:Integer}}=nothing, temperature::Real=0.01)

    curr_x = x0
    diameters = Float64[]

    for k = 1:H
        curr_x = minkowski_sum(linear_map(Φ(k-1), curr_x), W(k-1, curr_x))
        if remove_redundant
            curr_x = remove_redundant_generators(curr_x)
        end
        if order(curr_x) > max_order
            curr_x = reduce_order(curr_x, reduced_order)
        end
        push!(diameters, diam_smooth(curr_x, nx; dims=dims))
    end

    # Use softmax reduction for smooth maximum
    return softmax_reduction(diameters, temperature)
end

"""
    get_max_diam_diff(s::StateSpace, latency_ms::Real,
                      errors::AbstractVector{<:Real}, x0center::AbstractVector{<:Real},
                      x0size::AbstractVector{<:Real}; K::Union{Nothing,Matrix{<:Real}}=nothing,
                      dims::Union{Nothing,AbstractVector{<:Integer}}=nothing, temperature::Real=0.01)

Differentiable version of get_max_diam that works with ForwardDiff.
Uses smooth maximum approximation to enable gradient computation.

Args:
    s: State space model
    latency_ms: Control latency in milliseconds (can be Real for AD)
    errors: Error vector
    x0center: Initial state center
    x0size: Initial state size
    K: Optional controller gain matrix
    dims: Optional dimensions to compute diameter over
    temperature: Temperature for softmax approximation (smaller = closer to true max)

Returns:
    Smooth maximum diameter (differentiable w.r.t. latency_ms and errors)
"""
function get_max_diam_diff(s::StateSpace, latency_ms::Real,
    errors::AbstractVector{<:Real}, x0center::AbstractVector{<:Real},
    x0size::AbstractVector{<:Real}; K::Union{Nothing,Matrix{<:Real}}=nothing,
    dims::Union{Nothing,AbstractVector{<:Integer}}=nothing, temperature::Real=0.01)

    @boundscheck length(errors) == s.nx ||
        throw(ArgumentError("Error ($(length(errors))) must have same dimensions as the system model ($(s.nx))"))

    # Convert latency_ms to integer for modulo operations
    latency_ms_int = round(Int, latency_ms)

    A = c2d(s, 0.001).A
    B = c2d(s, 0.001).B
    if K === nothing
        K = lqr(c2d(s, 0.001 * latency_ms_int), I, I)
    end

    # Closed-loop dynamics
    Φc = [
        A B zeros(s.nx, s.nu)
        zeros(s.nu, (s.nx + s.nu)) I
        -K zeros(s.nu, 2 * s.nu)
    ]
    Φh = [
        A B zeros(s.nx, s.nu)
        zeros(s.nu, s.nx) I zeros(s.nu, s.nu)
        zeros(s.nu, s.nx + s.nu) I
    ]
    Φ(k::Integer) = k % latency_ms_int == 0 ? Φc : Φh

    # Error bound computation
    Wc(k::Integer, x::Zonotope) = let
        max_states = (sum(abs.(x.generators[1:s.nx,:]), dims=2) |> vec) + abs.(x.center[1:s.nx])
        Zonotope(
            zeros(s.nx + 2 * s.nu),
            Diagonal([zeros(s.nx); zeros(s.nu); -K * (errors .* max_states)])
        )
    end
    Wh = Zonotope(zeros(s.nx + 2 * s.nu), Diagonal(zeros(s.nx + 2 * s.nu)))
    W = (k, x) -> k % latency_ms_int == 0 ? Wc(k, x) : Wh

    x0 = Zonotope([x0center; zeros(2 * s.nu)], Diagonal([x0size; zeros(2 * s.nu)]))

    reach_smooth(Φ, x0, W, 1000, s.nx; dims=dims, temperature=temperature)
end

"""
    get_max_diam_with_grad(sysname::String, latency_ms::Real, errors::AbstractVector{<:Real};
                           temperature::Real=0.01)

Compute both the diameter and its gradients w.r.t. latency and errors.

Returns:
    (diameter, grad_latency, grad_errors)
"""
function get_max_diam_with_grad(sysname::String, latency_ms::Real, errors::AbstractVector{<:Real};
                                temperature::Real=0.01, x0center=nothing, x0size=nothing, K=nothing, dims=nothing)
    s = Models.benchmarks[Symbol(sysname)]

    # Use defaults if not provided
    if x0center === nothing
        x0center = [1.0 for _ in 1:s.nx]
    end
    if x0size === nothing
        x0size = [0.1 for _ in 1:s.nx]
    end

    # Create a function that takes a vector of all parameters
    # params = [latency_ms, errors...]
    function f(params)
        lat = params[1]
        err = params[2:end]
        return get_max_diam_diff(s, lat, err, x0center, x0size; K=K, dims=dims, temperature=temperature)
    end

    # Compute gradient using ForwardDiff
    params = [latency_ms; errors]
    grad = ForwardDiff.gradient(f, params)

    # Compute function value
    diameter = f(params)

    # Split gradient
    grad_latency = grad[1]
    grad_errors = grad[2:end]

    return (diameter, grad_latency, grad_errors)
end
