
using ControlSystemsBase
using LinearAlgebra
using ReachabilityAnalysis
using Polyhedra
using OffsetArrays: OffsetArray, Origin

push!(LOAD_PATH, @__DIR__)
using Models

function get_max_diam(s::StateSpace, latency_ms::Integer, errors::AbstractVector{<:Real}, 
        x0center::AbstractVector{<:Real}, x0size::AbstractVector{<:Real}; return_pipe::Bool=true)
    @boundscheck length(errors) == s.nx || 
        throw(ArgumentError("Error ($(length(errors))) must have same dimensions as the system model ($(s.nx))"))
    A = c2d(s, 0.001).A
    B = c2d(s, 0.001).B
    K = lqr(c2d(s, 0.001 * latency_ms), I, I)

    # Closed-loop dynamics -- control
    Φc = [A B; -K zeros(s.nu, s.nu)]
    # Closed-loop dynamics -- hold
    Φh = [A B; zeros(s.nu, s.nx) I]
    Φ(k::Integer) = k % latency_ms == 0 ? Φc : Φh

    # Error bound Zonotope when calculating control input. It adds reachable
    # regions to the control input dimensions. The error bound is calculated
    # by taking the maximum absolute value of each state dimension, and 
    # multiplying the error rate with it.
    Wc(k::Integer, x::Zonotope) = let
        # Use the generaotr matrix of the zonotope to calculate the maximum states
        max_states = sum(abs.(x.generators[1:s.nx,:]), dims=2) |> vec
        Zonotope(
            zeros(s.nx + s.nu), 
            Diagonal([zeros(s.nx); K * (errors .* max_states)])
        )
    end
    Wh = Zonotope(zeros(s.nx + s.nu), Diagonal(zeros(s.nx + s.nu)))
    W = (k, x) -> k % latency_ms == 0 ? Wc(k, x) : Wh

    x0 = Zonotope([x0center; zeros(s.nu)], Diagonal([x0size; zeros(s.nu)]))

    reach(Φ, x0, W, 1000, s.nx, return_pipe=return_pipe)
end

"""
	reach(Φ, x0, W, H; max_order=Inf, reduced_order=2, remove_redundant=true)

Compute reachable sets for the dynamics ``x[k+1] = Φ x[k] + w``, where ``w`` is a noise term bounded by `W`.  The initial state is `x0`, and the time horizon is `H`.

If `max_order` is given, we reduce order of the reachable set to `reduced_order` when it exceeds this limit.  If `remove_redundant` is true, redundant generators are removed at each step.
"""
function reach(Φ::Function, x0::LazySet, W::Function, H::Integer, nx::Integer; 
        max_order::Real=Inf, reduced_order::Real=2, remove_redundant::Bool=true, return_pipe::Bool=false)
	if return_pipe
		x = OffsetArray(Vector{LazySet}(undef, H+1), Origin(0))
		x[0] = x0
	end

	curr_x, maxdiam = x0, 0.
	for k = 1:H
		curr_x = minkowski_sum(linear_map(Φ(k-1), curr_x), W(k-1, curr_x))
		if remove_redundant
			curr_x = remove_redundant_generators(curr_x)
		end
		if order(curr_x) > max_order
			curr_x = reduce_order(curr_x, reduced_order)
		end
		if return_pipe
			x[k] = curr_x
		end
        maxdiam = max(maxdiam, diam(curr_x, nx))
	end
	
	return return_pipe ? (maxdiam, Flowpipe([ReachSet(x_k, k) for (k, x_k) in enumerate(x)])) : (maxdiam,)
end
reach(Φ, x0::LazySet, W, H::Integer, nx::Integer; kwargs...) = reach(tofunc(Φ), x0, tofunc(W), H, nx; kwargs...)

"""
Convert x to a function with constant return value if it is not already a function.
"""
function tofunc(x)
	x isa Function ? x : (args...) -> x
end

"""
	max_diam(pipe, nx)

Return the maximum diameter of reachable sets in a Flowpipe.
"""
function max_diam(pipe::Flowpipe, nx::Integer)
	[diam(rs.X, nx) for rs in pipe] |> maximum
end

function diam(x::LazySet, nx::Integer)
    2 * sum(abs.(x.generators[1:nx,:]), dims=2) |> maximum
end

# s = benchmarks[:F1]
# println(get_max_diam(s, 0.001, [0.5, 0.5], [10, 10], [1, 1]))
