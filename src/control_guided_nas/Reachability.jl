module Reachability

using OffsetArrays: OffsetArray, Origin
using ReachabilityAnalysis

export reach, max_diam, get_error_bound

"""
Convert x to a function with constant return value if it is not already a function.
"""
function tofunc(x)
	x isa Function ? x : (args...) -> x
end

"""
	reach(Φ, x0, W, H; max_order=Inf, reduced_order=2, remove_redundant=true)

Compute reachable sets for the dynamics ``x[k+1] = Φ x[k] + w``, where ``w`` is a noise term bounded by `W`.  The initial state is `x0`, and the time horizon is `H`.

If `max_order` is given, we reduce order of the reachable set to `reduced_order` when it exceeds this limit.  If `remove_redundant` is true, redundant generators are removed at each step.
"""
function reach(Φ::Function, x0::LazySet, W::Function, H::Integer; max_order::Real=Inf, reduced_order::Real=2, remove_redundant::Bool=true)
	# Preallocate x vector
	x = OffsetArray(Vector{LazySet}(undef, H+1), Origin(0))
	x[0] = x0

	for k = 1:H
		x[k] = minkowski_sum(linear_map(Φ(k-1), x[k-1]), W(k-1, x[k-1]))
		if remove_redundant
			x[k] = remove_redundant_generators(x[k])
		end
		if order(x[k]) > max_order
			x[k] = reduce_order(x[k], reduced_order)
		end
	end
	
	Flowpipe([ReachSet(x_k, k) for (k, x_k) in enumerate(x)])
end
reach(Φ, x0::LazySet, W, H::Integer; kwargs...) = reach(tofunc(Φ), x0, tofunc(W), H; kwargs...)

"""
	get_error_bound(B, K, E)

Calculate the additivie error bound zonotope W from matrices B, K, and 
perception error zonotope E. The perception error zonotope E is constructed
from the maximum perception error in each state dimension.
"""
function get_error_bound(B::Matrix, K::Matrix, E::Zonotope)
	linear_map(B * (-K), E) |> remove_redundant_generators
end

end
