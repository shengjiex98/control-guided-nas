module Reachability

using OffsetArrays: OffsetArray, Origin
using ReachabilityAnalysis

export reach, max_diam, get_error_bound

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
