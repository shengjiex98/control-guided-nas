### A Pluto.jl notebook ###
# v0.19.42

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 1af7bcac-165a-11ef-12ed-dd9cfcacfedd
begin
	using Pkg
	Pkg.activate("control", shared=true)

	using ControlBenchmarks

	# Dev dependencies
	using DataFrames: DataFrame
	using CSV, LaTeXStrings, OffsetArrays, PlutoUI
	using ControlSystemsBase, Distributions, LinearAlgebra
	using ReachabilityAnalysis
	using Plots
	import PlotlyJS
	plotlyjs()
end

# ╔═╡ e3246eee-8ece-4442-85d2-100ce9d50966
md"""
## Analyzing Errors from Neural Network Distance Estimation
"""

# ╔═╡ 3ebcd854-328b-4c5f-a673-4cc7ecbfa4c8
df = let nn_accuracy_file = "../data/distance_evaluation/result_0430_xception.csv"
	DataFrame(CSV.File(nn_accuracy_file))
end

# ╔═╡ 8bb68cc3-b715-487b-8061-35d6771960ab
begin
	abs_rel_errors = df[:, "Absolute Relative Error"] |> sort
	@info "Mean: $(mean(abs_rel_errors))\n" *
		"Standard Deviation: $(std(abs_rel_errors))"
	histogram(abs_rel_errors)
end

# ╔═╡ 50e10fe2-f3f7-4401-bbe9-7015dcc8acea
@info "Ratio of errors below 1 standard deviation: $(count(x -> x < std(abs_rel_errors), abs_rel_errors)/length(abs_rel_errors))"

# ╔═╡ f0c5c66e-95a2-4780-b333-60bc84c5edb1
md"""
## System and Noise Modelling

We assume a linear model, shown as
```math
\begin{align}
x[k+1] &= Ax[k] + Bu[k] \\
u[k] &= -Kx[k]
\end{align}
```
An example system is shown below
"""

# ╔═╡ 79a9b89d-9a0c-4cfd-aa56-939c7ec5bc76
md"""
**Sign of `K` must be NEGATIVE!!**
"""

# ╔═╡ 84efe83a-4d9b-4545-b0a7-bfc587463732
md"""
## 
We further assume additive noise form:
```math
\begin{align}
    x[k+1] &\in Ax[k] \oplus (-BK\hat{x}[k]) \\
    &= Ax[k] \oplus (-BK(x[k] \oplus E)) \\
    &= (A - BK)x[k] \oplus (-BKE)
\end{align}
```
Here, ``E`` is the error bounding matrix. Assume one standard deviation of error in both dimensions, ``E_1`` is
"""

# ╔═╡ 6e02cc9f-ffbf-4af1-8437-8c3776584300
E1 = Zonotope(zeros(Float64, 2), std(abs_rel_errors) * I(2))

# ╔═╡ 55caabfa-b239-4139-af00-6c2b6257e4c4
let 
	scatter([1.], [1.], label="x", xlim=(-1, 3), ylim=(-1, 3), ratio=1.0)
	plot!([1., 1.] + E1, label="Possible x_hat values")
end

# ╔═╡ 25c304f1-d620-4d30-b6d1-2c7e3f380cd7
md"""
## Reachability for Noisy Discrete Linear Systems

In this section, we use types from the JuliaReach organization to develop a reachability algorithm for discrete linear systems with an additive noise term.  That is, the dynamics are of the form

```math
X[k+1] = Φ X[k] \oplus W
```

where ``w`` is an additive noise term bounded by a zonotope ``W``.
The parameters, Φ and W, can be obtained from the previous equations
```math
\begin{align}
Φ &= A - BK \\
W &= -BKE.
\end{align}
```
"""

# ╔═╡ 6bf891ef-f82b-4c77-bb80-c4c525fafa0f
md"""
The definition above is equivalent to ``-BKE_1``, as shown below
"""

# ╔═╡ ac82150d-f963-4e27-b137-72c96da1f9b1
md"""
We visualize ``W`` below. Note that since ``B K`` has rank ``1``, the set reduces to one-dimensional.
"""

# ╔═╡ d6fdc630-ed70-4f2c-b72c-e0d6ad5f114c
plot(E1, ratio=1.)

# ╔═╡ 78402a2c-9d6b-4995-abdd-13de3b461d21
"""
	reach(Φ, x0, W, H; max_order=Inf, reduced_order=2, remove_redundant=true)

Compute reachable sets for the dynamics ``x[k+1] = Φ x[k] + w``, where ``w`` is a noise term bounded by `W`.  The initial state is `x0`, and the time horizon is `H`.

If `max_order` is given, we reduce order of the reachable set to `reduced_order` when it exceeds this limit.  If `remove_redundant` is true, redundant generators are removed at each step.
"""
function reach(Φ::AbstractMatrix, x0::LazySet, W::LazySet, H::Integer; max_order::Real=Inf, reduced_order::Real=2, remove_redundant::Bool=true)
	# Preallocate x vector
	x = OffsetArray(Vector{LazySet}(undef, H+1), OffsetArrays.Origin(0))
	x[0] = x0

	for k = 1:H
		x[k] = minkowski_sum(linear_map(Φ, x[k-1]), W)
		if remove_redundant
			x[k] = remove_redundant_generators(x[k])
		end
		if order(x[k]) > max_order
			x[k] = reduce_order(x[k], reduced_order)
		end
	end
	
	F = Flowpipe([ReachSet(x_k, k) for (k, x_k) in enumerate(x)])
end

# ╔═╡ bc5776f3-dbfa-42c3-854a-f1c1797dad58
x0center = 10.

# ╔═╡ 52451f35-9c22-47be-9652-79685c82e2a8
function max_diam(pipe::Flowpipe)
	@info findmax([diameter(rs.X) for rs in pipe])
	maximum([diameter(rs.X) for rs in pipe])
end

# ╔═╡ b1207434-dcf9-4d45-9f36-3c3d31fa13a9
md"""
| | |
|:--|:--|
|Initial set size | $(@bind x0size Slider(LinRange(0, x0center * 0.5, 11), default=1., show_value=true)) |
|Period | $(@bind period Slider([0.01:0.01:0.09; 0.10:0.10:0.50], default=0.02, show_value=true)) |
"""

# ╔═╡ 722186e8-46c5-4a4b-b682-751c6cfb253e
A = c2d(benchmarks[:F1], period).A

# ╔═╡ 70e766a1-f175-4ee7-829b-8b9830f64d73
B = c2d(benchmarks[:F1], period).B

# ╔═╡ 3b78c63d-d9bf-4224-b28b-14595c0c79c1
K = lqr(ControlSystemsBase.Discrete, A, B, I, I)

# ╔═╡ 18cfff5e-42f1-4384-88ee-926e9fba872f
let
	x = [1.0, 1.0]
	for k in 1:100
		x = (A + -B*K) * x
	end
	@info x
end

# ╔═╡ 19d122f8-b56b-4d2d-bc8e-bd2e5f683934
Φ = A - B * K

# ╔═╡ d80ad260-87c7-4fea-a6fc-b10129a6abf9
function get_W(E::AbstractZonotope)
	linear_map(B * (-K), E) |> remove_redundant_generators
end

# ╔═╡ 89d783f0-2433-46e1-b80b-54e4e9a4dd03
W = get_W(E1)

# ╔═╡ 6b95a565-b6d1-4b94-89a1-cc8b33ae1384
plot(W, ratio=1.0)

# ╔═╡ ab2f61c5-04f6-4ec3-8710-bfc1706b974d
isequivalent(W, -B*K*E1)

# ╔═╡ 9d3601e0-f3ef-461e-87f9-6c7b5c8e1922
x0 = Zonotope(x0center * ones(2), x0size * I(2))

# ╔═╡ e0ae4f1f-09bb-4123-9d82-56fcd5625050
r = reach(Φ, x0, get_W(Zonotope([0., 0.], 0.5 * std(abs_rel_errors) * I(2))), 100)

# ╔═╡ c4b58f79-a0b4-4788-9523-66fe75daa713
let
	plt = plot(ratio=1.)
	for i in 1:101
		Plots.plot!(r[i], vars=(1, 2), color=1)
	end
	plt
end

# ╔═╡ 57d8a50f-6172-45ef-95b4-0e5bd81daedb
let
	res = DataFrame(
		stderr=Int64[],
		x_err=Float64[],
		max_diam=Float64[]
	)
	for stderr in [0, 1, 2, 3]
		E = Zonotope([0., 0.], stderr * std(abs_rel_errors) * I(2))
		push!(res, [stderr, stderr * std(abs_rel_errors), max_diam(reach(Φ, x0, get_W(E), 100))])
	end
	res
end

# ╔═╡ Cell order:
# ╠═1af7bcac-165a-11ef-12ed-dd9cfcacfedd
# ╟─e3246eee-8ece-4442-85d2-100ce9d50966
# ╠═3ebcd854-328b-4c5f-a673-4cc7ecbfa4c8
# ╟─8bb68cc3-b715-487b-8061-35d6771960ab
# ╟─50e10fe2-f3f7-4401-bbe9-7015dcc8acea
# ╟─f0c5c66e-95a2-4780-b333-60bc84c5edb1
# ╠═722186e8-46c5-4a4b-b682-751c6cfb253e
# ╠═70e766a1-f175-4ee7-829b-8b9830f64d73
# ╠═3b78c63d-d9bf-4224-b28b-14595c0c79c1
# ╟─79a9b89d-9a0c-4cfd-aa56-939c7ec5bc76
# ╠═18cfff5e-42f1-4384-88ee-926e9fba872f
# ╟─84efe83a-4d9b-4545-b0a7-bfc587463732
# ╠═6e02cc9f-ffbf-4af1-8437-8c3776584300
# ╠═55caabfa-b239-4139-af00-6c2b6257e4c4
# ╟─25c304f1-d620-4d30-b6d1-2c7e3f380cd7
# ╠═19d122f8-b56b-4d2d-bc8e-bd2e5f683934
# ╠═d80ad260-87c7-4fea-a6fc-b10129a6abf9
# ╠═89d783f0-2433-46e1-b80b-54e4e9a4dd03
# ╟─6bf891ef-f82b-4c77-bb80-c4c525fafa0f
# ╠═ab2f61c5-04f6-4ec3-8710-bfc1706b974d
# ╟─ac82150d-f963-4e27-b137-72c96da1f9b1
# ╠═d6fdc630-ed70-4f2c-b72c-e0d6ad5f114c
# ╠═6b95a565-b6d1-4b94-89a1-cc8b33ae1384
# ╠═78402a2c-9d6b-4995-abdd-13de3b461d21
# ╠═bc5776f3-dbfa-42c3-854a-f1c1797dad58
# ╠═9d3601e0-f3ef-461e-87f9-6c7b5c8e1922
# ╠═e0ae4f1f-09bb-4123-9d82-56fcd5625050
# ╠═52451f35-9c22-47be-9652-79685c82e2a8
# ╠═c4b58f79-a0b4-4788-9523-66fe75daa713
# ╟─b1207434-dcf9-4d45-9f36-3c3d31fa13a9
# ╠═57d8a50f-6172-45ef-95b4-0e5bd81daedb
