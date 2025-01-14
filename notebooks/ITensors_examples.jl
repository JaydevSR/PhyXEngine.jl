### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ 37276541-0d97-406d-89af-748f1a2248cc
using ITensors

# ╔═╡ e7673cc1-606c-4356-991e-161e0cf01dce
using ITensorMPS

# ╔═╡ b7b69da7-e30b-425f-887b-0db8cffc7ae2
md"# Basic Examples"

# ╔═╡ 3f82a4c7-62c0-411a-abd4-f7dc3d59b3c2
# ╠═╡ disabled = true
#=╠═╡
# 2D SVD
let
	i = Index(10)
	j = Index(20)
	M = random_itensor(i, j)
	U, S, V = svd(M, i)
	@show M ≈ U*S*V
end
  ╠═╡ =#

# ╔═╡ f739a235-c083-4c3d-8f62-d2ae8b6d8dd9
# ╠═╡ disabled = true
#=╠═╡
#4D SVD
let
	i = Index(4)
	j = Index(4)
	k = Index(4)
	l = Index(4)
	T = random_itensor(i, j, k, l)

	U, S, V = svd(T, i, k)  # The indices i and k act as the input indices
	@show hasinds(U, i, k)
	@show hasinds(V, j, l)
	@show T ≈ U*S*V
end
  ╠═╡ =#

# ╔═╡ 09ceb413-ce3e-4ea3-a39e-2106ed4c07f7
i, j, k, l, m = Index.([10, 20, 10, 20, 20])

# ╔═╡ e11bf0d3-709d-4906-b14e-8aada86d4932
# ╠═╡ disabled = true
#=╠═╡
T = ITensor(i, j, k)
  ╠═╡ =#

# ╔═╡ ee20b5e4-ad61-4dd7-a3f8-b32ceb4c036f
# ╠═╡ disabled = true
#=╠═╡
inds(T)
  ╠═╡ =#

# ╔═╡ 72997929-2efc-4283-98fe-6c186355940c
# ╠═╡ disabled = true
#=╠═╡
# setting elements, order of indices doesn't matter
T[i=>2, j=>4, k=>2] = 3.14
  ╠═╡ =#

# ╔═╡ feca630e-0fd9-4437-a79a-8bf254fee865
# ╠═╡ disabled = true
#=╠═╡
pi_value = T[i=>2, j=>4, k=>2]
  ╠═╡ =#

# ╔═╡ 8d053767-0e18-4516-9138-e2dfa32fb382
md"# MPS and MPO"

# ╔═╡ 311626cd-8a88-439f-bc2f-4995a5736aba
# ITensor to MPS
let
	maxdim = 10
	cutoff = 1E-8
	T = random_itensor(i, j, k, l, m)
	M = MPS(T, (i,j,k,l,m); cutoff, maxdim)
end

# ╔═╡ e4a96a27-857f-4040-bde8-c7ef9b4c3eca
# Coefficient array to MPS
let
	d = 2
	N = 5
	A = randn(d^N) # or randn(d, d, ... N times)

	sites = siteinds(d, N)  # Sites for MPS
	cutoff = 1E-8
	maxdim = 10
	M = MPS(A, sites; cutoff, maxdim)
end

# ╔═╡ ded773f0-29eb-416f-abfd-56fc015c3549
# getting the coefficient of a particular state
let
	d = 2
	N = 10
	sites = siteinds(d, N)  # Sites for MPS
	chi = 5
	Ψ = random_mps(sites; linkdims=chi)

	# obtain the coefficient of the state
	st = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]

	V = ITensor(1.0)
	for j=1:N
		# @show state(sites[j], st[j])
		# state(site, phy_ind) gives a vector with
		# only 1.0 at the particular phy_ind rest zero
		V *= Ψ[j] * state(sites[j], st[j])
	end
	v = scalar(V)
end

# ╔═╡ 09998594-9260-4038-a841-248e21748840
# Expectation value of a local operator
let
	N = 10
	chi = 5
	sites = siteinds("S=1/2", N)  # Sites for MPS: spin-1/2
	Ψ = random_mps(sites; linkdims=chi)
	magz = expect(Ψ, "Sz")  # expectation value of the Sᶻ operator
	@show magz sum(magz)/N
end

# ╔═╡ 353a70ea-4455-4427-8b68-145fd09b72d2
# Expectation value of an MPO
let
	N = 10
	chi = 5
	sites = siteinds("S=1/2", N)  # Sites for MPS: spin-1/2

	state = [isodd(n) ? "Dn" : "Up" for n=1:N]
	Ψ = MPS(sites, state)

	os = OpSum()
	os += "Sz",1,"Sz",2

	XX_12 = MPO(os, sites)
	# @show XX_12
	ex_XX_12 = inner(Ψ', XX_12, Ψ)
end

# ╔═╡ 1cf92342-2040-4a26-872f-3fe228f3bd2c
# Correlation matrix
let
	N = 10
	chi = 5
	sites = siteinds("S=1/2", N)  # Sites for MPS: spin-1/2

	state = [isodd(n) ? "Dn" : "Up" for n=1:N]
	Ψ = MPS(sites, state)

	zzcorr = correlation_matrix(Ψ, "Sz", "Sz")
end

# ╔═╡ a8ec997a-49c9-4cfb-99ca-a74e8d66008c
# Entanglement Entropy
let
	N = 10
	chi = 5
	sites = siteinds("S=1/2", N)  # Sites for MPS: spin-1/2

	state = [isodd(n) ? "Dn" : "Up" for n=1:N]
	Ψ = MPS(sites, state)

	# Partition: 1:4 and 5:10
	b = 4
	orthogonalize!(Ψ, b)
	
	#           s_b                     s_b
	#            |         =             |
	#  l_b-1 ---Ψ[b]--- l_b     l_b-1 ---U-- * --S-- * --V--- l_b
	
	U, S, V = svd(Ψ[b], (linkinds(Ψ, b-1)..., siteinds(Ψ, b)...))

	SvN = 0.0
	for n=1:dim(S, 1)
		p = S[n,n]^2
		SvN -= p * log(p)
	end

	@show SvN
end

# ╔═╡ e11decd6-8fa4-4d3f-aa93-005d555051a1
md"# DMRG Calculations"

# ╔═╡ 8290c1fa-95fd-4d53-b970-c903eb721750
# Basic DMRG calculations with qn conservation
let
	N = 100  # 100 sites
	sites = siteinds("S=1", N; conserve_qns=true)  # conserve quantum numbers

	# 1D Heisenberg model
	os = OpSum()
	for j=1:N-1
		os += "Sz",j,"Sz",j+1
		os += 0.5,"S+",j,"S-",j+1
		os += 0.5,"S-",j,"S+",j+1
	end
	H = MPO(os, sites)

	# initial random MPS
	# initial state to determine the qns
	state = [isodd(n) ? "Up" : "Dn" for n in 1:N]
	ψ0 = random_mps(sites, state)

	# DMRG parameters
	nsweeps = 5
	maxdim = [10, 20, 100, 100, 200]
	cutoff = 1e-10

	# Run the algorithm
	energy, ψ = dmrg(H, ψ0; nsweeps, maxdim, cutoff)
	@show energy
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
ITensorMPS = "0d1a4710-d33b-49a5-8f18-73bdf49b47e2"
ITensors = "9136182c-28ba-11e9-034c-db9fb085ebd5"

[compat]
ITensorMPS = "~0.3.3"
ITensors = "~0.7.11"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.0"
manifest_format = "2.0"
project_hash = "dbf551e49b80c5e2eb774b6b40a433c50f605d3f"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "InverseFunctions", "LinearAlgebra", "MacroTools", "Markdown"]
git-tree-sha1 = "96bed9b1b57cf750cca50c311a197e306816a1cc"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.39"

    [deps.Accessors.extensions]
    AccessorsAxisKeysExt = "AxisKeys"
    AccessorsDatesExt = "Dates"
    AccessorsIntervalSetsExt = "IntervalSets"
    AccessorsStaticArraysExt = "StaticArrays"
    AccessorsStructArraysExt = "StructArrays"
    AccessorsTestExt = "Test"
    AccessorsUnitfulExt = "Unitful"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    Dates = "ade2ca70-3891-5945-98fb-dc099432e06a"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    Requires = "ae029012-a4dd-5104-9daa-d747884805df"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "50c3c56a52972d78e8be9fd135bfb91c9574c140"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.1.1"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.ArgCheck]]
git-tree-sha1 = "680b3b8759bd4c54052ada14e52355ab69e07876"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.4.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra"]
git-tree-sha1 = "017fcb757f8e921fb44ee063a7aafe5f89b86dd1"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.18.0"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceCUDSSExt = "CUDSS"
    ArrayInterfaceChainRulesCoreExt = "ChainRulesCore"
    ArrayInterfaceChainRulesExt = "ChainRules"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceReverseDiffExt = "ReverseDiff"
    ArrayInterfaceSparseArraysExt = "SparseArrays"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    CUDSS = "45b445bb-4962-46a0-9369-b4df9d0f772e"
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.ArrayLayouts]]
deps = ["FillArrays", "LinearAlgebra"]
git-tree-sha1 = "2bf6e01f453284cb61c312836b4680331ddfc44b"
uuid = "4c555306-a7a7-4459-81d9-ec55ddd5c99a"
version = "1.11.0"
weakdeps = ["SparseArrays"]

    [deps.ArrayLayouts.extensions]
    ArrayLayoutsSparseArraysExt = "SparseArrays"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.BangBang]]
deps = ["Accessors", "ConstructionBase", "InitialValues", "LinearAlgebra", "Requires"]
git-tree-sha1 = "e2144b631226d9eeab2d746ca8880b7ccff504ae"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.4.3"

    [deps.BangBang.extensions]
    BangBangChainRulesCoreExt = "ChainRulesCore"
    BangBangDataFramesExt = "DataFrames"
    BangBangStaticArraysExt = "StaticArrays"
    BangBangStructArraysExt = "StructArrays"
    BangBangTablesExt = "Tables"
    BangBangTypedTablesExt = "TypedTables"

    [deps.BangBang.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
    TypedTables = "9d95f2ec-7b3d-5a63-8d20-e2491e220bb9"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[deps.BitIntegers]]
deps = ["Random"]
git-tree-sha1 = "6158239ac409f960abbc232a9b24c00f5cce3108"
uuid = "c3b6d118-76ef-56ca-8cc7-ebb389d030a1"
version = "0.3.2"

[[deps.BlockArrays]]
deps = ["ArrayLayouts", "FillArrays", "LinearAlgebra"]
git-tree-sha1 = "b406207917260364a2e0287b42e4c6772cb9db88"
uuid = "8e7c35d0-a365-5155-bbbb-fb81a777f24e"
version = "1.3.0"

    [deps.BlockArrays.extensions]
    BlockArraysBandedMatricesExt = "BandedMatrices"

    [deps.BlockArrays.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "1713c74e00545bfe14605d2a2be1712de8fbcb58"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.25.1"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.CommonWorldInvalidations]]
git-tree-sha1 = "ae52d1c52048455e85a387fbee9be553ec2b68d0"
uuid = "f70d9fcc-98c5-4d4a-abd7-e4cdeebd8ca8"
version = "1.0.0"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "8ae8d32e09f0dcf42a36b90d4e17f5dd2e4c4215"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.16.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ConstructionBase]]
git-tree-sha1 = "76219f1ed5771adbb096743bff43fb5fdd4c1157"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.8"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.DefineSingletons]]
git-tree-sha1 = "0fba8b706d0178b4dc7fd44a96a92382c9065c2c"
uuid = "244e2a9f-e319-4986-a169-4d1fe445cd52"
version = "0.1.2"

[[deps.Dictionaries]]
deps = ["Indexing", "Random", "Serialization"]
git-tree-sha1 = "61ab242274c0d44412d8eab38942a49aa46de9d0"
uuid = "85a47980-9c8c-11e8-2b9f-f7ca1fa99fb4"
version = "0.4.3"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.EllipsisNotation]]
deps = ["StaticArrayInterface"]
git-tree-sha1 = "3507300d4343e8e4ad080ad24e335274c2e297a9"
uuid = "da5c29d0-fa7d-589e-88eb-ea29b0a81949"
version = "1.8.0"

[[deps.ExprTools]]
git-tree-sha1 = "27415f162e6028e81c72b82ef756bf321213b6ec"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.10"

[[deps.ExternalDocstrings]]
git-tree-sha1 = "1224740fc4d07c989949e1c1b508ebd49a65a5f6"
uuid = "e189563c-0753-4f5e-ad5c-be4293c83fb4"
version = "0.1.1"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "6a70198746448456524cb442b8af316927ff3e1a"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.13.0"

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

    [deps.FillArrays.weakdeps]
    PDMats = "90014a1f-27ba-587c-ab20-58faa44d9150"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.Folds]]
deps = ["Accessors", "BangBang", "Baselet", "DefineSingletons", "Distributed", "ExternalDocstrings", "InitialValues", "MicroCollections", "Referenceables", "Requires", "Test", "ThreadedScans", "Transducers"]
git-tree-sha1 = "7eb4bc88d8295e387a667fd43d67c157ddee76cf"
uuid = "41a02a25-b8f0-4f67-bc48-60067656b558"
version = "0.2.10"

    [deps.Folds.extensions]
    FoldsOnlineStatsBaseExt = "OnlineStatsBase"

    [deps.Folds.weakdeps]
    OnlineStatsBase = "925886fa-5bf2-5e8e-b522-a9147a512338"

[[deps.Functors]]
deps = ["Compat", "ConstructionBase", "LinearAlgebra", "Random"]
git-tree-sha1 = "60a0339f28a233601cb74468032b5c302d5067de"
uuid = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
version = "0.5.2"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.HalfIntegers]]
git-tree-sha1 = "9c3149243abb5bc0bad0431d6c4fcac0f4443c7c"
uuid = "f0d1745a-41c9-11e9-1dd9-e5d34d218721"
version = "1.6.0"

[[deps.ITensorMPS]]
deps = ["Adapt", "Compat", "ITensors", "IsApprox", "KrylovKit", "LinearAlgebra", "NDTensors", "Printf", "Random", "SerializedElementArrays", "TupleTools"]
git-tree-sha1 = "f98f5b376af969de18d58471679e420999c71112"
uuid = "0d1a4710-d33b-49a5-8f18-73bdf49b47e2"
version = "0.3.3"

    [deps.ITensorMPS.extensions]
    ITensorMPSChainRulesCoreExt = "ChainRulesCore"
    ITensorMPSHDF5Ext = "HDF5"
    ITensorMPSObserversExt = "Observers"
    ITensorMPSPackageCompilerExt = "PackageCompiler"
    ITensorMPSZygoteRulesExt = ["ChainRulesCore", "ZygoteRules"]

    [deps.ITensorMPS.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    HDF5 = "f67ccb44-e63f-5c2f-98bd-6dc0ccc4ba2f"
    Observers = "338f10d5-c7f1-4033-a7d1-f9dec39bcaa0"
    PackageCompiler = "9b87118b-4619-50d2-8e1e-99f35a4d4d9d"
    ZygoteRules = "700de1a5-db45-46bc-99cf-38207098b444"

[[deps.ITensors]]
deps = ["Adapt", "BitIntegers", "ChainRulesCore", "Compat", "Dictionaries", "DocStringExtensions", "Functors", "IsApprox", "LinearAlgebra", "NDTensors", "Pkg", "Printf", "Random", "Requires", "SerializedElementArrays", "SimpleTraits", "SparseArrays", "StaticArrays", "Strided", "TimerOutputs", "TupleTools", "Zeros"]
git-tree-sha1 = "a3068934a8e9e5199a89176169f44faa0bb1c39a"
uuid = "9136182c-28ba-11e9-034c-db9fb085ebd5"
version = "0.7.11"

    [deps.ITensors.extensions]
    ITensorsHDF5Ext = "HDF5"
    ITensorsVectorInterfaceExt = "VectorInterface"
    ITensorsZygoteRulesExt = "ZygoteRules"

    [deps.ITensors.weakdeps]
    HDF5 = "f67ccb44-e63f-5c2f-98bd-6dc0ccc4ba2f"
    VectorInterface = "409d34a3-91d5-4945-b6ec-7529ddf182d8"
    ZygoteRules = "700de1a5-db45-46bc-99cf-38207098b444"

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.Indexing]]
git-tree-sha1 = "ce1566720fd6b19ff3411404d4b977acd4814f9f"
uuid = "313cdc1a-70c2-5d6a-ae34-0150d3930a38"
version = "1.1.1"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.InlineStrings]]
git-tree-sha1 = "45521d31238e87ee9f9732561bfee12d4eebd52d"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.2"

    [deps.InlineStrings.extensions]
    ArrowTypesExt = "ArrowTypes"
    ParsersExt = "Parsers"

    [deps.InlineStrings.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"
    Parsers = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.InverseFunctions]]
git-tree-sha1 = "a779299d77cd080bf77b97535acecd73e1c5e5cb"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.17"
weakdeps = ["Dates", "Test"]

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

[[deps.IsApprox]]
deps = ["Dictionaries", "LinearAlgebra", "PrecompileTools"]
git-tree-sha1 = "597fa86ccb967c315dae711a83a234b28c0c6852"
uuid = "28f27b66-4bd8-47e7-9110-e2746eb8bed7"
version = "2.0.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.KrylovKit]]
deps = ["LinearAlgebra", "PackageExtensionCompat", "Printf", "Random", "VectorInterface"]
git-tree-sha1 = "d7ed24a88732689f26d3f12a817d181d4024bf44"
uuid = "0b1a1467-8014-51b9-945f-bf0ae24f4b77"
version = "0.8.3"
weakdeps = ["ChainRulesCore"]

    [deps.KrylovKit.extensions]
    KrylovKitChainRulesCoreExt = "ChainRulesCore"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.6.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.7.2+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "2fa9ee3e63fd3a4f7a9a4f4744a52f4856de82df"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.13"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.6+0"

[[deps.MicroCollections]]
deps = ["Accessors", "BangBang", "InitialValues"]
git-tree-sha1 = "44d32db644e84c75dab479f1bc15ee76a1a3618f"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.2.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.12.12"

[[deps.NDTensors]]
deps = ["Accessors", "Adapt", "ArrayLayouts", "BlockArrays", "Compat", "Dictionaries", "EllipsisNotation", "FillArrays", "Folds", "Functors", "HalfIntegers", "InlineStrings", "LinearAlgebra", "MacroTools", "PackageExtensionCompat", "Random", "SimpleTraits", "SparseArrays", "SplitApplyCombine", "StaticArrays", "Strided", "StridedViews", "TimerOutputs", "TupleTools", "VectorInterface"]
git-tree-sha1 = "a1df4a860b32c179ff5123639829bd3ad997f99a"
uuid = "23ae76d9-e61a-49c4-8f12-3f1a16adf9cf"
version = "0.3.74"

    [deps.NDTensors.extensions]
    NDTensorsAMDGPUExt = ["AMDGPU", "GPUArraysCore"]
    NDTensorsCUDAExt = ["CUDA", "GPUArraysCore"]
    NDTensorsGPUArraysCoreExt = "GPUArraysCore"
    NDTensorsHDF5Ext = "HDF5"
    NDTensorsJLArraysExt = ["GPUArraysCore", "JLArrays"]
    NDTensorsMappedArraysExt = ["MappedArrays"]
    NDTensorsMetalExt = ["GPUArraysCore", "Metal"]
    NDTensorsOctavianExt = "Octavian"
    NDTensorsTBLISExt = "TBLIS"
    NDTensorscuTENSORExt = "cuTENSOR"

    [deps.NDTensors.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    HDF5 = "f67ccb44-e63f-5c2f-98bd-6dc0ccc4ba2f"
    JLArrays = "27aeb0d3-9eb9-45fb-866b-73c2ecf80fcb"
    MappedArrays = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    Octavian = "6fd5a793-0b7e-452c-907f-f8bfe9c57db4"
    TBLIS = "48530278-0828-4a49-9772-0f3830dfa1e9"
    cuTENSOR = "011b41b2-24ef-40a8-b3eb-fa098493e9e1"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.27+1"

[[deps.OrderedCollections]]
git-tree-sha1 = "12f1439c4f986bb868acda6ea33ebc78e19b95ad"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.7.0"

[[deps.PackageExtensionCompat]]
git-tree-sha1 = "fb28e33b8a95c4cee25ce296c817d89cc2e53518"
uuid = "65ce6f38-6b18-4e1d-a461-8949797d7930"
version = "1.0.2"
weakdeps = ["Requires", "TOML"]

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.11.0"

    [deps.Pkg.extensions]
    REPLExt = "REPL"

    [deps.Pkg.weakdeps]
    REPL = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.Referenceables]]
deps = ["Adapt"]
git-tree-sha1 = "02d31ad62838181c1a3a5fd23a1ce5914a643601"
uuid = "42d2dcc6-99eb-4e98-b66c-637b7d73030e"
version = "0.1.3"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.SerializedElementArrays]]
deps = ["Serialization"]
git-tree-sha1 = "8e73e49eaebf73486446a3c1eede403bff259826"
uuid = "d3ce8812-9567-47e9-a7b5-65a6d70a3065"
version = "0.1.0"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "e2cc6d8c88613c05e1defb55170bf5ff211fbeac"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.1"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.11.0"

[[deps.SplitApplyCombine]]
deps = ["Dictionaries", "Indexing"]
git-tree-sha1 = "c06d695d51cfb2187e6848e98d6252df9101c588"
uuid = "03a91e81-4c3e-53e1-a0a4-9c0c8f19dd66"
version = "1.2.3"

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "e08a62abc517eb79667d0a29dc08a3b589516bb5"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.15"

[[deps.Static]]
deps = ["CommonWorldInvalidations", "IfElse", "PrecompileTools"]
git-tree-sha1 = "87d51a3ee9a4b0d2fe054bdd3fc2436258db2603"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "1.1.1"

[[deps.StaticArrayInterface]]
deps = ["ArrayInterface", "Compat", "IfElse", "LinearAlgebra", "PrecompileTools", "Static"]
git-tree-sha1 = "96381d50f1ce85f2663584c8e886a6ca97e60554"
uuid = "0d7ed370-da01-4f52-bd93-41d350b8b718"
version = "1.8.0"

    [deps.StaticArrayInterface.extensions]
    StaticArrayInterfaceOffsetArraysExt = "OffsetArrays"
    StaticArrayInterfaceStaticArraysExt = "StaticArrays"

    [deps.StaticArrayInterface.weakdeps]
    OffsetArrays = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "47091a0340a675c738b1304b58161f3b0839d454"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.10"

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

    [deps.StaticArrays.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StaticArraysCore]]
git-tree-sha1 = "192954ef1208c7019899fbf8049e717f92959682"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.3"

[[deps.Strided]]
deps = ["LinearAlgebra", "StridedViews", "TupleTools"]
git-tree-sha1 = "f9ce8284e6eec72a21de3603493eb5355fcf7f39"
uuid = "5e0ebb24-38b0-5f93-81fe-25c709ecae67"
version = "2.2.0"

[[deps.StridedViews]]
deps = ["LinearAlgebra", "PackageExtensionCompat"]
git-tree-sha1 = "b60baf1998bcdccc57e1cc2c6703df1f619a3754"
uuid = "4db3bf67-4bd7-4b4e-b153-31dc3fb37143"
version = "0.3.2"

    [deps.StridedViews.extensions]
    StridedViewsCUDAExt = "CUDA"

    [deps.StridedViews.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.7.0+0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "598cd7c1f68d1e205689b1c2fe65a9f85846f297"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.ThreadedScans]]
deps = ["ArgCheck"]
git-tree-sha1 = "ca1ba3000289eacba571aaa4efcefb642e7a1de6"
uuid = "24d252fe-5d94-4a69-83ea-56a14333d47a"
version = "0.1.0"

[[deps.TimerOutputs]]
deps = ["ExprTools", "Printf"]
git-tree-sha1 = "d7298ebdfa1654583468a487e8e83fae9d72dac3"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "0.5.26"

[[deps.Transducers]]
deps = ["Accessors", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "ConstructionBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "Requires", "SplittablesBase", "Tables"]
git-tree-sha1 = "7deeab4ff96b85c5f72c824cae53a1398da3d1cb"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.84"

    [deps.Transducers.extensions]
    TransducersAdaptExt = "Adapt"
    TransducersBlockArraysExt = "BlockArrays"
    TransducersDataFramesExt = "DataFrames"
    TransducersLazyArraysExt = "LazyArrays"
    TransducersOnlineStatsBaseExt = "OnlineStatsBase"
    TransducersReferenceablesExt = "Referenceables"

    [deps.Transducers.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    BlockArrays = "8e7c35d0-a365-5155-bbbb-fb81a777f24e"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    LazyArrays = "5078a376-72f3-5289-bfd5-ec5146d43c02"
    OnlineStatsBase = "925886fa-5bf2-5e8e-b522-a9147a512338"
    Referenceables = "42d2dcc6-99eb-4e98-b66c-637b7d73030e"

[[deps.TupleTools]]
git-tree-sha1 = "41e43b9dc950775eac654b9f845c839cd2f1821e"
uuid = "9d95972d-f1c8-5527-a6e0-b4b365fa01f6"
version = "1.6.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.VectorInterface]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "cea8abaa6e43f72f97a09cf95b80c9eb53ff75cf"
uuid = "409d34a3-91d5-4945-b6ec-7529ddf182d8"
version = "0.4.9"

[[deps.Zeros]]
deps = ["Test"]
git-tree-sha1 = "7eb4fd47c304c078425bf57da99a56606150d7d4"
uuid = "bd1ec220-6eb4-527a-9b49-e79c3db6233b"
version = "0.3.0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.59.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"
"""

# ╔═╡ Cell order:
# ╠═37276541-0d97-406d-89af-748f1a2248cc
# ╠═e7673cc1-606c-4356-991e-161e0cf01dce
# ╟─b7b69da7-e30b-425f-887b-0db8cffc7ae2
# ╠═3f82a4c7-62c0-411a-abd4-f7dc3d59b3c2
# ╠═f739a235-c083-4c3d-8f62-d2ae8b6d8dd9
# ╠═09ceb413-ce3e-4ea3-a39e-2106ed4c07f7
# ╠═e11bf0d3-709d-4906-b14e-8aada86d4932
# ╠═ee20b5e4-ad61-4dd7-a3f8-b32ceb4c036f
# ╠═72997929-2efc-4283-98fe-6c186355940c
# ╠═feca630e-0fd9-4437-a79a-8bf254fee865
# ╟─8d053767-0e18-4516-9138-e2dfa32fb382
# ╠═311626cd-8a88-439f-bc2f-4995a5736aba
# ╠═e4a96a27-857f-4040-bde8-c7ef9b4c3eca
# ╠═ded773f0-29eb-416f-abfd-56fc015c3549
# ╠═09998594-9260-4038-a841-248e21748840
# ╠═353a70ea-4455-4427-8b68-145fd09b72d2
# ╠═1cf92342-2040-4a26-872f-3fe228f3bd2c
# ╠═a8ec997a-49c9-4cfb-99ca-a74e8d66008c
# ╟─e11decd6-8fa4-4d3f-aa93-005d555051a1
# ╠═8290c1fa-95fd-4d53-b970-c903eb721750
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
