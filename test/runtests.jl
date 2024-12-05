using Test
using PhyXEngine
using LinearAlgebra

@testset "Diagonalisation" begin
    Arand = rand(100, 100)
    Asym = Hermitian((Arand .+ transpose(Arand)) ./ 2)

    eigensys = eigen(Asym)
    domei = maximum(abs.(eigensys.values))
    smali = minimum(abs.(eigensys.values))

    tol = 1e-8
    domval, domev = PhyXEngine.powiter(Asym; tol=tol/10)
    smaval, smaev = PhyXEngine.ipowiter(Asym; tol=tol/10)

    @test isapprox(abs(domval), domei; atol=tol)
    @test isapprox(abs(smaval), smali; atol=tol)

    # Test deflation
    num_vals = 5
    max_evals = sort(abs.(eigensys.values), rev=true)[1:num_vals]

    vals, vecs = PhyXEngine.deflation(Asym, num_vals; solver=PhyXEngine.powiter, tol=tol/10)
    for i in 1:num_vals
        @test isapprox(abs(vals[i]), max_evals[i]; atol=1e-6)
    end
end
