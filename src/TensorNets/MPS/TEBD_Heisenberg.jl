using ITensors
using ITensorMPS


function HeisenbergTEBD(N::Int, dt::Float64, tt::Float64, initial_state=nothing;
                        Svalue="1/2",
                        cutoff=1E-8,
                        conserve_qns=false,
                        linkdims=10)
    sites = siteinds("S=$Svalue", N; conserve_qns)

    # Gates for time evolution: Trotter-Suzuki decomposition of O(dt^3)
    gates = ITensor[]
    for j=1:N-1
        s1 = sites[j]
        s2 = sites[j+1]
        hj = op("Sz", s1) * op("Sz", s2) +
             0.5 * (op("S+", s1) * op("S-", s2) +
                    op("S-", s1) * op("S+", s2))
        gj = exp(-im * dt * hj / 2)
        push!(gates, gj)
    end
    append!(gates, reverse(gates))

    if isnothing(initial_state)
        psi = random_mps(sites; linkdims)
    else
        psi = MPS(sites, initial_state)
    end
    csite = N ÷ 2

    Szvals = Float64[]
    for t=0:dt:tt
        Sz = expect(psi, "Sz"; sites=csite)
        push!(Szvals, Sz)
        println("t = $t, <Sz> = $Sz")

        t≈tt && break

        psi = apply(gates, psi; cutoff)
        normalize!(psi)
    end

    return psi, Szvals
end

N = 100

# Time evolution
psi, Szvals = HeisenbergTEBD(N, 0.1, 7.0, initial_state_neel; conserve_qns=true);
plot(0.0:0.1:7.0, Szvals, label="Neel state", xlabel="t", ylabel="<Sz>")

psi, Szvals = HeisenbergTEBD(N, 0.1, 3.0; conserve_qns=false);
plot!(0.0:0.1:3.0, Szvals, label="Random state", xlabel="t", ylabel="<Sz>")