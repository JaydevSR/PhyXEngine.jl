using ITensors
using ITensorMPS


function HeisenbergDMRG(N::Int, initial_state=nothing;
                        Svalue=1/2,
                        nsweeps=5,
                        cutoff=[1E-10],
                        maxdim=[10, 20, 100, 100, 200],
                        conserve_qns=false,
                        linkdims=10)
    sites = siteinds("S=$Svalue", N; conserve_qns)

    op = OpSum()
    for j=1:N-1
        op += "Sz", j, "Sz", j+1
        op += 0.5, "S+", j, "S-", j+1
        op += 0.5, "S-", j, "S+", j+1
    end
    Ham = MPO(op, sites)

    if isnothing(initial_state)
        psi0 = random_mps(sites; linkdims)
    else
        psi0 = MPS(sites, initial_state)
    end

    energy, psi = dmrg(Ham, psi0; nsweeps, cutoff, maxdim)

    return energy, psi
end


function HeisenbergTEBD(N::Int, initial_state=nothing;
                        Svalue="1/2",
                        dt=0.1,
                        tt=5.0,
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

# Non-quantum number conserving
N = 100
energy_rand, psi_rand = HeisenbergDMRG(N; conserve_qns=false);

# Quantum number conserving
initial_state_neel = [isodd(n) ? "Up" : "Dn" for n=1:N]  # Neel state
energy_neel, psi_neel = HeisenbergDMRG(N, initial_state_neel; conserve_qns=true);

# Time evolution
psi, Szvals = HeisenbergTEBD(N, initial_state_neel; conserve_qns=true);
plot(0.0:0.1:5.0, Szvals, label="Neel state", xlabel="t", ylabel="<Sz>")