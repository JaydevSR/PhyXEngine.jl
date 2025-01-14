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


# Non-quantum number conserving
N = 100
energy_rand, psi_rand = HeisenbergDMRG(N; conserve_qns=false);

# Quantum number conserving
initial_state_neel = [isodd(n) ? "Up" : "Dn" for n=1:N]  # Neel state
energy_neel, psi_neel = HeisenbergDMRG(N, initial_state_neel; conserve_qns=true);
