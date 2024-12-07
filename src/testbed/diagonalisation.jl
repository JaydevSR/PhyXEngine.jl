#? Ref: https://jaydevsr.me/computational-physics-notes/preliminaries/eigenvalue-problem.html
#? After timing:
#? - `powiter` is faster for the largest eigenvalue that `eigensolve` from KrylovKit.jl (about 5x faster)
#? - `deflation` scales worse than `eigensolve` from KrylovKit.jl so for more than one exremal eigenvalue it is better to use the latter
#? It could be possible to improve the perfomance of `deflation`

rayleigh_quotient(A::AbstractArray{T, 2}, x::Vector) where T = dot(x, A, x) / dot(x, x)

"""Calculate the dominant eigenvalue and corresponding eigenvector of a diagonalisable (eg. Hermitian) matrix"""
function powiter(A::AbstractArray{T, 2}; tol::Float64=1e-6, start_vector=nothing) where T
    n, m = size(A)
    n != m ? throw(ArgumentError("Input must be a square matrix.")) : nothing

    CT = complex(float(T))
    x = isnothing(start_vector) ? rand(CT, n) : start_vector

    normalize!(x)
    RQold = 0
    RQnew = rayleigh_quotient(A, x)

    while (abs(RQold - RQnew) > tol)
        x = normalize(A * x)
        RQold, RQnew = RQnew, rayleigh_quotient(A, x)
    end

    return RQnew, x  # eigenvalue, eigenvector
end

"""Calculate the smallest eigenvalue and corresponding eigenvector of a diagonalisable (eg. Hermitian) matrix"""
function ipowiter(A::AbstractArray{T, 2}; tol::Float64=1e-6, start_vector=nothing) where T
    n, m = size(A)
    n != m ? throw(ArgumentError("Input must be a square matrix.")) : nothing

    Ai = inv(A)
    ei, xi = powiter(Ai; tol, start_vector)
    return inv(ei), xi  # eigenvalue, eigenvector
end

"""Calculate the eigenvalue and eigenvector pair closest to a given guess eigenvalue for a diagonalisable matrix"""
function rqiter(A::AbstractArray{T, 2}, guess_eval::T; tol::Float64=1e-6, start_vector=nothing) where T
    n, m = size(A)
    n != m ? throw(ArgumentError("Input must be a square matrix.")) : nothing

    Aμ = A - guess_eval * I
    eμ, xμ = ipowiter(Aμ; tol, start_vector)
    return eμ + guess_eval, xμ  # eigenvalue, eigenvector
end


"""Calculate the first `num_vals` eigenvalues and eigenvectors of a Hermitian matrix using the method of deflation.
    
    This method uses a `solver` function to determine the particular eigenvalue eigenvector pair at each step which is 
    then projected out for the next step. Any additional arguments for the `solver` function can be passed as keyword 
    arguments.
"""
function deflation(A::Hermitian{T}, num_vals::Int; 
    solver::Function=powiter,
    kwargs...) where T
    n, m = size(A)
    num_vals > n ? throw(ArgumentError("Number of eigenvalues requested exceeds matrix size.")) : nothing

    CT = complex(float(T))
    vals = zeros(CT, num_vals)
    vecs = zeros(CT, n, num_vals)
    for i in 1:num_vals
        ei, xi = solver(A; kwargs...)
        vals[i] = ei
        vecs[:, i] = xi
        A -= ei * xi * xi'
    end

    return vals, vecs
end

