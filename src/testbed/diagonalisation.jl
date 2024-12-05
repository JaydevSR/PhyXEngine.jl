rayleigh_quotient(A::AbstractArray{T, 2}, x::Vector) where T = dot(x, A, x) / dot(x, x)

"""Calculate the dominant eigenvalue and corresponding eigenvector of a diagonalisable (eg. Hermitian) matrix"""
function powiter(A::AbstractArray{T, 2}; tol::Float64=1e-6, start_vector=nothing) where T
    n, m = size(A)
    n != m ? throw(ArgumentError("Input must be a square matrix.")) : nothing
    x = isnothing(start_vector) ? rand(T, n) : start_vector

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

