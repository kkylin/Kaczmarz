## BlockToeplitz.jl

## This design still leads to a lot of allocations.  Can be
## more efficient still:

#=

1) view() can return a lightweight object instead of
   allocating a matrix; and

2) for Kaczmarz, define a lightweight conj() object.

2) Or, rewrite Kaczmarz code to avoid view() altogether.

=#

module BlockToeplitz

struct BTMatrix{T} <: AbstractMatrix{T}
    a::AbstractMatrix{T}
    r::Int
    m::Int
    n::Int
    M::Int
    N::Int
    # row::Vector{T}
    # col::Vector{T}
end

BTMatrix(v::AbstractVector, r::Int) = BTMatrix(reshape(v,length(v),1), r)

function BTMatrix(a::AbstractMatrix{T}, r::Int) where T
    m,n = size(a)
    M = m-r+1
    N = r*n
    return BTMatrix(a, r, m, n, M, N)
    # return BTMatrix(a, r, m, n, M, N, zeros(T,N), zeros(T,M))
end

## If we don't mind allocating lots, e.g., view() returns a
## new vector or matrix every time, then we only need to
## extend view() for REK.jl to work.  OTOH if we want to be
## more efficient, then we'll also need to define getindex()
## and setindex!() for View objects.

import Base:getindex,setindex!,view,size

size(A::BTMatrix) = A.M,A.N

function getindex(A::BTMatrix, i::Int, j::Int)
    A.a[i-1-div(j-1,A.n)+A.r, (j-1)%A.n+1]
end

function setindex!(A::BTMatrix, i::Int, j::Int, rhs)
    A.a[i-1-div(j-1,A.n)+A.r, (j-1)%A.n+1] = rhs
end


## view() basically returns a submatrix

struct BTSubMatrix{T} <: AbstractMatrix{T}
    A::BTMatrix{T}
    m::Int
    n::Int
    i0::Int
    j0::Int
end

size(A::BTSubMatrix) = A.m,A.n

view(A::BTMatrix, i::Int, j::Int) = reshape([getindex(A, i, j)],1,1)

function view(A::BTMatrix, i::Int, ::Colon)
    m,n = size(A)
    BTSubMatrix(A, 1, n, i-1, 0)
end

function view(A::BTMatrix, ::Colon, j::Int)
    m,n = size(A)
    BTSubMatrix(A, m, 1, 0, j-1)
end

function getindex(A::BTSubMatrix, i::Int, j::Int)
    A.A[A.i0+i,A.j0+j]
end

function setindex!(A::BTSubMatrix, i::Int, j::Int, rhs)
    A.A[A.i0+i,A.j0+j] = rhs
end

end #module
