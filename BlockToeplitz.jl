## BlockToeplitz.jl

## This design still leads to a lot of allocations.  Can be
## more efficient still:

#=

1) view() can return a lightweight object instead of
   allocating a matrix; and

2) for Kaczmarz, can define a lightweight conj() object.

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

getindex(A::BTMatrix, i::Int, j::Int) = A.a[i-1-div(j-1,A.n)+A.r, (j-1)%A.n+1]

function setindex!(A::BTMatrix, i::Int, j::Int, rhs)
    A.a[i-1-div(j-1,A.n)+A.r, (j-1)%A.n+1] = rhs
end

view(A::BTMatrix, i::Int, j::Int) = getindex(A, i, j)

function view(A::BTMatrix, i::Int, ::Colon)
    map(j->A[i,j], 1:A.N)
    # for j=1:A.N
    #     A.row[j] = A[i,j]
    # end
    # return A.row
end

function view(A::BTMatrix, ::Colon, j::Int)
    map(i->A[i,j], 1:A.M)
    # for i=1:A.M
    #     A.col[i] = A[i,j]
    # end
    # return A.col
end

# function view(A::BTMatrix, ::Colon, ::Colon)
#     [A[i,j] for i=1:A.M,j=1:A.N]
# end

end #module
