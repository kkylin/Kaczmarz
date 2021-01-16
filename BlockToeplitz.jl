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

export BTMatrix

struct BTMatrix{T} <: AbstractMatrix{T}
    a::AbstractMatrix{T}
    r::Int
    m::Int
    n::Int
    M::Int
    N::Int
end

BTMatrix(v::AbstractVector, r::Int) = BTMatrix(reshape(v,length(v),1), r)

function BTMatrix(a::AbstractMatrix{T}, r::Int) where T
    m,n = size(a)
    M = m-r+1
    N = r*n
    return BTMatrix(a, r, m, n, M, N)
end

## If we don't mind allocating lots, e.g., view() returns a
## new vector or matrix every time, then we only need to
## extend view() for REK.jl to work.  OTOH if we want to be
## more efficient, then we'll also need to define getindex()
## and setindex!() for View objects.

import Base:conj,getindex,setindex!,size,view

size(A::BTMatrix) = A.M,A.N

function getindex(A::BTMatrix{T}, i::Int, j::Int)::T where T
    A.a[i-1-div(j-1,A.n)+A.r, (j-1)%A.n+1]
end



## column vectors
struct BTCol{T} <: AbstractVector{T}
    A::BTMatrix{T}
    m::Int
    j::Int
end

size(A::BTCol) = (A.m,)

function view(A::BTMatrix, ::Colon, j::Int)
    BTCol(A, size(A)[1], j)
end

function getindex(A::BTCol{T}, i::Int)::T where T
    A.A[i,A.j]
end

## row vectors
struct BTRow{T} <: AbstractVector{T}
    A::BTMatrix{T}
    n::Int
    i::Int
end

size(A::BTRow) = (A.n,)

function view(A::BTMatrix, i::Int, ::Colon)
    BTRow(A, size(A)[2], i)
end

function getindex(A::BTRow{T}, j::Int)::T where T
    A.A[A.i,j]
end

## conjugate
struct BTConj{T} <: AbstractVector{T}
    v::Union{BTRow{T},BTCol{T}}
end

size(A::BTConj) = size(A.v)
function getindex(A::BTConj{T}, i::Int)::T where T 
    conj(A.v[i])
end

conj(v::BTRow) = BTConj(v)
conj(v::BTCol) = BTConj(v)

end #module
