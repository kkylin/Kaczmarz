######################################################
## BlockToeplitz.jl

## Given an m-vector A, this provides a representation of
## the size-(m,r) Toeplitz matrix of the form

## A[r]   A[r-1] ⋯  A[2]    A[1]
## A[r+1] A[r]   ⋯  A[3]    A[2]
## A[r+2] A[r+1] ⋯  A[4]    A[3]
##   ⋮      ⋮     ⋱    ⋮        ⋮
## A[m]   A[m-1] ⋯ A[m-r+2] A[m-r+1]

## The goal is to enable solving least squares problems of
## the form A*x = b for given n-vectors b, with n=m-r+1.  If
## A is a stationary time series, then
## A[k]*x[1]+A[k-1]*x[2]+⋯+A[k-r+1]*x[r] is the optimal
## linear estimate of b[k] using A[k],A[k-1],⋯,A[k-r+1].

## The construct also handles block Toeplitz matrices in a
## similar fashion.


################################
## data structures

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
    @assert M>0
    @assert N>0
    return BTMatrix(a, r, m, n, M, N)
end

################################
## define methods necessary for linear solve

import Base:conj,getindex,setindex!,size,view

size(A::BTMatrix) = A.M,A.N

function getindex(A::BTMatrix{T}, i::Int, j::Int)::T where T
    A.a[i-1-div(j-1,A.n)+A.r, rem(j-1,A.n)+1]
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
