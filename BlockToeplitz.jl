######################################################
## BlockToeplitz.jl

## This file implements a simple block Toeplitz matrix data
## structure.  It is meant for use with Kaczmarz.jl.

# Copyright (C) 2021 by Kevin K Lin <klin@math.arizona.edu>

# This program is free software; you can redistribute it
# and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation;
# either version 2 of the License, or (at your option) any
# later version.

# This program is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE.  See the GNU General Public License for more
# details.

# You should have received a copy of the GNU General Public
# License along with this program; if not, write to the Free
# Software Foundation, Inc., 51 Franklin Street, Fifth
# Floor, Boston, MA 02110-1301 USA.

######################################################
## Description: given an m-vector A, this provides a
## representation of the size-(m,r) Toeplitz matrix of the
## form

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

## This code is neither minimal (it implements more than is
## needed for Kaczmarz) nor complete (it does not implement
## all matrix operations).  But should be good enough for
## now.


################################
## data structures

module BlockToeplitz

include("Kaczmarz.jl")

using LinearAlgebra

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

## this is slow
function getindex(A::BTMatrix{T}, i::Int, j::Int)::T where T
    A.a[i-1-div(j-1,A.n)+A.r, rem(j-1,A.n)+1]
end

function conj(A::BTMatrix{T}) where T
    BTMatrix(conj(A.a), A.r, A.m, A.n, A.M, A.N)
end

################################
## column vectors

## Kaczmarz.jl relies on taking views of columns.  This is
## straightforward, since columns of the block Toeplitz
## matrix correspond to contiguous sub-columns in the
## original matrix.

function view(A::BTMatrix, ::Colon, j::Int)
    m = A.M
    n = A.n
    r = A.r
    a = A.a

    jj     = rem(j-1,n)+1
    ishift = -1-div(j-1,n)+r

    return view(a, (1+ishift):(m+ishift), jj)
end

################################
## row vectors

## To avoid copying, this means building custom view
## operations.
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
    v::BTRow{T}
end

size(A::BTConj) = size(A.v)
function getindex(A::BTConj{T}, i::Int)::T where T 
    conj(A.v[i])
end

conj(v::BTRow) = BTConj(v)

## Custom dot(), sum(), etc, for row vectors, to avoid
## repeatedly calling the (slow) getindex().  This is
## probably not as big a deal as the performance we get from
## having fast column operations, however, since columns
## tend to be much larger than rows in least squares
## problems.

## TODO: This may have some room for improvement,
## performance-wise.  One possiblity is changing the order
## foreachrowblock() loops through the original matrix to
## improve locality: right now, the block number k=1:r is
## the outer loop, and the inner loop goes across the rows
## of the original A.  But Julia uses column-major format,
## and if we make k=1:r the inner loop then entries of A are
## referenced sequentially.  Of course in operations like
## dot(), what we gain in locality on one argument, we'll
## lose on the other.  So maybe not worth the trouble.
import .Kaczmarz:sumabs2
import LinearAlgebra:dot,BLAS.axpy!

function dot(x::BTConj{T}, y::AbstractVector{T}) where T
    foreachrowblock(sum, x, y) do xblk,yblk
        BLAS.dotu(xblk,yblk)
    end
end

function axpy!(a::Number, x::BTConj{T}, y::AbstractVector{T}) where T <:Union{Complex{Float64},Float64}
    foreachrowblock(x,y) do xblk,yblk
        ## the conj() is allocating
        BLAS.axpy!(a, conj(xblk), yblk)
    end
end

## this is the one Kaczmarz-specific optimization
function sumabs2(x::Union{BTRow{T},BTConj{T}}) where T
    foreachrowblock(sum, x) do xblk
        sum(abs2,xblk)
    end
end

foreachrowblock(f::Function, x::BTConj{T}) where T <:Union{Complex{Float64},Float64} = foreachrowblock(f, foreach, x)

function foreachrowblock(f::Function, accum::Function, x::BTConj{T}) where T <:Union{Complex{Float64},Float64}
    r = x.v.A.r
    A = x.v.A.a
    i = x.v.i
    n = x.v.A.n
    accum(k->f(view(A,i-k+r,:)),1:r)
end

foreachrowblock(f::Function, x::BTConj{T}, y::AbstractVector{T}) where T <:Union{Complex{Float64},Float64} = foreachrowblock(f, foreach, x, y)

function foreachrowblock(f::Function, accum::Function, x::BTConj{T}, y::AbstractVector{T}) where T <:Union{Complex{Float64},Float64}
    r = x.v.A.r
    A = x.v.A.a
    i = x.v.i
    n = x.v.A.n
    accum(k->f(view(A,i-k+r,:),view(y,(k-1)*n+1:k*n)),1:r)
end

end #module
