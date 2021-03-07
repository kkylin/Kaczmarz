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
import Base:sum
import LinearAlgebra:dot,BLAS.axpby!

function dot(x::BTConj{T}, y::AbstractVector{T}) where T
    foreachrowblock(x, y; accum=sum) do xblk,yblk
        BLAS.dot(xblk,yblk)
    end
end

function sum(f::Function, x::BTConj{T}) where T
    foreachrowblock(x; accum=sum) do xblk
        sum(f,xblk)
    end
end

function axpby!(a::Number, x::BTConj{T}, b::Number, y::AbstractVector{T}) where T <:Union{Complex{Float64},Float64}
    foreachrowblock(x, y) do xblk,yblk
        BLAS.axpby!(a, xblk, b, yblk)
    end
end

function foreachrowblock(f::Function, x::BTConj{T}; accum=foreach) where T <:Union{Complex{Float64},Float64}
    i = x.v.i
    n = x.v.A.n
    r = x.v.A.r
    v = x.v.A.a
    accum(k->f(conj(view(v,i-k+r,:))),1:r)
end

function foreachrowblock(f::Function, x::BTConj{T}, y::AbstractVector{T}; accum=foreach) where T <:Union{Complex{Float64},Float64}
    i = x.v.i
    n = x.v.A.n
    r = x.v.A.r
    v = x.v.A.a
    accum(k->f(conj(view(v,i-k+r,:)),view(y,(k-1)*n+1:k*n)),1:r)
end

end #module
