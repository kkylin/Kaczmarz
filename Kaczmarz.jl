######################################################
## Kaczmarz.jl

## This file implements the Randomized extended Kaczmarz
## algorithm as described in

## A Zouzias and NM Freris, "Randomized extended Kaczmarz
## for solving least squares," SIAM J Matrix Anal Appl 34
## (2013), doi:10.1137/120889897

## with minor extendsions to handle complex inputs.

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

module Kaczmarz

include("Util.jl")

using LinearAlgebra,.Util

export solve

function solve(A::AbstractMatrix{T},
               b::AbstractVector{T};
               eps          = 1e-6, ## relative error tolerance
               maxcount     = 1000, ## max number outer loops
               delay        = 10,   ## report freq, in sec
               verbose      = true,
               verbosity    = verbose ? 1 : 0,
               reportperiod = 10.0, # sec
               ) where T <: Number

    m,n = size(A)

    ## precompute rows, their squared sums, and
    ## corresponding probabilities
    row      = map(i->conj(view(A,i,:)), 1:m)
    rowsum   = map(sumabs2, row)
    rowtotal = sum(rowsum)
    rowprob  = rowtotal > 0 ? rowsum ./ rowtotal : rowsum

    ## same for cols
    col     = map(j->view(A,:,j), 1:n)
    colsum  = map(sumabs2, col)
    coltotal = sum(colsum)
    colprob = coltotal > 0 ? colsum ./ coltotal : colsum
    
    ## these are needed for the convergence test
    Asum      = max(sum(colsum),sum(rowsum))
    epsFnorm2 = eps^2 * Asum

    ## the paper suggests this (presumably) because the
    ## error check is actually pretty expensive
    subcount = 8*min(m,n)  

    verbose && @show (m,n,subcount,Asum)

    ## main loop
    z = copy(b)  ## we'll be modifying z and b shouldn't change
    x = zeros(T,n)

    norm2 = row_resid2 = col_resid2 = 0.

    ## progress report
    if verbose
        update = TimeReporter(maxcount*subcount;
                              tag="Kaczmarz",
                              period=reportperiod)
    else
        update = ()->nothing
    end

    ## The algorithm alternates between moving the right
    ## hand side (the b in Ax=b) closer to col(A), and
    ## solving for x.
    for c=1:maxcount
        for cc=1:subcount

            ## make z orthogonal to col[j]
            j = rpick(colprob)
            BLAS.axpy!(-dot(col[j],z)/colsum[j], col[j], z)

            ## project x onto the hyperplane {x|dot(row[i],x)=b[i]-z[i]}
            i = rpick(rowprob)
            BLAS.axpy!((b[i] - z[i] - dot(row[i],x)) / rowsum[i], row[i], x)

            ## progress report
            update()
        end

        ## don't check too often as the error estimates are
        ## expensive
        norm2 = sumabs2(x)
        row_resid2 = sum(i->abs2(dot(row[i],x) - b[i] + z[i]), 1:m)
        col_resid2 = sum(j->abs2(dot(col[j],z)), 1:n)
        threshold  = epsFnorm2*norm2

        if verbosity >= 2
            println("\nouter loop ", c)
            @show norm2
            @show row_resid2
            @show col_resid2
            @show threshold
        end
        
        if ( row_resid2 <= threshold && col_resid2 <= threshold )
            verbose && println("#Kaczmarz: early exit")
            return ( sol        = x,
                     outercount = c,
                     innercount = c*subcount,
                     norm2      = norm2,
                     row_resid2 = row_resid2,
                     col_resid2 = col_resid2,
                     threshold  = threshold,
                     )
        end
    end
    verbose && println("#Kaczmarz: $maxcount outer loops reached")
    return ( sol        = x,
             outercount = maxcount,
             innercount = maxcount*subcount,
             norm2      = norm2,
             row_resid2 = row_resid2,
             col_resid2 = col_resid2,
             threshold  = threshold,
             )
end

## Users can provide their own implementation of this to
## improve performance.  See BlockToeplitz.jl for an
## example.
sumabs2(x::AbstractVector) = sum(abs2,x)

################################
## some simple tests

test(m=3,n=3; flags...) = test(Complex{Float64},m,n; flags...)

function test(::Type{Float64},m=3,n=3; flags...)
    A = randn(m,n)
    b = randn(m)
    @time x0 = A\b
    @time x1,k = solve(A,b; flags...)
    (
        backslash = x0,
        kaczmarz = x1,
        err = norm(backslash-kaczmarz),
        itercount = k,
     )
end

function test(::Type{Complex{Float64}},m=3,n=3; flags...)
    A = randn(m,n) + im*randn(m,n)
    b = randn(m) + im*randn(m)
    @time backslash = A\b
    @time kaczmarz,count = solve(A,b; flags...)
    print("## ")
    @show maximum(abs,backslash-kaczmarz)
    print("## ")
    @show norm(backslash)
    print("## ")
    @show norm(kaczmarz)
    (
        backslash = backslash,
        kaczmarz = kaczmarz,
        err = norm(backslash-kaczmarz),
        count = count,
    ) 
end

end#module
