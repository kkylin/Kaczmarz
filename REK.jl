######################################################
## REK.jl

## This file implements the algorithm proposed in

## A Zouzias and NM Freris, "Randomized extended Kaczmarz
## for solving least squares," SIAM J Matrix Anal Appl 34
## (2013), doi:10.1137/120889897

## It has been extended slightly to handle complex vectors.

module REK

using LinearAlgebra,Util

export solve

function solve(A::AbstractMatrix{T},
               b::AbstractVector{T};
               eps      = 1e-6,  ## relative error tolerance
               maxcount = 1000,
               delay    = 10,  ## report freq, in sec
               ) where T <: Number

    m,n = size(A)

    ## precompute rows, their squared sums, and
    ## corresponding probabilities
    row     = map(i->conj(view(A,i,:)), 1:m)
    rowsum  = map(r->sum(abs2,r), row)
    Asum    = sum(rowsum)
    rowprob = rowsum ./ Asum

    ## same for cols
    col     = map(j->view(A,:,j), 1:n)
    colsum  = map(c->sum(abs2,c), col)
    colprob = colsum ./ Asum
    
    ## these are needed for the convergence test
    epsFnorm2 = eps^2 * Asum
    subcount = 8*min(m,n)

    ## main loop
    z = copy(b)  ## we'll be modifying z and b shouldn't change
    x = zeros(T,n)

    iabs2(i) = abs2(dot(row[i],x) - b[i] + z[i])
    jabs2(j) = abs2(dot(col[j],z))

    norm2 = row_resid2 = col_resid2 = 0.

    ## In the paper, the algorithm is formulated as a pair
    ## of nested loops.  Here I have unrolled the loops so
    ## that ETA is calculated correctly.
    foreach(1:(maxcount*subcount), "REK"; delay=delay) do loopcount
        c  = div(loopcount,subcount)
        cc = rem(loopcount,subcount)

        if cc > 0
            i = rpick(rowprob)
            j = rpick(colprob)

            z .-= dot(col[j],z)/colsum[j] .* col[j]
            x .+= (b[i] - z[i] - dot(row[i],x)) / rowsum[i] .* row[i]
        else
            ## don't check too often
            norm2 = sum(abs2,x)
            row_resid2 = sum(iabs2,1:m)
            col_resid2 = sum(jabs2,1:n)
            
            if ( row_resid2 <= epsFnorm2*norm2 &&
                 col_resid2 <= epsFnorm2*norm2 )
                return x,loopcount,norm2,row_resid2,col_resid2
            end
        end
    end
    return x,maxcount*subcount,norm2,row_resid2,col_resid2
end

function rpick(probs)
    u = rand()
    s = 0.0
    for i = 1:length(probs)
        s += probs[i]
        if u <= s
            return i
        end
    end
    return length(probs)
end

################################
## some simple tests

test(m=3,n=3; flags...) = test(Complex{Float64},m,n; flags...)

function test(::Type{Float64},m=3,n=3; flags...)
    A = randn(m,n)
    b = randn(m)
    @time x0 = A\b
    @time x1,k = solve(A,b; flags...)
    (backslash = x0, kaczmarz = x1, itercount = k) 
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
    (backslash = backslash, kaczmarz = kaczmarz, count = count) 
end

end#module
