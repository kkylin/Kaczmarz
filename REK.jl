## REK.jl

## The Random Extended Kaczmarz method as described in

## A Zouzias and NM Freris, SIAM J Matrix Anal Appl 34
## (2013), doi:10.1137/120889897

module REK

using LinearAlgebra

export solve, rpick

function solve(A::Matrix{T}, b::Vector{T};
               eps = 1e-12,
               maxcount=1000) where T

    m,n = size(A)

    z = copy(b)  ## we'll be modifying z and b shouldn't change
    x = zeros(T,n)

    ## probabilities for sampling rows and cols do not
    ## change
    rowsum  = sum(abs2,A,dims=2)
    rowprob = rowsum ./ sum(rowsum)

    colsum  = sum(abs2,A,dims=1)
    colprob = colsum ./ sum(colsum)

    Ac = conj(A)

    ## these are needed for the convergence test
    epsFnorm = eps * sqrt(sum(abs2,A))
    subcount = 8*min(m,n)

    ## main loop
    for k = 1:maxcount
        for kk = 1:subcount
            i = rpick(rowprob)
            j = rpick(colprob)
            z .-= (dot(A[:,j],z)/colsum[j]) .* A[:,j]
            x .+= ((b[i] - z[i] - dot(Ac[i,:],x))/rowsum[i]) .* Ac[i,:]
        end
        
        tol = epsFnorm * norm(x)

        if norm(A*x .- b .+ z) <= tol && norm(A'*z) <= tol
            return x,k*subcount
        end
    end
    return x,maxcount*subcount
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

end#module
