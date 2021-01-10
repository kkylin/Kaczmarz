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
    row     = map(i->conj(A[i,:]), i=1:m)
    rowsum  = map(v->sum(abs2,v), row)
    rowprob = rowsum ./ sum(rowsum)

    col     = map(j->A[:,j], j=1:n)
    colsum  = map(v->sum(abs2,v), col)
    colprob = colsum ./ sum(colsum)

    ## these are needed for the convergence test
    epsFnorm = eps * sqrt(sum(abs2,A))
    subcount = 8*min(m,n)

    ## main loop
    for k = 1:maxcount
        for kk = 1:subcount
            i = rpick(rowprob)
            j = rpick(colprob)
            z .-= (dot(col[j],z)/colsum[j]) .* col[j]
            x .+= ((b[i] - z[i] - dot(row[i],x))/rowsum[i]) .* row[i]
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
