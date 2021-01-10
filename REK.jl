## REK.jl

## The Random Extended Kaczmarz method.  See A Zouzias and
## NM Freris, SIAM J Matrix Anal Appl 34 (2013),
## doi:10.1137/120889897

## Still to add: convergence criterion

module REK

using LinearAlgebra

export reksolve, rpick

function reksolve(A,b;eps = 1e-12, maxcount=1000)
    m,n = size(A)
    @assert m >= n

    z = copy(b)
    x = zeros(n)

    rowsum  = sum(abs2,A,dims=2)
    Fnorm2  = sum(rowsum)
    Fnorm   = sqrt(Fnorm2)
    rowprob = rowsum / Fnorm2

    colsum  = sum(abs2,A,dims=1)
    colprob = colsum / Fnorm2

    @assert abs(sum(colsum)-Fnorm2) <= eps

    subcount = 8*min(m,n)

    for k = 1:maxcount
        for kk = 1:subcount
            i = rpick(rowprob)
            j = rpick(colprob)
            z .-= (dot(A[:,j],z)/colsum[j]) .* A[:,j]
            x .+= ((b[i] - z[i] - dot(x,A[i,:]))/rowsum[i]) .* A[i,:]
        end
        
        tol = eps * Fnorm * norm(x)

        if norm(A*x .- b .+ z) <= tol && norm(A'*z) <= tol
            return x,k
        end
    end
    return x,maxcount
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
