## REK.jl

## The Random Extended Kaczmarz method.  See A Zouzias and
## NM Freris, SIAM J Matrix Anal Appl 34 (2013),
## doi:10.1137/120889897

## Still to add: convergence criterion

module REK

using LinearAlgebra

export reksolve, rpick

function reksolve(A,b;eps = 1e-6, maxcount=100)
    m,n = size(A)
    @assert m >= n

    z = copy(b)
    x = zeros(n)

    rowsums  = sum(abs2,A,dims=2)
    Fnorm2   = sum(rowsums)
    Fnorm    = sqrt(Fnorm2)
    rowprobs = rowsums / Fnorm

    colsums  = sum(abs2,A,dims=1)
    colprobs = colsums / Fnorm

    subcount = 8*min(m,n)

    for k = 1:maxcount
        for kk = 1:subcount
            i = rpick(rowprobs)
            j = rpick(colprobs)
            z .= z .- (dot(A[:,j],z)/colsums[j]) .* A[:,j]
            x .= x .+ ((b[i] - z[i]  - dot(x,A[i,:]))/rowsums[i]) .* A[i,:]
        end
        
        normx = norm(x)

        if ( norm(A*x - b - z) <= eps * Fnorm * normx &&
             norm(A'*z) <= eps * Fnorm2 * normx )
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
