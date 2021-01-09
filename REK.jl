## REK.jl

## The Random Extended Kaczmarz method.  See A Zouzias and
## NM Freris, SIAM J Matrix Anal Appl 34 (2013),
## doi:10.1137/120889897

module REK

using LinearAlgebra

export reksolve, rpick

function reksolve(A,b;maxcount=100)
    m,n = size(A)
    @assert m >= n

    z = copy(b)
    x = zeros(n)

    rowsums  = sum(abs2,A,dims=2)
    Fnorm    = sum(rowsums)
    rowprobs = rowsums / Fnorm

    colsums  = sum(abs2,A,dims=1)
    colprobs = colsums / Fnorm

    for k = 1:maxcount
        i = rpick(rowprobs)
        j = rpick(colprobs)
        z = z - dot(A[:,j],z)/colsums[j]*A[:,j]
        x = x + (b[i] - z[i]  - dot(x,A[i,:]))/rowsums[i]*A[i,:]
    end
    return x
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
