## REK.jl

## A Zouzias and NM Freris, "Randomized extended Kaczmarz
## for solving least squares," SIAM J Matrix Anal Appl 34
## (2013), doi:10.1137/120889897

## This design is perhaps too rigid.

module REK

using LinearAlgebra

export solve

function solve(A::AbstractMatrix{T},
               b::AbstractVector{T};
               eps = 1e-6,
               maxcount=1000) where T <: Number

    m,n = size(A)

    ## precompute rows, their squared sums, and
    ## corresponding probabilities
    rowsum  = map(i->sum(abs2,view(A,i,:)), 1:m)
    rowprob = rowsum ./ sum(rowsum)

    ## same for cols
    colsum  = map(j->sum(abs2,view(A,:,j)), 1:n)
    colprob = colsum ./ sum(colsum)

    ## these are needed for the convergence test
    epsFnorm = eps * sqrt(sum(abs2,A))
    subcount = 8*min(m,n)

    ## main loop
    z = copy(b)  ## we'll be modifying z and b shouldn't change
    x = zeros(T,n)

    for k = 1:maxcount
        for kk = 1:subcount
            i = rpick(rowprob)
            j = rpick(colprob)
            z .-= (dot(view(A,:,j),z)/colsum[j]) .* view(A,:,j)
            x .+= ((b[i] - z[i] - dot(conj(view(A,i,:)),x))/rowsum[i]) .* conj(view(A,i,:))
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


## tests

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
