module BTtest

using Kaczmarz,BlockToeplitz

Complex128 = Complex{Float64}

test(m=100,n=3,r=5; flags...) = test(Complex128,m,n,r; flags...)

function test(::Type{Float64},m=100,n=3,r=5; method = :backslash, flags...)

    X = randn(m,n)
    c = rand(n*r)

    A = BTMatrix(X,r)
    b = A*c

    x     = Float64[]
    iters = 0
    norm2 = row_resid2 = col_resid2 = 0.

    tstart = time()
    
    if method === :backslash
        x = A\b
    elseif method === :kaczmarz
        x,outercount,iters,norm2,row_resid2,col_resid2 = solve(A,b; flags...)
    else
        error("unknown method $method")
    end
    return (sol = x,
            truth = c,
            A = A,
            b = b,
            iters = iters,
            method = method,
            norm2 = norm2,
            row_resid2 = row_resid2,
            col_resid2 = col_resid2,
            runtime = time()-tstart,
            )
end

function test(::Type{Complex128},m=100,n=3,r=5; method = :backslash, flags...)

    X = randn(m,n) + im*randn(m,n)
    c = rand(n*r) + im*rand(n*r)

    A = BTMatrix(X,r)
    b = A*c

    x     = Complex128[]
    iters = 0
    norm2 = row_resid2 = col_resid2 = 0.
    
    tstart = time()
    
    if method === :backslash
        x = A\b
    elseif method === :kaczmarz
        x,outercount,iters,norm2,row_resid2,col_resid2 = solve(A,b; flags...)
    else
        error("unknown method $method")
    end
    return (sol = x,
            truth = c,
            A = A,
            b = b,
            iters = iters,
            method = method,
            norm2 = norm2,
            row_resid2 = row_resid2,
            col_resid2 = col_resid2,
            runtime = time()-tstart,
            )
end

end#module
