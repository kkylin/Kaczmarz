module BTtest

using REKBT,BlockToeplitz

Complex128 = Complex{Float64}

test(m=100,n=3,r=5; flags...) = test(Complex128,m,n,r; flags...)

function test(::Type{Float64},m=100,n=3,r=5; method = :backslash, flags...)
    A = randn(m,n)
    b = randn(m-r+1)
    A = BTMatrix(A,r)

    x     = Float64[]
    iters = 0
    norm2 = row_resid2 = col_resid2 = 0.
    
    if method === :backslash
        x = A\b
    elseif method === :kaczmarz
        x,iters,norm2,row_resid2,col_resid2 = solve(A,b; flags...)
    else
        error("unknown method $method")
    end
    return (sol = x,
            iters = iters,
            method = method,
            norm2 = norm2,
            row_resid2 = row_resid2,
            col_resid2 = col_resid2,
            )
end

function test(::Type{Complex128},m=100,n=3,r=5; method = :backslash, flags...)
    A = randn(m,n) + im*randn(m,n)
    b = randn(m-r+1) + im*randn(m-r+1)
    A = BTMatrix(A,r)

    x     = Complex128[]
    iters = 0
    norm2 = row_resid2 = col_resid2 = 0.
    
    if method === :backslash
        x = A\b
    elseif method === :kaczmarz
        x,iters,norm2,row_resid2,col_resid2 = solve(A,b; flags...)
    else
        error("unknown method $method")
    end
    return (sol = x,
            iters = iters,
            method = method,
            norm2 = norm2,
            row_resid2 = row_resid2,
            col_resid2 = col_resid2,
            )
end

end#module
