module BTtest

using REK,BlockToeplitz,LinearAlgebra

test(m=100,n=3,r=5; flags...) = test(Complex{Float64},m,n,r; flags...)

function test(::Type{Float64},m=3,n=3,r=5; flags...)
    A = randn(m,n)
    b = randn(m-r+1)
    A = BTMatrix(A,r)
    @time x0 = A\b
    @time x1,k = solve(A,b; flags...)
    (backslash = x0, kaczmarz = x1, itercount = k) 
end

function test(::Type{Complex{Float64}},m=100,n=3,r=5; flags...)
    A = randn(m,n) + im*randn(m,n)
    b = randn(m-r+1) + im*randn(m-r+1)
    A = BTMatrix(A,r)
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
