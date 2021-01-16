module Tests

using REK,BlockToeplitz

function test1(m=100000,n=30,r=3; flags...)
    a = randn(m,n)
    A = BTMatrix(a,r)
    b = randn(size(A)[1])

    @time solve(A,b; flags...)
end

end#module

