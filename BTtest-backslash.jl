using BTtest,LinearAlgebra

BTtest.test()

include("BTtest-params.jl")

@time output = BTtest.test(N, n, r; method = :backslash)

println("## method: $(output.method)")
println("## est min mem: $(N*n*(N-r+1)*16*1e-9) GB")
println("## $(output.iters) iterations")
println("## norm = $(norm(output.sol))")
