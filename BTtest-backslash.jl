using BTtest,LinearAlgebra,Random

BTtest.test()

include("BTtest-params.jl")

Random.seed!(42)

@time output = BTtest.test(N, n, r; method = :backslash)

println("## method: $(output.method)")
println("## est min mem: $((N-r+1)*n*r*16*1e-9) GB")
println("## $(output.iters) iterations")
println("## norm = $(norm(output.sol))")
println("## resid = $(norm(output.A*(output.sol-output.truth)))")
println("## rel err = $(norm(output.sol-output.truth)/norm(output.truth))")
println("## runtime = $(output.runtime) sec")
