using BTtest,LinearAlgebra

include("BTtest-params.jl")

output = BTtest.test(N, n, r; method = :kaczmarz)

println("## method: $(output.method)")
println("## $(output.iters) iterations")
println("## norm = $(norm(output.sol))")
