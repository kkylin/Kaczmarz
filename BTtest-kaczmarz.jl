using BTtest,LinearAlgebra

BTtest.test()

include("BTtest-params.jl")

@time output = BTtest.test(N, n, r; method = :kaczmarz)

println("## method: $(output.method)")
println("## $(output.iters) iterations")
println("## norm = $(norm(output.sol))")
