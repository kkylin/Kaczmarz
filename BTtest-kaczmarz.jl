using BTtest,LinearAlgebra,Random

BTtest.test()

include("BTtest-params.jl")

Random.seed!(42)

@time output = BTtest.test(N, n, r;
                           method = :kaczmarz,
                           eps=0.01,  ## 1% rel err
                           maxcount=100,
                           )

println("## method: $(output.method)")
println("## est min mem: $(N*n*16*1e-9) GB")
println("## $(output.iters) iterations")
println("## norm = $(sqrt(output.norm2))")
println("## row-resid = $(sqrt(output.row_resid2))")
println("## col-resid = $(sqrt(output.col_resid2))")
println("## resid = $(norm(output.A*(output.sol-output.truth)))")
println("## rel err = $(norm(output.sol-output.truth)/norm(output.truth))")
