using BTtest,LinearAlgebra

BTtest.test()

include("BTtest-params.jl")

@time output = BTtest.test(N, n, r;
                           method = :kaczmarz,
                           eps=0.05,
                           maxcount=10,
                           )

println("## method: $(output.method)")
println("## est min mem: $(N*n*16*1e-9) GB")
println("## $(output.iters) iterations")
println("## norm = $(sqrt(output.norm2))")
println("## row-resid = $(sqrt(output.row_resid2))")
println("## col-resid = $(sqrt(output.col_resid2))")
