# Randomized Extended Kaczmarz method

[Kaczmarz.jl](https://github.com/kkylin/Kaczmarz/blob/github/Kaczmarz.jl) implements the Randomized Extended Kaczmarz
method (REK) method as described in

> A Zouzias and NM Freris, "Randomized extended Kaczmarz for
solving least squares," *SIAM J. Matrix Anal. Appl.*  **34**
(2013),
[doi:10.1137/120889897](https://doi.org/10.1137/120889897)
and [arXiv:1205.5770](https://arxiv.org/abs/1205.5770).

This code is the result of my effort to understand the
algorithm.  There are some toy tests, but it has not been
extensively tested.  Similarly, I tried to pay some
attention to speed and memory, but there's no doubt more
room for optimization.  The code can be a little cleaner,
and better commented.

This repository also contains [BlockToeplitz.jl](https://github.com/kkylin/Kaczmarz/blob/github/BlockToeplitz.jl), a minimal,
memory-efficient data structure for block Toeplitz matrices
designed to work with [Kaczmarz.jl](https://github.com/kkylin/Kaczmarz/blob/github/Kaczmarz.jl). The two can operate
independently, e.g., you can use [Kaczmarz.jl](https://github.com/kkylin/Kaczmarz/blob/github/Kaczmarz.jl) with whatever
matrix structures you like.

The main test scripts are

- [BTtest-backslash.jl](https://github.com/kkylin/Kaczmarz/blob/github/BTtest-backslash.jl): generate a test least-squares problem
  and solve it using Julia's `\`;
  and

- [BTtest-kaczmarz.jl](https://github.com/kkylin/Kaczmarz/blob/github/BTtest-kaczmarz.jl): generate a test least-squares problem
  and solve it using REK.

For Toeplitz problems, this implementation of REK seems
reasonably fast, and only requires storing the block column
matrix.

[Kevin K Lin](https://math.arizona.edu/~klin)
