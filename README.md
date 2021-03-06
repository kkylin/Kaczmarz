# Randomized Extended Kaczmarz method

The code in this directory implements the Randomized
Extended Kaczmarz method (REK) method as described in

	A Zouzias and NM Freris, "Randomized extended Kaczmarz
	for solving least squares," *SIAM J. Matrix Anal. Appl.*
	**34** (2013), [doi:10.1137/120889897](https://doi.org/10.1137/120889897)

There are some toy tests.  The main test scripts are

- BTtest-backslash.jl: generate a test least-squares problem
  and solve it using '\'; and

- BTtest-kaczmarz.jl: generate a test least-squares problem
  and solve it using REK.

This implementation of REK seems reasonably fast, and has
good memory usage compared to \ (which presumably computes
an SVD).
