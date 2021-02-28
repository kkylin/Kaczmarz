# Randomized Extended Kaczmarz method

The code in this directory implements the Randomized
Extended Kaczmarz method (REK) method as described in

A Zouzias and NM Freris, "Randomized extended Kaczmarz for
solving least squares," SIAM J Matrix Anal Appl 34 (2013),
doi:10.1137/120889897

There are some toy tests.  The main test scripts are

- BTtest-backslash.jl: generate a test least-squares problem
  and solve it using '\'; and

- BTtest-kaczmarz.jl: generate a test least-squares problem
  and solve it using REK.

On my old machine, REK is very slow: it computes a solution
to about 1% relative accuracy after 12 or 13 minutes.
Backslash (which presumably computes a SVD) does it in about
a minute.  But backslash takes 2+ GB, whereas REK uses about
300MB.  For larger problems this memory-speed trade-off may
be worthwhile.
