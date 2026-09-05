# Production timing and memory of one sheet label (production preset 1/64 + 0.02, carrier 1/32, Tet10, Pardiso)

Measured on the 64-core / 377 GB development box with four labels running at once, 8 threads each
(`/usr/bin/time -v`; CPU share is what the job actually got under that contention):

| cell | rho | tets | fine Tet10 dof | q_active | surface+Gmsh | port | Schur | total wall | peak RSS | CPU share |
|---|---|---|---|---|---|---|---|---|---|---|
| G0 uncut | 0.229 | 106.6k | 539k | 5643 | 23 s | 36 s | 92 s | 3 min 09 s | 19.2 GB | 309 % |
| thin uncut | 0.100 | 68.2k | 380k | 3744 | 22 s | 27 s | 41 s | 2 min 06 s | 13.0 GB | 212 % |
| thick uncut | 0.500 | 171.4k | 814k | 10101 | 32 s | 60 s | 314 s | 7 min 43 s | 31.6 GB | 472 % |
| G5 cut (rho 0.019) | 0.019 | 8.4k | 46k | 882 | 4 s | 3 s | 3 s | 30 s | 1.4 GB | 146 % |

Peak memory is the Pardiso factorization of the internal block plus the dense layout Schur (18438^2 doubles = 2.7 GB
on a tau = 0.4 cell, 30294^2 = 7.3 GB on the thick cell): roughly 35 MB per 1000 fine dof.
