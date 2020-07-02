# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2020       Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
#
# StarPU is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation; either version 2.1 of the License, or (at
# your option) any later version.
#
# StarPU is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
# See the GNU Lesser General Public License in COPYING.LGPL for more details.
#
using LinearAlgebra

function check(mat::Matrix{Float32})
    size_p = size(mat, 1)

    for i in 1:size_p
        for j in 1:size_p
            if j < i
                mat[i, j] = 0.0f0
            end
        end
    end

    test_mat ::Matrix{Float32} = zeros(Float32, size_p, size_p)

    BLAS.syrk!('L', 'T', 1.0f0, mat, 0.0f0, test_mat)

    for i in 1:size_p
        for j in 1:size_p
            if j <= i
                orig = (1.0f0/(1.0f0+(i-1)+(j-1))) + ((i == j) ? 1.0f0*size_p : 0.0f0)
                err = abs(test_mat[i,j] - orig) / orig
                if err > 0.0001
                    got = test_mat[i,j]
                    expected = orig
                    error("[$i, $j] -> $got != $expected (err $err)")
                end
            end
        end
    end

    println(stderr, "Verification successful !")
end

function main(size_p :: Int; verify = false, verbose = false)
    mat = zeros(Float32, size_p, size_p)
    # create a simple definite positive symetric matrix
    # Hilbert matrix h(i,j) = 1/(i+j+1)

    for i in 1:size_p
        for j in 1:size_p
            mat[i, j] = 1.0f0 / (1.0f0+(i-1)+(j-1)) + ((i == j) ? 1.0f0*size_p : 0.0f0)
        end
    end

    if verbose
        display(mat)
    end

    t_start = time_ns()

    cholesky!(mat)

    t_end = time_ns()

    flop = (1.0*size_p*size_p*size_p)/3.0
    time_ms = (t_end-t_start) / 1e6
    gflops = flop/(time_ms*1000)/1000
    println("$size_p\t$time_ms\t$gflops")

    if verbose
        display(mat)
    end

    if verify
        check(mat)
    end
end

println("# size\tms\tGFlops")

if length(ARGS) > 0 && ARGS[1] == "-quickcheck"
    main(1024, verify = true)
else
    for size in 1024:1024:15360
        main(size)
    end
end

