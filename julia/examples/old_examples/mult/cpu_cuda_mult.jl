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

include("../../src/Compiler/include.jl")

starpu_new_cpu_kernel_file("../build/generated_cpu_mult.c")
starpu_new_cuda_kernel_file("../build/generated_cuda_mult.cu")

@cpu_cuda_kernel function matrix_mult(m1 :: Matrix{Float32}, m2 :: Matrix{Float32}, m3 :: Matrix{Float32}) :: Void

    width_m2 :: Int64 = width(m2)
    height_m1 :: Int64 = height(m1)
    width_m1 :: Int64 = width(m1)
    A ::Float64 = abs(-4.0)
    @indep for j in (1 : width_m2)
        @indep for i in (1 : height_m1)

            sum :: Float32 = 0.

            for k in (1 : width_m1)
                sum = sum + m1[i, k] * m2[k, j]
            end

            m3[i, j] = sum
        end
    end
end

compile_cpu_kernels("../build/generated_cpu_mult.so")
compile_cuda_kernels("../build/generated_cuda_mult.so")
combine_kernel_files("../build/generated_tasks.so", ["../build/generated_cpu_mult.so", "../build/generated_cuda_mult.so"])
