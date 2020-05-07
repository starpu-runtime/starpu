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
using Base.LinAlg
include("mult_naive.jl")

#   A of size (y,z)
#   B of size (z,x)
#   C of size (y,x)


#              |---------------|
#            z |       B       |
#              |---------------|
#       z              x
#     |----|   |---------------|
#     |    |   |               |
#     |    |   |               |
#     | A  | y |       C       |
#     |    |   |               |
#     |    |   |               |
#     |----|   |---------------|
#





function multiply_with_starpu(A :: Matrix{Float32}, B :: Matrix{Float32}, C :: Matrix{Float32}, nslicesx, nslicesy)

    vert = StarpuDataFilter(STARPU_MATRIX_FILTER_VERTICAL_BLOCK, nslicesx)
    horiz = StarpuDataFilter(STARPU_MATRIX_FILTER_BLOCK, nslicesy)

    @starpu_block let

        hA,hB,hC = starpu_data_register(A, B, C)

        starpu_data_partition(hB, vert)
        starpu_data_partition(hA, horiz)
        starpu_data_map_filters(hC, vert, horiz)

        @starpu_sync_tasks for taskx in (1 : nslicesx)
            for tasky in (1 : nslicesy)
                @starpu_async_cl cl(hA[tasky], hB[taskx], hC[taskx, tasky])
            end
        end
    end

    return nothing
end

function multiply_with_starpu_cpu(A :: Matrix{Float32}, B :: Matrix{Float32}, C :: Matrix{Float32}, nslicesx, nslicesy)

    vert = StarpuDataFilter(STARPU_MATRIX_FILTER_VERTICAL_BLOCK, nslicesx)
    horiz = StarpuDataFilter(STARPU_MATRIX_FILTER_BLOCK, nslicesy)

    @starpu_block let

        hA,hB,hC = starpu_data_register(A, B, C)

        starpu_data_partition(hB, vert)
        starpu_data_partition(hA, horiz)
        starpu_data_map_filters(hC, vert, horiz)

        @starpu_sync_tasks for taskx in (1 : nslicesx)
            for tasky in (1 : nslicesy)
                @starpu_async_cl clcpu(hA[tasky], hB[taskx], hC[taskx, tasky])
            end
        end
    end

    return nothing
end

function multiply_with_starpu_gpu(A :: Matrix{Float32}, B :: Matrix{Float32}, C :: Matrix{Float32}, nslicesx, nslicesy)

    vert = StarpuDataFilter(STARPU_MATRIX_FILTER_VERTICAL_BLOCK, nslicesx)
    horiz = StarpuDataFilter(STARPU_MATRIX_FILTER_BLOCK, nslicesy)

    @starpu_block let

        hA,hB,hC = starpu_data_register(A, B, C)

        starpu_data_partition(hB, vert)
        starpu_data_partition(hA, horiz)
        starpu_data_map_filters(hC, vert, horiz)

        @starpu_sync_tasks for taskx in (1 : nslicesx)
            for tasky in (1 : nslicesy)
                @starpu_async_cl clgpu(hA[tasky], hB[taskx], hC[taskx, tasky])
            end
        end
    end

    return nothing
end



function approximately_equals(
    A :: Matrix{Cfloat},
    B :: Matrix{Cfloat},
    eps = 1e-2
)
    (height, width) = size(A)

    for j in (1 : width)
        for i in (1 : height)
            if (abs(A[i,j] - B[i,j]) > eps * max(abs(B[i,j]), abs(A[i,j])))
                println("A[$i,$j] : $(A[i,j]), B[$i,$j] : $(B[i,j])")
                return false
            end
        end
    end

    return true
end



function median_times(nb_tests, xdim, zdim, ydim, nslicesx, nslicesy)

    exec_times_st ::Vector{Float64} = [0 for i = 1:nb_tests]
    exec_times_cpu ::Vector{Float64} = [0 for i = 1:nb_tests]
    exec_times_gpu ::Vector{Float64} = [0 for i = 1:nb_tests]
    exec_times_jl ::Vector{Float64} = [0 for i = 1:nb_tests]

    A = Array(rand(Cfloat, ydim, zdim))
    B = Array(rand(Cfloat, zdim, xdim))
    C = zeros(Float32, ydim, xdim)
    D  = A * B

    for i in (1 : nb_tests)
        
        # tic()
        # multiply_with_starpu(A, B, C, nslicesx, nslicesy)
        # t = toq()

        # if (!approximately_equals(D, C))
        #     error("Invalid st result")
        # end
        # exec_times_st[i] = t



        # tic()
        # multiply_with_starpu_cpu(A, B, C, nslicesx, nslicesy)
        # tcpu = toq()

        # if (!approximately_equals(D, C))
        #     error("Invalid cpu result")
        # end
        # exec_times_cpu[i] = tcpu



        # tic()
        # multiply_with_starpu_gpu(A, B, C, nslicesx, nslicesy)
        # tgpu = toq()

        # if (!approximately_equals(D, C))
        #     error("Invalid gpu result")
        # end
        # exec_times_gpu[i] = tgpu

        al ::Float32 = 1.0
        be ::Float32 = 0.0 

        tic()
        # multjl(A, B, C)
        BLAS.gemm!('N','N', al, A, B, be, C)
        # C = BLAS.gemm!('N', 'N', 1.0, A, B)
        tjl = toq()

        if (!approximately_equals(D, C))
            error("Invalid jl result")
        end

        exec_times_jl[i] = tjl

    end

  
    # sort!(exec_times_st)
    # sort!(exec_times_cpu)
    # sort!(exec_times_gpu)
    sort!(exec_times_jl)
  
    results ::Vector{Float64} = [exec_times_jl[1 + div(nb_tests-1, 2)]]#, exec_times_cpu[1 + div(nb_tests-1, 2)], exec_times_gpu[1 + div(nb_tests-1, 2)], exec_times_jl[1 + div(nb_tests-1, 2)]]

    return results
end



function display_times(start_dim, step_dim, stop_dim, nb_tests, nslicesx, nslicesy)
    # mtc = map( (x->parse(Float64,x)), open("DAT/mult_c.dat") do f
    #              readlines(f)
    #              end)

    # mtext = map( (x->parse(Float64,x)), open("DAT/mult_ext.dat") do f
    #              readlines(f)
    #              end)

    # mtjl = map( (x->parse(Float64,x)), open("DAT/mult_jl.dat") do f
    #             readlines(f)
    #             end)
    

    # open("../DAT/mult_ext.dat", "w") do f    
    # open("../DAT/mult_jl.dat", "w") do f
    open("../DAT/mult_jl_times.dat", "w") do ft
        # open("DAT/mult.dat", "w") do f
            # i = 1
        for dim in (start_dim : step_dim : stop_dim)
            println("Dimension: $dim")
            # println("C: $(mtc[i])")
            res ::Vector{Float64} = median_times(nb_tests, dim, dim, dim, nslicesx, nslicesy)
            println("jl: $(res[1])")
            # println("jlcpu: $(res[2])")
            # println("jlgpu: $(res[3])")
            # println("jl: $(res[4])")
            # write(f, "$(dim) $(res[4]/res[1]) $(res[4]/res[2]) $(res[4]/res[3]) $(res[4]/mtc[i])\n")
            # write(f, "$dim $(mtjl[i]/res[1]) $(mtjl[i]/mtext[i]) $(mtjl[i]/mtc[i])\n")
            # write(ft, "$(res[1]) $(mtc[i]) $(mtext[i]) $(mtjl[i])\n")
            write(ft, "$(res[1])\n")
            # i = i + 1
            # end
        end
    end
end
