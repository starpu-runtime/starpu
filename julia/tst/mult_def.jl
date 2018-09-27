

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



function median_time(nb_tests, xdim, zdim, ydim, nslicesx, nslicesy)

    exec_times = Float64[]

    for i in (1 : nb_tests)

        A = Array(rand(Cfloat, ydim, zdim))
        B = Array(rand(Cfloat, zdim, xdim))
        C = zeros(Float32, ydim, xdim)
        D  = A * B

        tic()
        multiply_with_starpu(A, B, C, nslicesx, nslicesy)
        t = toq()

        if (!approximately_equals(D, C))
            error("Invalid result")
        end

        push!(exec_times, t)
    end

    sort!(exec_times)

    return exec_times[1 + div(nb_tests-1, 2)]
end



function display_times(start_dim, step_dim, stop_dim, nb_tests, nslicesx, nslicesy)

    for dim in (start_dim : step_dim : stop_dim)
        mt = median_time(nb_tests, dim, dim, dim, nslicesx, nslicesy)
        println("$dim ; $mt")
    end
end
