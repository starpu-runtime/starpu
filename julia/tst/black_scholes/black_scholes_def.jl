function black_scholes_starpu(data ::Matrix{Float64}, res ::Matrix{Float64}, nslices ::Int64)
    vert = StarpuDataFilter(STARPU_MATRIX_FILTER_VERTICAL_BLOCK, nslices)

    @starpu_block let
        dat_handle, res_handle = starpu_data_register(data, res)

        starpu_data_partition(dat_handle, vert)
        starpu_data_partition(res_handle, vert)
        
        #Compute the price of call and put option in the res matrix
        @starpu_sync_tasks for task in (1:nslices)
            @starpu_async_cl cl(dat_handle[task], res_handle[task])
        end
    end
end


function init_data(data, data_nbr);
    for i in 1:data_nbr
        data[1,i] = rand(Float64) * 100
        data[2,i] = rand(Float64) * 100
        data[3,i] = rand(Float64)
        data[4,i] = rand(Float64) * 10
        data[5,i] = rand(Float64) * 10
    end
    return data
end
        


function median_times(data_nbr, nslices, nbr_tests)

    data ::Matrix{Float64} = zeros(5, data_nbr)
    # data[1,1] = 100.0
    # data[2,1] = 100.0
    # data[3,1] = 0.05
    # data[4,1] = 1.0
    # data[5,1] = 0.2


    res ::Matrix{Float64} = zeros(2, data_nbr)

    exec_times ::Vector{Float64} = [0. for i in 1:nbr_tests]

    for i = 1:nbr_tests
        
        init_data(data, data_nbr)

        tic()
        black_scholes_starpu(data, res, nslices);
        t = toq()

        exec_times[i] = t
    end
    sort!(exec_times)
    # println(data)
    # println(res)
    
    return exec_times[1 + div(nbr_tests - 1, 2)]
end

function display_times(start_nbr, step_nbr, stop_nbr, nslices, nbr_tests)

    mtc = map( (x->parse(Float64,x)), open("../DAT/black_scholes_c_times.dat") do f
                  readlines(f)
                  end)


    mtcgen = map( (x->parse(Float64,x)), open("../DAT/black_scholes_c_generated_times.dat") do f
                  readlines(f)
                  end)
    i = 1
    open("../DAT/black_scholes_times.dat", "w") do f 
        for data_nbr in (start_nbr : step_nbr : stop_nbr)
            t = median_times(data_nbr, nslices, nbr_tests)
            println("Number of data:\n$data_nbr\nTimes:\njl: $t\nC: $(mtc[i])\nGen: $(mtcgen[i])")
            write(f, "$data_nbr $(t) $(mtcgen[i]) $(mtc[i])\n")
            i = i + 1
        end
    end
end