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
import Libdl
using StarPU

@target STARPU_CPU+STARPU_CUDA
@codelet function black_scholes(data ::Matrix{Float64}, res ::Matrix{Float64}) :: Float32
    
    widthn ::Int64 = width(data)
        
    # data[1,...] -> S
    # data[2,...] -> K
    # data[3,...] -> r
    # data[4,...] -> T
    # data[4,...] -> sig

    p ::Float64 = 0.2316419
    b1 ::Float64 = 0.31938153
    b2 ::Float64 = -0.356563782
    b3 ::Float64 = 1.781477937
    b4 ::Float64 = -1.821255978
    b5 ::Float64 = 1.330274428

    
    @parallel for i = 1:widthn
        

        d1 ::Float64 = (log(data[1,i] / data[2,i]) + (data[3,i] + pow(data[5,i], 2.0) * 0.5) * data[4,i]) / (data[5,i] * sqrt(data[4,i]))
        d2 ::Float64 = (log(data[1,i] / data[2,i]) + (data[3,i] - pow(data[5,i], 2.0) * 0.5) * data[4,i]) / (data[5,i] * sqrt(data[4,i]))
        



        f ::Float64 = 0
        ff ::Float64 = 0
        s1 ::Float64 = 0
        s2 ::Float64 = 0
        s3 ::Float64 = 0
        s4 ::Float64 = 0
        s5 ::Float64 = 0
        sz ::Float64 = 0
        


        
        ######## Compute normcdf of d1

        normd1p ::Float64 = 0
        normd1n ::Float64 = 0

        boold1 ::Int64 = (d1 >= 0) + (d1 <= 0)
        
        if (boold1 >= 2)
            normd1p = 0.5
            normd1n = 0.5
        else
            tmp1 ::Float64 = abs(d1)
            f = 1 / sqrt(2 * M_PI)
            ff = exp(-pow(tmp1, 2.0) / 2) * f
            s1 = b1 / (1 + p * tmp1)
            s2 = b2 / pow((1 + p * tmp1), 2.0)
            s3 = b3 / pow((1 + p * tmp1), 3.0)
            s4 = b4 / pow((1 + p * tmp1), 4.0)
            s5 = b5 / pow((1 + p * tmp1), 5.0)
            sz = ff * (s1 + s2 + s3 + s4 + s5)
        
            if (d1 > 0)
                normd1p = 1 - sz # normcdf(d1)
                normd1n = sz # normcdf(-d1)
            else
                normd1p = sz
                normd1n = 1 - sz
            end    
        end
        ########
        

        ######## Compute normcdf of d2
        normd2p ::Float64 = 0
        normd2n ::Float64 = 0

        boold2 ::Int64 = (d2 >= 0) + (d2 <= 0)
        
        if (boold2 >= 2)
            normd2p = 0.5
            normd2n = 0.5
        else
            tmp2 ::Float64 = abs(d2)
            f = 1 / sqrt(2 * M_PI)
            ff = exp(-pow(tmp2, 2.0) / 2) * f
            s1 = b1 / (1 + p * tmp2)
            s2 = b2 / pow((1 + p * tmp2), 2.0)
            s3 = b3 / pow((1 + p * tmp2), 3.0)
            s4 = b4 / pow((1 + p * tmp2), 4.0)
            s5 = b5 / pow((1 + p * tmp2), 5.0)
            sz = ff * (s1 + s2 + s3 + s4 + s5)
        
        
            if (d2 > 0)
                normd2p = 1 - sz # normcdf(d2)
                normd2n = sz # normcdf(-d2)
            else
                normd2p = sz
                normd2n = 1 - sz
            end
        end
        # normd1p = (1 + erf(d1/sqrt(2.0)))/2.0
        # normd1n = (1 + erf(-d1/sqrt(2.0)))/2.0
        
        # normd2p = (1 + erf(d2/sqrt(2.0)))/2.0
        # normd2n = (1 + erf(-d2/sqrt(2.0)))/2.0
        
        res[1,i] = data[1,i] * (normd1p) - data[2,i]*exp(-data[3,i]*data[4,i]) * (normd2p) # S * N(d1) - r*exp(-r*T) * norm(d2)
        res[2,i] = -data[1,i] * (normd1n) + data[2,i]*exp(-data[3,i]*data[4,i]) * (normd2n) # -S * N(-d1) + r*exp(-r*T) * norm(-d2)
        
    end
    return 0
end

starpu_init()

function black_scholes_starpu(data ::Matrix{Float64}, res ::Matrix{Float64}, nslices ::Int64)
    vert = StarpuDataFilter(STARPU_MATRIX_FILTER_VERTICAL_BLOCK, nslices)

    @starpu_block let
        dat_handle, res_handle = starpu_data_register(data, res)

        starpu_data_partition(dat_handle, vert)
        starpu_data_partition(res_handle, vert)
        
        #Compute the price of call and put option in the res matrix
        @starpu_sync_tasks for task in (1:nslices)
            @starpu_async_cl black_scholes(dat_handle[task], res_handle[task]) [STARPU_RW, STARPU_RW] 
        end
    end
    return 0
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
    i = 1
    open("black_scholes_times.dat", "w") do f 
        for data_nbr in (start_nbr : step_nbr : stop_nbr)
            t = median_times(data_nbr, nslices, nbr_tests)
            println("Number of data:\n$data_nbr\nTimes:\njl: $t\nC: $(mtc[i])\nGen: $(mtcgen[i])")
            write(f, "$data_nbr $(t)\n")
            i = i + 1
        end
    end
end
