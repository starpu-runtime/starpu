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
include("mandelbrot.jl")


function mandelbrot_with_starpu(A ::Matrix{Int64}, params ::Vector{Float64}, nslicesx ::Int64, nslicesy ::Int64) #mettre params en matrice. (pour que starpu le traite en matrice et non vecteur)
    vert = StarpuDataFilter(STARPU_MATRIX_FILTER_VERTICAL_BLOCK, nslicesy)
    horiz = StarpuDataFilter(STARPU_MATRIX_FILTER_BLOCK, nslicesx)
    @starpu_block let
	hA = starpu_data_register(A)
        hP = starpu_data_register(params)
	starpu_data_map_filters(hA, vert, horiz)
        
	@starpu_sync_tasks for tasky in (1:nslicesy)
            for taskx in (1 : nslicesx)
                @starpu_block let
                    v = Int64[tasky, taskx] #C'est le x qu'on augmente en fonction du nombre de slicey. Si il y a trois colonnes, x sera coupé en 3. Donc on inverse dans v.
                    hV = starpu_data_register(v)
                    @starpu_async_cl cl(hA[tasky, taskx], hP, hV)
	        end
            end
        end
    end
    return nothing
end

function mandelbrot_with_starpu_cpu(A ::Matrix{Int64}, params ::Vector{Float64}, nslicesx ::Int64, nslicesy ::Int64) #mettre params en matrice. (pour que starpu le traite en matrice et non vecteur)
    vert = StarpuDataFilter(STARPU_MATRIX_FILTER_VERTICAL_BLOCK, nslicesy)
    horiz = StarpuDataFilter(STARPU_MATRIX_FILTER_BLOCK, nslicesx)
    @starpu_block let
	hA = starpu_data_register(A)
        hP = starpu_data_register(params)
	starpu_data_map_filters(hA, vert, horiz)
        
	@starpu_sync_tasks for tasky in (1:nslicesy)
            for taskx in (1 : nslicesx)
                v = Int64[tasky, taskx] #C'est le x qu'on augmente en fonction du nombre de slicey. Si il y a trois colonnes, x sera coupé en 3. Donc on inverse dans v.
                hV = starpu_data_register(v)
                @starpu_async_cl clcpu(hA[tasky, taskx], hP, hV)
	    end
        end
    end
    return nothing
end

function mandelbrot_with_starpu_gpu(A ::Matrix{Int64}, params ::Vector{Float64}, nslicesx ::Int64, nslicesy ::Int64) #mettre params en matrice. (pour que starpu le traite en matrice et non vecteur)
    vert = StarpuDataFilter(STARPU_MATRIX_FILTER_VERTICAL_BLOCK, nslicesx)
    horiz = StarpuDataFilter(STARPU_MATRIX_FILTER_BLOCK, nslicesy)
    @starpu_block let
	hA = starpu_data_register(A)
        hP = starpu_data_register(params)
	starpu_data_map_filters(hA, vert, horiz)
        
	@starpu_sync_tasks for taskx in (1:nslicesx)
            for tasky in (1 : nslicesy)
                v = Int64[taskx, tasky] #C'est le x qu'on augmente en fonction du nombre de slicey. Si il y a trois colonnes, x sera coupé en 3. Donc on inverse dans v.
                hV = starpu_data_register(v)
                @starpu_async_cl clgpu(hA[taskx, tasky], hP, hV)
	    end
        end
    end
    return nothing
end

function init_zero(Pixels ::Matrix{Int64}, width ::Int64, height ::Int64)
    for i in 1:height
        for j in 1:width
            Pixels[i,j] = 0
        end
    end
end

function graph_pixels(Pixels ::Matrix{Int64}, width ::Int64, height ::Int64, filename ::String)
    open(filename, "w") do f
        write(f, "P3\n$width $height\n255\n")
        for i = 1:height
            for j = 1:width
                write(f, "$(Pixels[i,j]) 0 0 ")
            end
            write(f, "\n")
        end
    end
end

function median_times(nbr_tests ::Int64, cr ::Float64, ci ::Float64, dim ::Int64, nslices ::Int64)

    exec_times_st ::Vector{Float64} = [0 for i = 1:nbr_tests]
    exec_times_cpu ::Vector{Float64} = [0 for i = 1:nbr_tests]
    exec_times_gpu ::Vector{Float64} = [0 for i = 1:nbr_tests]
    exec_times_jl ::Vector{Float64} = [0 for i = 1:nbr_tests]
    
    Pixels_st ::Matrix{Int64} = zeros(dim, dim)
    Pixels_cpu ::Matrix{Int64} = copy(Pixels_st)
    Pixels_gpu ::Matrix{Int64} = copy(Pixels_st)
    Pixels_jl ::Matrix{Int64} = copy(Pixels_st)
    
    max_iter ::Float64 = (dim/2) * 0.049715909 * log10(dim * 0.25296875)


    params = [cr, ci, dim, dim, max_iter]

    for i = 1:nbr_tests
        
        tic()
        mandelbrot_with_starpu(Pixels_st, params, nslices, nslices)
        t = toq()
        
        
        exec_times_st[i] = t
        
        # tic()
        # mandelbrot_with_starpu_cpu(Pixels_cpu, params, nslices, nslices)
        # t = toq()
        

        # exec_times_cpu[i] = t        

        # tic()
        # mandelbrot_with_starpu_gpu(Pixels_gpu, params, nslices, nslices)
        # t = toq()
        
        
        # exec_times_gpu[i] = t


        # tic()
        # mandelbrotjl(Pixels_jl, cr, ci)
        # t = toq()
        
        
        # exec_times_jl[i] = t
     end
    # graph_pixels(Pixels_st, dim, dim, "../PPM/mandelbrotst$(dim).ppm")
    # graph_pixels(Pixels_cpu, dim, dim, "../PPM/mandelbrotcpu$(dim).ppm")
    # graph_pixels(Pixels_gpu, dim, dim, "../PPM/mandelbrotgpu$(dim).ppm")
    # graph_pixels(Pixels_jl, dim, dim, "../PPM/mandelbrotjl$(dim).ppm")

    sort!(exec_times_st)
    # sort!(exec_times_cpu)
    # sort!(exec_times_gpu)
    # sort!(exec_times_jl)

    
    results ::Vector{Float64} = [exec_times_st[1 + div(nbr_tests-1, 2)]]
    # results ::Vector{Float64} = [exec_times_st[1 + div(nbr_tests-1, 2)]]#, exec_times_cpu[1 + div(nbr_tests-1, 2)], exec_times_gpu[1 + div(nbr_tests-1, 2)]]#, exec_times_jl[1 + div(nbr_tests-1, 2)]]
    return results
end

function display_time(cr ::Float64, ci ::Float64, start_dim ::Int64, step_dim ::Int64, stop_dim ::Int64, nslices ::Int64, nbr_tests ::Int64)
    # mtc = map( (x->parse(Float64,x)), open("../DAT/mandelbrot_c.dat") do f
    #             readlines(f)
    #             end)

    # mtgen = map( (x->parse(Float64,x)), open("../DAT/mandelbrot_with_generated_times.dat") do f
    #             readlines(f)
    #             end)

    mtjl = map( (x->parse(Float64,x)), open("../DAT/mandelbrot_jl_times.dat") do f
                    readlines(f)
                    end)

    # mtjlcpu = map( (x->parse(Float64,x)), open("../DAT/mandelbrot_jl_cpu.dat") do f
    #             readlines(f)
    #             end)

    mtstruct = map( (x->parse(Float64,x)), open("../DAT/mandelbrot_c_struct_times.dat") do f
                    readlines(f)
                    end)


    mtarray = map( (x->parse(Float64,x)), open("../DAT/mandelbrot_c_array_times.dat") do f
                    readlines(f)
                    end)


    i = 1
    # open("../DAT/mandelbrot.dat", "w") do f
    # open("../DAT/mandelbrot_gen_times.dat", "w") do ft
    open("../DAT/mandelbrot_speedups.dat", "w") do f
        for dim in (start_dim : step_dim : stop_dim)
            println("Dimension: $dim")
            
            res ::Vector{Float64} = median_times(nbr_tests, cr, ci, dim, nslices)
            # println("C: $(mtc[i])")
            # println("C with generated: $(mtgen[i])")
            # println("Julia with starpu: $(res[1])")
            # println("cpu: $(res[2])")
            println("c_struct: $(mtstruct[i])")
            println("c_array: $(mtarray[i])")
            println("jl_st: $(res[1])")
            # println("cpu: $(res[1])")
            # write(ft, "$(dim) $(res[1]) $(mtgen[i])\n")
            # write(f, "$(dim) $(res[4]/res[1]) $(res[4]/res[2]) $(res[4]/res[3]) $(res[4]/mtc[i])\n")
            write(f, "$(dim) $(mtjl[i]/res[1]) $(mtjl[i]/mtstruct[i]) $(mtjl[i]/mtarray[i])\n")
            # write(f, "$(res[1])\n")
            i = i + 1
        end
    end
end
