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
#include("./nbody_display.jl")
include("nbody.jl")
function nbody_with_starpu(positions ::Matrix{Float64}, velocities ::Matrix{Float64}, accelerations ::Matrix{Float64}, masses ::Vector{Float64}, parameters ::Vector{Float64}, nbr_simulations ::Int64, nslices ::Int64, nbr_planets ::Int64)
    
    vert = StarpuDataFilter(STARPU_MATRIX_FILTER_VERTICAL_BLOCK, nslices)

    for i = 1:nbr_simulations
 
        @starpu_block let
            hPOS, hVEL, hACC, hPAR, hMA = starpu_data_register(positions,velocities,accelerations,parameters,masses)
            
            starpu_data_partition(hACC,vert)
            @starpu_sync_tasks for task in (1:nslices)
                @starpu_block let
                    id = Int64[task-1]
                    hID = starpu_data_register(id)
                    @starpu_async_cl claccst(hPOS,hACC[task],hMA,hPAR,hID)
                end
            end
            
            starpu_data_partition(hPOS,vert)
            starpu_data_partition(hVEL,vert)
            
            @starpu_sync_tasks for task in (1:nslices)
                @starpu_async_cl clupdtst(hPOS[task],hVEL[task],hACC[task],hPAR)
            end
            
        end
        # pixels ::Array{Array{Int64}} = [[0,0,0] for i = 1:1000*1000]
        # graph_planets(pixels, positions, -4E8, 4E8, 1000, 1000, "PPM/nbody_st$(nbr_planets)_$i.ppm")
    end
    return nothing
end



function nbody_with_starpu_cpu(positions ::Matrix{Float64}, velocities ::Matrix{Float64}, accelerations ::Matrix{Float64}, masses ::Vector{Float64}, parameters ::Vector{Float64}, nbr_simulations ::Int64, nslices ::Int64)

    vert = StarpuDataFilter(STARPU_MATRIX_FILTER_VERTICAL_BLOCK, nslices)
 
    for i = 1:nbr_simulations

        @starpu_block let
            hPOS, hVEL, hACC, hPAR, hMA = starpu_data_register(positions,velocities,accelerations,parameters,masses)
            
            starpu_data_partition(hACC,vert)
            @starpu_sync_tasks for task in (1:nslices)
                id = Int64[task-1]
                hID = starpu_data_register(id)
                @starpu_async_cl clacccpu(hPOS,hACC[task],hMA,hPAR,hID)
            end
            
            starpu_data_partition(hPOS,vert)
            starpu_data_partition(hVEL,vert)
            
            @starpu_sync_tasks for task in (1:nslices)
                @starpu_async_cl clupdtcpu(hPOS[task],hVEL[task],hACC[task],hPAR)
            end
            
        end
        
        # pixels ::Array{Array{Int64}} = [[0,0,0] for i = 1:1000*1000]
        # graph_planets(pixels, positions, -4E8, 4E8, 1000, 1000, "PPM/nbody_cpu$i.ppm")
    end
    return nothing
end



function nbody_with_starpu_gpu(positions ::Matrix{Float64}, velocities ::Matrix{Float64}, accelerations ::Matrix{Float64}, masses ::Vector{Float64}, parameters ::Vector{Float64}, nbr_simulations ::Int64, nslices ::Int64)

    vert = StarpuDataFilter(STARPU_MATRIX_FILTER_VERTICAL_BLOCK, nslices)

    for i = 1:nbr_simulations

        @starpu_block let
            hPOS, hVEL, hACC, hPAR, hMA = starpu_data_register(positions,velocities,accelerations,parameters,masses)
            
            starpu_data_partition(hACC,vert)
            @starpu_sync_tasks for task in (1:nslices)
                id = Int64[task-1]
                hID = starpu_data_register(id)
                @starpu_async_cl claccgpu(hPOS,hACC[task],hMA,hPAR,hID)
            end
            
            starpu_data_partition(hPOS,vert)
            starpu_data_partition(hVEL,vert)
            
            @starpu_sync_tasks for task in (1:nslices)
                @starpu_async_cl clupdtgpu(hPOS[task],hVEL[task],hACC[task],hPAR)
            end
        end       
    
        # pixels ::Array{Array{Int64}} = [[0,0,0] for i = 1:1000*1000]
        # graph_planets(pixels, positions, -4E8, 4E8, 1000, 1000, "PPM/nbody_gpu$i.ppm")
    end
    return nothing
end



# function display_times(starting_nbr_planets::Int64, step_nbr ::Int64, nbr_steps ::Int64, nbr_simulations::Int64, nb_tests ::Int64, nslices::Int64)
#     width = 1000
#     height = 1000

#     times_starpu ::Vector{Float64} = [0 for i = 1:nbr_steps+1]
#     times_julia ::Vector{Float64} = [0 for i = 1:nbr_steps+1]
    
#     for k = 0:nbr_steps
#         nbr_planets ::Int64 = starting_nbr_planets + k * step_nbr
#         println("Number of planets: $nbr_planets")

#         epsilon ::Float64 = 2.5E8
#         dt ::Float64 = 36000
#         G ::Float64 = 6.67408E-11
#         parameters ::Vector{Float64} = [G,dt,epsilon]
        
#         maxcoord ::Int64 = 2E8
#         mincoord ::Int64 = -2E8
        
#         positions ::Matrix{Float64} = zeros(2, nbr_planets)
#         velocities ::Matrix{Float64} = zeros(2, nbr_planets)
#         accelerations ::Matrix{Float64} = zeros(2,nbr_planets)
        
#         positionsjl ::Vector{Vector{Float64}} = [[0, 0] for i = 1:nbr_planets]
#         velocitiesjl ::Vector{Vector{Float64}} = [[0, 0] for i = 1:nbr_planets]
#         accelerationsjl ::Vector{Vector{Float64}} = [[0, 0] for i = 1:nbr_planets]
        
        
#         masses ::Vector{Float64} = [0 for i = 1:nbr_planets]
        
        
        

#         for i = 1:nbr_planets
#             mi ::Float64 = rand() * 5E22
#             ri = mi * 2.5E-15
            
#             angle ::Float64 = rand() * 2 * pi
#             distToCenter ::Float64 = rand() * 1.0E8 + 1.0E8
            
#             qix ::Float64 = cos(angle) * distToCenter
#             qiy ::Float64 = sin(angle) * distToCenter
            
#             vix ::Float64 = qiy * 4.0E-6
#             viy ::Float64 = -qix * 4.0E-6
            
#             masses[i] = mi
        
#             positions[1,i] = qix
#             positions[2,i] = qiy
            
#             positionsjl[i] = [qix, qiy]
            
#             velocities[1,i] = vix
#             velocities[2,i] = viy
            
#             velocitiesjl[i] = [vix, viy]
#         end
        
#         max_value = 4E8
#         min_value = -4E8
        
#         println("Starpu...")
#         tic()
#         for i = 1:nbr_simulations
#             nbody_with_starpu(positions, velocities, accelerations, masses, parameters, nslices)
#         end
#         t_starpu = toq()
        
#         println("No Starpu...")
#         tic()
#         for i = 1:nbr_simulations
#             nbody_jl(positionsjl, velocitiesjl, accelerationsjl, masses, epsilon, dt)
#         end
#         t_julia = toq()
        
        
        
#         times_starpu[k+1] = t_starpu
#         times_julia[k+1] = t_julia
#     end
#     open("./DAT/nbody.dat", "w") do f
#         for i = 0:nbr_steps
#             write(f, "$(starting_nbr_planets + i*step_nbr)")
#             write(f, " $(times_starpu[i+1])")
#             write(f, " $(times_julia[i+1])\n")
#         end
#     end
# end


function set_to_zero(A ::Array{<:Real,2})
    height,width = size(A)
    for i = 1:height
        for j = 1:width
            A[i,j] = 0
        end
    end
end

function median_times(nbr_tests ::Int64, nbr_planets ::Int64, nbr_simulations ::Int64, nslices ::Int64)

######################### INITIALIZATION #########################

    width ::Int64 = 1000
    height ::Int64 = 1000

    epsilon ::Float64 = 2.5E8
    dt ::Float64 = 3600
    G ::Float64 = 6.67408E-11
    parameters ::Vector{Float64} = [G,dt,epsilon]

    # Coordinate interval for the final display screen.
    maxcoord ::Int64 = 2E8
    mincoord ::Int64 = -2E8

    exec_times_st ::Vector{Float64} = [0 for i = 1:nbr_tests]
    exec_times_cpu ::Vector{Float64} = [0 for i = 1:nbr_tests]
    exec_times_gpu ::Vector{Float64} = [0 for i = 1:nbr_tests]
    # exec_times_jl ::Vector{Float64} = [0 for i = 1:nbr_tests]

    # Arrays used for each of the starpu-using functions. 
    positions ::Matrix{Float64} = zeros(2, nbr_planets)
    velocities ::Matrix{Float64} = zeros(2, nbr_planets)
    accelerations_st ::Matrix{Float64} = zeros(2, nbr_planets)    
    
    # Arrays used for the naive julia function.
    # positions_jl ::Vector{Vector{Float64}} = [[0, 0] for i = 1:nbr_planets]
    # velocities_jl ::Vector{Vector{Float64}} = [[0, 0] for i = 1:nbr_planets]
    # accelerations_jl ::Vector{Vector{Float64}} = [[0, 0] for i = 1:nbr_planets]
    
    
    masses ::Vector{Float64} = [0 for i = 1:nbr_planets]
    

    for k = 1:nbr_tests
        println("Test $k...")
        # Initializing the starpu and naive julia arrays with the same values.
        for i = 1:nbr_planets
            mi ::Float64 = rand() * 5E22
            ri = mi * 2.5E-15
            
            angle ::Float64 = rand() * 2 * pi
            distToCenter ::Float64 = rand() * 1.0E8 + 1.0E8
            
            qix ::Float64 = cos(angle) * distToCenter
            qiy ::Float64 = sin(angle) * distToCenter
            
            vix ::Float64 = qiy * 4.0E-6
            viy ::Float64 = -qix * 4.0E-6
            
            masses[i] = mi
            
            positions[1,i] = qix
            positions[2,i] = qiy
            
            #positions_jl[i] = [qix, qiy]
            
            velocities[1,i] = vix
            velocities[2,i] = viy
            
            #velocities_jl[i] = [vix, viy]
        end


######################### SIMULATION #########################

        # Using new arrays for the starpu functions, so we can keep in memory the initial values.
        positions_st ::Matrix{Float64} = copy(positions)
        velocities_st ::Matrix{Float64} = copy(velocities)
        set_to_zero(accelerations_st)

        tic()
        nbody_with_starpu(positions_st, velocities_st, accelerations_st, masses, parameters, nbr_simulations, nslices, nbr_planets)
        t_st = toq()



        # positions_st = copy(positions)
        # velocities_st = copy(velocities)
        # set_to_zero(accelerations_st)
                
        # tic()
        # nbody_with_starpu_cpu(positions_st, velocities_st, accelerations_st, masses, parameters, nbr_simulations, nslices)
        # t_cpu = toq()
        


        # positions_st = copy(positions)
        # velocities_st = copy(velocities)
        # set_to_zero(accelerations_st)

        # tic()
        # nbody_with_starpu_gpu(positions_st, velocities_st, accelerations_st, masses, parameters, nbr_simulations, nslices)
        # t_gpu = toq()

        
        #tic()
        #nbody_jl(positions_jl, velocities_jl, accelerations_jl, masses, nbr_simulations, epsilon, dt)
        #t_jl = toq()
        
        exec_times_st[k] = t_st
        # exec_times_cpu[k] = t_cpu
        # exec_times_gpu[k] = t_gpu
        #exec_times_jl[k] = t_jl
        
    end

    sort!(exec_times_st)
    # sort!(exec_times_cpu)
    # sort!(exec_times_gpu)
    #sort!(exec_times_jl)

    res ::Vector{Float64} = [exec_times_st[1 + div(nbr_tests-1,2)]]#, exec_times_cpu[1 + div(nbr_tests-1,2)], exec_times_gpu[1 + div(nbr_tests-1,2)]]#, exec_times_jl[1 + div(nbr_tests-1,2)]]
    
    return res
end
# Adds the median times of each function inside a .DAT file.
function display_times(start_nbr ::Int64, step_nbr ::Int64, stop_nbr ::Int64, nbr_simulations ::Int64, nslices ::Int64, nbr_tests ::Int64)


    # mtc = map( (x->parse(Float64,x)), open("DAT/nbody_c_times.dat") do f
    #         readlines(f)
    #         end)

    mtjl = map( (x->parse(Float64,x)), open("../DAT/nbody_jl.dat") do f
            readlines(f)
            end)

    mtcstr = map( (x->parse(Float64,x)), open("../DAT/nbody_c_struct_times.dat") do f
                  readlines(f)
                  end)

    mtcarr = map( (x->parse(Float64,x)), open("../DAT/nbody_c_array_times.dat") do f
                  readlines(f)
                  end)


    i = 1;

    #open("./DAT/nbody_jl.dat", "w") do fjl
    open("../DAT/nbody_jl_array_times.dat", "w") do ft
        open("../DAT/nbody_speedups.dat", "w") do f
            for nbr_planets in (start_nbr : step_nbr : stop_nbr)
                println("$(nbr_planets) planets:")
                mt ::Vector{Float64} = median_times(nbr_tests, nbr_planets, nbr_simulations, nslices)
                println("C struct time: $(mtcstr[i])")
                println("C array time: $(mtcarr[i])")
                println("Starpujl time: $(mt[1])")
                # println("CPUjl time: $(mt[2])")
                # println("GPUjl time: $(mt[3])")
                println("Julia time: $(mtjl[i])")
                write(f, "$(nbr_planets) $(mtjl[i]/mt[1]) $(mtjl[i]/mtcstr[i]) $(mtjl[i]/mtcarr[i])\n") 
                write(ft, "$(mt[1])\n")
                #write(fjl, "$(mt[4])\n")
                i = i + 1
                
            end
        end
    end
    #end
end


        