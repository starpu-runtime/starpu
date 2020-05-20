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

starpu_new_cpu_kernel_file("../build/generated_cpu_mandelbrot.c")
starpu_new_cuda_kernel_file("../build/generated_cuda_mandelbrot.cu")

@cpu_cuda_kernel function mandelbrot(pixels ::Matrix{Int64}, params ::Vector{Float64}, slice_pos ::Vector{Int64}) :: Void
    

    local_width ::Int64 = width(pixels) 
    local_height ::Int64 = height(pixels)

    #max_iterations ::Int64 = 250
    conv_limit ::Float64 = 2.0
   
    @indep for x in (1 : local_width)
	@indep for y in (1 : local_height)

            max_iterations ::Float64 = params[5]

            zoom ::Float64 = params[3] * 0.25296875

	    X ::Int64 = x + local_width * (slice_pos[1] - 1)
	    Y ::Int64 = y + local_height * (slice_pos[2] - 1)
	    
            cr ::Float64 = params[1] + (X - params[3]/2)/zoom
	    zr ::Float64 = cr

            ci ::Float64 = params[2] + (Y - params[4]/2)/zoom
            zi ::Float64 = ci

	    n ::Int64 = 0

            b1 ::Int64 = (n < max_iterations) + (zr*zr + zi*zi < conv_limit * conv_limit)

	    while (b1 >= 2)#n <= max_iterations) && (z * z < conv_limit * conv_limit) #Double condition impossible!!!

                tmp ::Float64 = zr*zr - zi*zi + cr
                zi = 2*zr*zi + ci
                zr = tmp

		n = n + 1
                b1 = (n <= max_iterations) + (zr*zr + zi*zi <= conv_limit * conv_limit)

	    end 

	    if (n < max_iterations)
		pixels[y,x] = 255 * n / max_iterations
	    else
	        pixels[y,x] = 0
	    end
	end
    end
end


compile_cpu_kernels("../build/generated_cpu_mandelbrot.so")
compile_cuda_kernels("../build/generated_cuda_mandelbrot.so")
combine_kernel_files("../build/generated_tasks_mandelbrot.so", ["../build/generated_cpu_mandelbrot.so","../build/generated_cuda_mandelbrot.so"])