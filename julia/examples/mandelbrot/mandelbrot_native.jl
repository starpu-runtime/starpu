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

function mandelbrot(pixels, centerr ::Float64, centeri ::Float64, offset ::Int64, dim ::Int64) :: Nothing
    height :: Int64, width :: Int64 = size(pixels)
    zoom :: Float64 = width * 0.25296875
    iz :: Float64 = 1. / zoom
    diverge :: Float32 = 4.0
    max_iterations :: Float32 = ((width/2) * 0.049715909 * log10(zoom));
    imi :: Float64 = 1. / max_iterations
    cr :: Float64 = 0.
    zr :: Float64 = 0.
    ci :: Float64 = 0.
    zi :: Float64 = 0.
    n :: Int64 = 0
    tmp :: Float64 = 0.
    for y = 1:height
        for x = 1:width
            cr = centerr + (x-1 - (dim / 2)) * iz
            zr = cr
            ci = centeri + (y-1+offset - (dim / 2)) * iz
            zi = ci
            n = 0
            for i = 0:max_iterations
                n = i
                if (zr*zr + zi*zi > diverge)
                    break
                end
                tmp = zr*zr - zi*zi + cr
                zi = 2*zr*zi + ci
                zr = tmp
            end

            if (n < max_iterations)
                pixels[y,x] = round(15 * n * imi)
            else
                pixels[y,x] = 0
            end
        end
    end

    return
end

function mandelbrot_without_starpu(A ::Matrix{Int64}, cr ::Float64, ci ::Float64, dim ::Int64, nslicesx ::Int64)
    width,height = size(A)
    step = height / nslicesx

    for taskx in (1 : nslicesx)
        start_id = floor(Int64, (taskx-1)*step+1)
        end_id = floor(Int64, (taskx-1)*step+step)
        a = view(A, start_id:end_id, :)

        offset ::Int64 = (taskx-1)*dim/nslicesx
        mandelbrot(a, cr, ci, offset, dim)
    end
end

function pixels2img(pixels ::Matrix{Int64}, width ::Int64, height ::Int64, filename ::String)
    MAPPING = [[66,30,15],[25,7,26],[9,1,47],[4,4,73],[0,7,100],[12,44,138],[24,82,177],[57,125,209],[134,181,229],[211,236,248],[241,233,191],[248,201,95],[255,170,0],[204,128,0],[153,87,0],[106,52,3]]
    open(filename, "w") do f
        write(f, "P3\n$width $height\n255\n")
        for i = 1:height
            for j = 1:width
                write(f,"$(MAPPING[1+pixels[i,j]][1]) $(MAPPING[1+pixels[i,j]][2]) $(MAPPING[1+pixels[i,j]][3]) ")
            end
            write(f, "\n")
        end
    end
end

function min_times(cr ::Float64, ci ::Float64, dim ::Int64, nslices ::Int64, gen_images)
    tmin=0;

    pixels ::Matrix{Int64} = zeros(dim, dim)
    for i = 1:10
        t = time_ns();
        mandelbrot_without_starpu(pixels, cr, ci, dim, nslices)
        t = time_ns()-t
        if (tmin==0 || tmin>t)
            tmin=t
        end
    end
    if (gen_images == 1)
        pixels2img(pixels,dim,dim,"out$(dim).ppm")
    end
    return tmin
end

function display_time(cr ::Float64, ci ::Float64, start_dim ::Int64, step_dim ::Int64, stop_dim ::Int64, nslices ::Int64, gen_images)
    for dim in (start_dim : step_dim : stop_dim)
        res = min_times(cr, ci, dim, nslices, gen_images)
        res=res/dim/dim; # time per pixel
        println("$(dim) $(res)")
    end
end


display_time(-0.800671,-0.158392,32,32,512,4, 0)
