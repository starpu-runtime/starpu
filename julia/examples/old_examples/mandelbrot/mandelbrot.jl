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
function mandelbrotjl(pixels ::Matrix{Int64}, centerr ::Float64, centeri ::Float64)
    height,width = size(pixels)
    zoom = width * 0.25296875
    val_diverge = 2.0
    max_iterations = (width/2) * 0.049715909 * log10(zoom);


    for y = 1:height
        for x = 1:width
            cr = centerr + (x - (width / 2))/zoom
            zr = cr
            ci = centeri + (y - (height / 2))/zoom
            zi = ci

            n = 0
            while ((n < max_iterations) && (zr*zr + zi*zi < val_diverge*val_diverge))
                tmp = zr*zr - zi*zi + cr
                zi = 2*zr*zi + ci
                zr = tmp
                n = n+1
            end
            
            if (n < max_iterations)
                pixels[y,x] = round(255 * n / max_iterations)
            else
                pixels[y,x] = 0
            end
        end
    end
end