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
function get_planet_pixels(X ::Int64, Y ::Int64)
    pix ::Array{Tuple{Int64,Int64}} = []
    for i = X-1:X+1
        for j = Y-3:Y+3
            push!(pix,(i,j))
        end
    end
    for i = Y-2:Y+2
        push!(pix,(X-2,i))
        push!(pix,(X+2,i))
    end
    for i = Y-1:Y+1
        push!(pix,(X-3,i))
        push!(pix,(X+3,i))
    end
    return pix
end


function is_inside(t ::Tuple{Int64,Int64}, width ::Int64, height ::Int64)
    if (t[1] > 0 && t[1] <= width && t[2] > 0 && t[2] <= height)
        return true
    else
        return false
    end
end

function graph_planets(pixels ::Array{Array{Int64}}, positions ::Matrix{Float64}, min_value ::Float64, max_value ::Float64, width ::Int64, height ::Int64, file_name ::String)
    n = size(positions)[2]
    for i = 1:n
        X ::Int64= round( ( (positions[1, i] - min_value) / (max_value - min_value) ) * (width - 1) ) + 1
        Y ::Int64= round( (1 - ( (positions[2, i] - min_value) / (max_value - min_value) ) ) * (height - 1) ) + 1
        pix ::Array{Tuple{Int64,Int64}} = get_planet_pixels(X,Y)
        for pixel in pix
            if is_inside(pixel, width, height)
                pixels[(pixel[2] - 1)*width + pixel[1]] = [125, round(255*i/n)]
            end
        end
    end

    open(file_name,"w") do f
        write(f, "P3\n$width $height\n255\n")
        for he = 1:height
            for wi = 1:width
                write(f, "$(pixels[(he - 1)*width + wi][1]) 0 $(pixels[(he - 1)*width + wi][2]) ")
            end
            write(f,"\n")
        end
    end
end

# function graph_planets(pixels ::Array{Array{Int64}}, positions ::Vector{Vector{Float64}}, min_value ::Float64, max_value ::Float64, width ::Int64, height ::Int64, file_name ::String)
#     n = length(positions)
#     for i = 1:n
#         X ::Int64= round( ( (positions[i][1] - min_value) / (max_value - min_value) ) * (width - 1) ) + 1
#         Y ::Int64= round( (1 - ( (positions[i][2] - min_value) / (max_value - min_value) ) ) * (height - 1) ) + 1
#         pix ::Array{Tuple{Int64,Int64}} = get_planet_pixels(X,Y)
#         for pixel in pix
#             if is_inside(pixel, width, height)
#                 pixels[(pixel[2] - 1)*width + pixel[1]] = [125, round(255*i/n)]
#             end
#         end
#     end

#     open(file_name,"w") do f
#         write(f, "P3\n$width $height\n255\n")
#         for he = 1:height
#             for wi = 1:width
#                 write(f, "$(pixels[(he - 1)*width + wi][1]) 0 $(pixels[(he - 1)*width + wi][2]) ")
#             end
#             write(f,"\n")
#         end
#     end
# end
