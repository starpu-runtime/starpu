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
function multjl(A ::Matrix{Float32}, B ::Matrix{Float32}, C ::Matrix{Float32})
    heightC, widthC = size(C)
    widthA = size(A)[2]
    for i = 1:heightC
        for j = 1:widthC
            sum = 0
            for k = 1:widthA
                sum = sum + A[i, k] * B[k, j]
            end
            C[i,j] = sum
        end
    end
end