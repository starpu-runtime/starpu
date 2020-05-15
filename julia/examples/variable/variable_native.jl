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
function variable(val ::Ref{Float32}) :: Nothing
    val[] = val[] + 1

    return
end

function variable_without_starpu(val ::Ref{Float32}, niter)
    for i = 1:niter
        variable(val)
    end
end

function display(niter)
    foo = Ref(0.0f0)

    variable_without_starpu(foo, niter)

    println("variable -> ", foo[])
    if foo[] == niter
        println("result is correct")
    else
        println("result is incorret")
    end
end

display(10)
