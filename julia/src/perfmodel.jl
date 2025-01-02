# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2020-2025   University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

perfmodel_list = Vector{starpu_perfmodel}()

function starpu_perfmodel(; perf_type::starpu_perfmodel_type, symbol::String)
    output = starpu_perfmodel(zero)
    output.type = perf_type
    output.symbol = Cstring_from_String(symbol)

    # Performance models must not be garbage collected before starpu_shutdown
    # is called.
    lock(mutex)
    push!(perfmodel_list, output)
    unlock(mutex)

    return output
end
