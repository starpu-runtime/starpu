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
"""
__precompile__()
"""
module StarPU
import Libdl
using CBinding


include("utils.jl")

const starpu_wrapper_library_name=fstarpu_task_library_name()

include("translate_headers.jl")

translate_starpu_headers()

include("../gen/libstarpu_common.jl")
include("../gen/libstarpu_api.jl")
include("globals.jl")

include("compiler/include.jl")
include("linked_list.jl")
include("destructible.jl")
include("perfmodel.jl")
include("data.jl")
include("task.jl")
include("task_dep.jl")
include("init.jl")

export STARPU_CPU
export starpu_init
export starpu_shutdown
export starpu_memory_pin
export starpu_memory_unpin
export starpu_data_access_mode
export starpu_data_unregister
export starpu_data_register
export starpu_data_get_sub_data
export StarpuDataFilterFunc
export STARPU_MATRIX_FILTER_VERTICAL_BLOCK, STARPU_MATRIX_FILTER_BLOCK
export StarpuDataFilter
export starpu_data_partition
export starpu_data_unpartition
export starpu_data_map_filters
export @starpu_sync_tasks
export starpu_task_wait_for_all
export @starpu_async_cl
export starpu_task_submit
export @starpu_block
export StarpuPerfmodel
export @starpu_filter
export STARPU_PERFMODEL_INVALID, STARPU_PER_ARCH, STARPU_COMMON
export STARPU_HISTORY_BASED, STARPU_REGRESSION_BASED
export STARPU_NL_REGRESSION_BASED, STARPU_MULTIPLE_REGRESSION_BASED
export starpu_task_declare_deps
export starpu_task_wait_for_n_submitted
export starpu_task_destroy
export starpu_tag_wait
export starpu_iteration_pop
export starpu_iteration_push
export starpu_tag_declare_deps
export starpu_task
export STARPU_NONE,STARPU_R,STARPU_W,STARPU_RW, STARPU_SCRATCH
export STARPU_REDUX,STARPU_COMMUTE, STARPU_SSEND, STARPU_LOCALITY
export STARPU_ACCESS_MODE_MAX
export starpu_codelet
export starpu_perfmodel
export starpu_perfmodel_type

end
