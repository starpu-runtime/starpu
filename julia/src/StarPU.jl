# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2020-2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

if !isfile(joinpath(fstarpu_build_dir(), "julia/gen/libstarpu_common.jl")) || !isfile(joinpath(fstarpu_build_dir(), "julia/gen/libstarpu_api.jl")) ||
    mtime(joinpath(@__FILE__, "translate_headers.jl")) > mtime(joinpath(fstarpu_build_dir(), "julia/gen/libstarpu_api.jl"))
    starpu_translate_headers()
end

include(joinpath(fstarpu_build_dir(), "julia/gen/libstarpu_common.jl"))
include(joinpath(fstarpu_build_dir(), "julia/gen/libstarpu_api.jl"))
include("globals.jl")

include("compiler/include.jl")
include("linked_list.jl")
include("destructible.jl")
include("perfmodel.jl")
include("data.jl")
include("blas.jl")
include("task.jl")
include("task_dep.jl")
include("init.jl")

# macro
export @starpu_filter
export @starpu_block
export @starpu_async_cl
export @starpu_sync_tasks

# enum / define
export STARPU_CPU
export STARPU_CUDA
export STARPU_CUDA_ASYNC
export STARPU_OPENCL
export STARPU_MAIN_RAM
export StarpuDataFilterFunc
export STARPU_MATRIX_FILTER_VERTICAL_BLOCK, STARPU_MATRIX_FILTER_BLOCK
export STARPU_VECTOR_FILTER_BLOCK
export STARPU_PERFMODEL_INVALID, STARPU_PER_ARCH, STARPU_COMMON
export STARPU_HISTORY_BASED, STARPU_REGRESSION_BASED
export STARPU_NL_REGRESSION_BASED, STARPU_MULTIPLE_REGRESSION_BASED
export starpu_tag_t
export STARPU_NONE,STARPU_R,STARPU_W,STARPU_RW, STARPU_SCRATCH
export STARPU_MPI_REDUX, STARPU_REDUX,STARPU_COMMUTE, STARPU_SSEND, STARPU_LOCALITY
export STARPU_ACCESS_MODE_MAX

# BLAS
export STARPU_SAXPY

# functions
export starpu_cublas_init
export starpu_init
export starpu_shutdown
export starpu_memory_pin
export starpu_memory_unpin
export starpu_data_access_mode
export starpu_data_acquire_on_node
export starpu_data_release_on_node
export starpu_data_unregister
export starpu_data_register
export starpu_data_get_sub_data
export starpu_data_partition
export starpu_data_unpartition
export starpu_data_map_filters
export starpu_data_wont_use
export starpu_task_insert
export starpu_task_wait_for_all
export starpu_task_submit
export starpu_task_end_dep_add
export starpu_task_end_dep_release
export starpu_task_declare_deps
export starpu_task_declare_end_deps
export starpu_task_wait_for_n_submitted
export starpu_task_destroy
export starpu_tag_remove
export starpu_tag_wait
export starpu_tag_notify_from_apps
export starpu_iteration_pop
export starpu_iteration_push
export starpu_tag_declare_deps
export starpu_task
export starpu_task_wait
export starpu_codelet
export starpu_perfmodel
export starpu_perfmodel_type
export starpu_translate_headers
export starpu_data_get_default_sequential_consistency_flag
export starpu_data_set_default_sequential_consistency_flag
export starpu_data_get_sequential_consistency_flag
export starpu_data_set_sequential_consistency_flag
export starpu_worker_get_count
export starpu_cpu_worker_get_count
export starpu_cuda_worker_get_count
export starpu_opencl_worker_get_count

end
