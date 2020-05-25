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

global starpu_wrapper_library_handle = C_NULL

global starpu_tasks_library_handle = C_NULL

global starpu_target=STARPU_CPU

global generated_cuda_kernel_file_name = "PRINT TO STDOUT"
global generated_cpu_kernel_file_name = "PRINT TO STDOUT"

global CPU_CODELETS=Dict{String,String}()
global CUDA_CODELETS=Dict{String,String}()

global CODELETS_SCALARS=Dict{String,Any}()
global CODELETS_PARAMS_STRUCT=Dict{String,Any}()

global starpu_type_traduction_dict = Dict(
    Int32 => "int32_t",
    UInt32 => "uint32_t",
    Float32 => "float",
    Int64 => "int64_t",
    UInt64 => "uint64_t",
    Float64 => "double",
    Nothing => "void"
)
export starpu_type_traduction_dict

global mutex = Threads.SpinLock()

# detect CUDA support
try
    STARPU_USE_CUDA == 1
catch
   global  const STARPU_USE_CUDA = 0
end
