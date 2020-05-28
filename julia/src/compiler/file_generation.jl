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
const cpu_kernel_file_start = "#include <stdio.h>
#include <stdint.h>
#include <starpu.h>
#include <math.h>

#include \"blas.h\"

static inline long long jlstarpu_max(long long a, long long b)
{
	return (a > b) ? a : b;
}

static inline long long jlstarpu_interval_size(long long start, long long step, long long stop)
{
    if (stop >= start){
            return jlstarpu_max(0, (stop - start + 1) / step);
    } else {
            return jlstarpu_max(0, (stop - start - 1) / step);
    }
}

"

const cuda_kernel_file_start = "#include <stdio.h>
#include <stdint.h>
#include <starpu.h>
#include <math.h>
#include <starpu_cublas_v2.h>

#define THREADS_PER_BLOCK 64

__attribute__((unused)) static inline long long jlstarpu_max(long long a, long long b)
{
	return (a > b) ? a : b;
}

__attribute__((unused)) static inline long long jlstarpu_interval_size(long long start, long long step, long long stop)
{
    if (stop >= start){
            return jlstarpu_max(0, (stop - start + 1) / step);
    } else {
            return jlstarpu_max(0, (stop - start - 1) / step);
    }
}


__attribute__((unused)) __device__ static inline long long jlstarpu_max__device(long long a, long long b)
{
	return (a > b) ? a : b;
}

__attribute__((unused)) __device__ static inline long long jlstarpu_interval_size__device(long long start, long long step, long long stop)
{
	if (stop >= start){
		return jlstarpu_max__device(0, (stop - start + 1) / step);
	} else {
		return jlstarpu_max__device(0, (stop - start - 1) / step);
	}
}

"

"""
	Opens a new Cuda source file, where generated GPU kernels will be written
"""
function starpu_new_cuda_kernel_file(file_name :: String)

    global generated_cuda_kernel_file_name = file_name

    kernel_file = open(file_name, "w")
    print(kernel_file, cuda_kernel_file_start)
    close(kernel_file)

    return nothing
end

export target
macro target(x)
    targets = eval(x)
    return quote
        starpu_target=$targets
        global starpu_target
    end
end

"""
	    Executes @cuda_kernel and @cpu_kernel
        """
macro codelet(x)
    parsed = starpu_parse(x)
    name=string(x.args[1].args[1].args[1]);
    cpu_name = name
    cuda_name = "CUDA_"*name
    dump(name)
    parse_scalar_parameters(parsed, name)
    c_struct_param_decl = generate_c_struct_param_declaration(name)
    cpu_expr = transform_to_cpu_kernel(parsed)

    generated_cpu_kernel_file_name=string("genc_",string(x.args[1].args[1].args[1]),".c")
    generated_cuda_kernel_file_name=string("gencuda_",string(x.args[1].args[1].args[1]),".cu")

    if (starpu_target & STARPU_CPU != 0)
        kernel_file = open(generated_cpu_kernel_file_name, "w")
        debug_print("generating ", generated_cpu_kernel_file_name)
        print(kernel_file, cpu_kernel_file_start)
        print(kernel_file, c_struct_param_decl)
        print(kernel_file, cpu_expr)
        close(kernel_file)
        CPU_CODELETS[name]=cpu_name
    end

    if (starpu_target & STARPU_CUDA!=0) && STARPU_USE_CUDA == 1
        kernel_file = open(generated_cuda_kernel_file_name, "w")
        debug_print("generating ", generated_cuda_kernel_file_name)
        print(kernel_file, cuda_kernel_file_start)
        prekernel, kernel = transform_to_cuda_kernel(parsed)

        if kernel != nothing
            print(kernel_file, "__global__ ", kernel)
        end

        print(kernel_file, c_struct_param_decl)
        print(kernel_file, "\nextern \"C\" ", prekernel)
        close(kernel_file)
        CUDA_CODELETS[name]=cuda_name
    end
end

function parse_scalar_parameters(expr :: StarpuExprFunction, codelet_name)
    scalar_parameters = []
    for i in (1 : length(expr.args))
        type = expr.args[i].typ
        if (type <: Number || type <: AbstractChar)
            push!(scalar_parameters, (expr.args[i].name, type))
        end
    end

    CODELETS_SCALARS[codelet_name] = scalar_parameters

    # declare structure carrying scalar parameters
    struct_params_name = Symbol("params_", rand_string())
    structure_decl_str = "mutable struct " * "$struct_params_name\n"
    for p in scalar_parameters
        structure_decl_str *= "$(p[1])::$(p[2])\n"
    end
    structure_decl_str *= "end"
    eval(Meta.parse(structure_decl_str))

    # add structure type to dictionnary
    add_to_dict_str = "starpu_type_traduction_dict[$struct_params_name] = \"struct $struct_params_name\""
    eval(Meta.parse(add_to_dict_str))

    # save structure name
    CODELETS_PARAMS_STRUCT[codelet_name] = struct_params_name
end
