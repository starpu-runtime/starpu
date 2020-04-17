


global generated_cuda_kernel_file_name = "PRINT TO STDOUT"



global generated_cpu_kernel_file_name = "PRINT TO STDOUT"

const cpu_kernel_file_start = "#include <stdio.h>
#include <stdint.h>
#include <starpu.h>
#include <math.h>

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

#define THREADS_PER_BLOCK 64

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


__device__ static inline long long jlstarpu_max__device(long long a, long long b)
{
	return (a > b) ? a : b;
}

__device__ static inline long long jlstarpu_interval_size__device(long long start, long long step, long long stop)
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

export CPU_CODELETS
global CPU_CODELETS=Dict{String,String}()
export CUDA_CODELETS
global CUDA_CODELETS=Dict{String,String}()

export CODELETS_SCALARS
global CODELETS_SCALARS=Dict{String,Any}()
export CODELETS_PARAMS_STRUCT
global CODELETS_PARAMS_STRUCT=Dict{String,Any}()

"""
	    Executes @cuda_kernel and @cpu_kernel
        """
macro codelet(x)
    parsed = starpu_parse(x)
    name=string(x.args[1].args[1].args[1]);
    dump(name)
    parse_scalar_parameters(parsed, name)
    cpu_expr = transform_to_cpu_kernel(parsed)
    prekernel, kernel = transform_to_cuda_kernel(parsed)
    generated_cpu_kernel_file_name=string("genc_",string(x.args[1].args[1].args[1]),".c")
    generated_cuda_kernel_file_name=string("gencuda_",string(x.args[1].args[1].args[1]),".cu")
    targets=starpu_target
    return quote
        
        if ($targets&$STARPU_CPU!=0)
            kernel_file = open($(esc(generated_cpu_kernel_file_name)), "w")
            @debugprint "generating " $(generated_cpu_kernel_file_name)
            print(kernel_file, $(esc(cpu_kernel_file_start)))
            print(kernel_file, generate_c_struct_param_declaration($name))
            print(kernel_file, $cpu_expr)
            close(kernel_file)
            CPU_CODELETS[$name]=$name
        end
        
        if ($targets&$STARPU_CUDA!=0)
            kernel_file = open($(esc(generated_cuda_kernel_file_name)), "w")
            @debugprint "generating " $(generated_cuda_kernel_file_name)
            print(kernel_file, $(esc(cuda_kernel_file_start)))
            print(kernel_file, "__global__ ", $kernel)
            print(kernel_file, "\nextern \"C\" ", $prekernel)
            close(kernel_file)
            CUDA_CODELETS[$name]="CUDA_"*$name
        end
        print("end generation")
        #starpu_task_library_name="generated_tasks"
        #global starpu_task_library_name
    end
end

function parse_scalar_parameters(expr :: StarpuExprFunction, name::String)
    scalar_parameters = []
    for i in (1 : length(expr.args))
        type = expr.args[i].typ
        if (type <: Number || type <: AbstractChar)
            push!(scalar_parameters, (expr.args[i].name, type))
        end
    end

    CODELETS_SCALARS[name] = scalar_parameters

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
    CODELETS_PARAMS_STRUCT[name] = struct_params_name
end
