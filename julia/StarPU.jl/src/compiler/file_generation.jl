


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

"""
	    Executes @cuda_kernel and @cpu_kernel
        """
macro codelet(x)
    parsed = starpu_parse(x)
    name=string(x.args[1].args[1].args[1]);
    dump(name)
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
