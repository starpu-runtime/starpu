


global generated_cuda_kernel_file_name = "PRINT TO STDOUT"

const cuda_kernel_file_start = "#include <stdio.h>
#include <stdint.h>
#include <starpu.h>

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


"""
	Executes the StarPU Cuda compiler to the following function declaration.
	If no call to starpu_new_cuda_kernel_file has been made before, it only
	prints the reulting function. Otherwise, it writes into the source file
	specified when starpu_new_cuda_kernel_file was called.
"""
macro cuda_kernel(x)

    prekernel, kernel = transform_to_cuda_kernel(starpu_parse(x))

    return quote

        to_stdout = ($(esc(generated_cuda_kernel_file_name)) == "PRINT TO STDOUT")

        if to_stdout
			println("\nNo specified CUDA kernel file to write into : writting to STDOUT instead\n")
            kernel_file = STDOUT
        else
            kernel_file = open($(esc(generated_cuda_kernel_file_name)), "a+")
        end

        print(kernel_file, "__global__ ", $kernel)
        print(kernel_file, "\nextern \"C\" ", $prekernel)

        if (!to_stdout)
            close(kernel_file)
        end
    end
end



"""
	Executes @cuda_kernel and @cpu_kernel
"""
macro cpu_cuda_kernel(x)

	parsed = starpu_parse(x)
	cpu_expr = transform_to_cpu_kernel(parsed)
	prekernel, kernel = transform_to_cuda_kernel(parsed)

	return quote

		to_stdout = ($(esc(generated_cpu_kernel_file_name)) == "PRINT TO STDOUT")

        if to_stdout
            kernel_file = STDOUT
			println("\nNo specified CPU kernel file to write into : writting to STDOUT instead\n")
        else
            kernel_file = open($(esc(generated_cpu_kernel_file_name)), "a+")
        end

        print(kernel_file, $cpu_expr)

        if (!to_stdout)
            close(kernel_file)
        end


		to_stdout = ($(esc(generated_cuda_kernel_file_name)) == "PRINT TO STDOUT")

        if to_stdout
            kernel_file = STDOUT
			println("\nNo specified CUDA kernel file to write into : writting to STDOUT instead\n")
        else
            kernel_file = open($(esc(generated_cuda_kernel_file_name)), "a+")
        end

        print(kernel_file, "__global__ ", $kernel)
        print(kernel_file, "\nextern \"C\" ", $prekernel)

        if (!to_stdout)
            close(kernel_file)
        end
	end
end
