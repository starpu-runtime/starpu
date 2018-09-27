


global generated_cpu_kernel_file_name = "PRINT TO STDOUT"

const cpu_kernel_file_start = "#include <stdio.h>
#include <stdint.h>
#include <starpu.h>

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


"""
	Opens a new C source file, where generated CPU kernels will be written
"""
function starpu_new_cpu_kernel_file(file_name :: String)

    global generated_cpu_kernel_file_name = file_name

    kernel_file = open(file_name, "w")
    print(kernel_file, cpu_kernel_file_start)
    close(kernel_file)

    return nothing
end


"""
	Executes the StarPU C compiler to the following function declaration.
	If no call to starpu_new_cpu_kernel_file has been made before, it only
	prints the reulting function. Otherwise, it writes into the source file
	specified when starpu_new_cpu_kernel_file was called.
"""
macro cpu_kernel(x)

    starpu_expr = transform_to_cpu_kernel(starpu_parse(x))

    return quote

        to_stdout = ($(esc(generated_cpu_kernel_file_name)) == "PRINT TO STDOUT")

        if to_stdout
			println("\nNo specified CPU kernel file to write into : writting to STDOUT instead\n")
            kernel_file = STDOUT
        else
            kernel_file = open($(esc(generated_cpu_kernel_file_name)), "a+")
        end

        print(kernel_file, $starpu_expr)

        if (!to_stdout)
            close(kernel_file)
        end
    end
end
