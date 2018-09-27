

"""
	Compiles C source file opened by starpu_new_cpu_kernel_file
    and filled by @cpu_kernel declarations.
    Output file is a shared library which can be provided to starpu_init() in
    order to find kernel.
"""
function compile_cpu_kernels(output_file :: String)

    starpu_cflags = readstring(`pkg-config --cflags starpu-1.3`)[1:end-1]
    starpu_libs = readstring(`pkg-config --libs starpu-1.3`)[1:end-1]
    options = "-O3 -shared -fPIC"

    system("gcc $generated_cpu_kernel_file_name $options $starpu_cflags $starpu_libs -o $output_file")

    global generated_cpu_kernel_file_name = "PRINT TO STDOUT"

    return nothing
end


"""
	Compiles Cuda source file opened by starpu_new_cuda_kernel_file
    and filled by @cuda_kernel declarations.
    Output file is a shared library which can be provided to starpu_init() in
    order to find kernel.
"""
function compile_cuda_kernels(output_file :: String)

    starpu_cflags = readstring(`pkg-config --cflags starpu-1.3`)[1:end-1]
    starpu_libs = readstring(`pkg-config --libs starpu-1.3`)[1:end-1]
    options = " -O3 --shared --compiler-options \'-fPIC\' "

    system("nvcc $generated_cuda_kernel_file_name $options $starpu_cflags $starpu_libs -o $output_file")

    global generated_cuda_kernel_file_name = "PRINT TO STDOUT"

    return nothing
end


"""
    Combines several shared library into a new one.
    Can be used to have both CPU and Cuda kernels (from compile_cpu_kernels
    compile_cuda_kernels) accessible from the same library.
"""
function combine_kernel_files(output_file :: String, input_files :: Vector{String})

    input_str = (*)(map((x -> x * " "), input_files)...)

    system("gcc -shared -fPIC $input_str -o $output_file")

end
