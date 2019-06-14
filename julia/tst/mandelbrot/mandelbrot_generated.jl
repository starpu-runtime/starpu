

if length(ARGS) != 7
    println("Usage : julia prog.jl cr ci start_dim step_dim stop_dim nslices nbr_tests")
    quit()
end

if (parse(Int64,ARGS[3]) % parse(Int64,ARGS[6]) != 0)
    println("The number of slices should divide all the dimensions.")
    quit()
end

include("../../src/Wrapper/Julia/starpu_include.jl")
using StarPU

@debugprint "starpu_init"
starpu_init(extern_task_path = "../build/generated_tasks_mandelbrot.so")

perfmodel = StarpuPerfmodel(
    perf_type = STARPU_HISTORY_BASED,
    symbol = "history_perf"
)

cl = StarpuCodelet(
    cpu_func = "mandelbrot",
    gpu_func = "CUDA_mandelbrot",
    modes = [STARPU_W, STARPU_R, STARPU_R],
    perfmodel = perfmodel
)

clcpu = StarpuCodelet(
    cpu_func = "mandelbrot",
    modes = [STARPU_W, STARPU_R, STARPU_R],
    perfmodel = perfmodel
)

clgpu = StarpuCodelet(
    gpu_func = "CUDA_mandelbrot",
    modes = [STARPU_W, STARPU_R, STARPU_R],
    perfmodel = perfmodel
)

include("mandelbrot_def.jl")

display_time(parse(Float64,ARGS[1]), parse(Float64,ARGS[2]), map((x -> parse(Int64, x)), ARGS[3:7])...)

@debugprint "starpu_shutdown"
starpu_shutdown()