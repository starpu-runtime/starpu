if length(ARGS) != 5
    println("Usage: julia prog.jl start_data_nbr step_data_nbr stop_data_nbr nslices nbr_tests")
    quit()
end


if (parse(Int64,ARGS[1]) < parse(Int64,ARGS[4]))
    println("The number of slices must be smaller than the number of data")
    quit()
end

include("../../src/Wrapper/Julia/starpu_include.jl")
using StarPU

@debugprint "starpu_init"
starpu_init(extern_task_path = "../build/generated_tasks_black_scholes")

perfmodel = StarpuPerfmodel(
    perf_type = STARPU_HISTORY_BASED,
    symbol = "history_perf"
)

cl = StarpuCodelet(
cpu_func = "black_scholes",
gpu_func = "CUDA_black_scholes",
modes = [STARPU_RW, STARPU_RW],
perfmodel = perfmodel
)

include("black_scholes_def.jl")

display_times(map( (x->parse(Int64, x)), ARGS)...)

@debugprint "starpu_shutdown"
starpu_shutdown()