
if length(ARGS) != 6
    println("Usage : julia prog.jl start_dim step_dim stop_dim nb_tests nslicesx nslicesy")
    quit()
end

include("../../src/Wrapper/Julia/starpu_include.jl")
using StarPU

@debugprint "starpu_init"
starpu_init(extern_task_path = "../build/extern_tasks.so")

perfmodel = StarpuPerfmodel(
    perf_type = STARPU_HISTORY_BASED,
    symbol = "history_perf"
)

cl = StarpuCodelet(
    cpu_func = "cpu_mult",
    gpu_func = "gpu_mult",
    modes = [STARPU_R, STARPU_R, STARPU_W],
    perfmodel = perfmodel
)

include("mult_def.jl")

display_times(map((x -> parse(Int64,x)), ARGS)..., "../mult_extern.dat")

@debugprint "starpu_shutdown"
starpu_shutdown()
