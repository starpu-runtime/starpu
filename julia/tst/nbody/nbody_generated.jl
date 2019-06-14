if length(ARGS) != 6
    println("Usage: julia nbody_generated.jl start_nbr step_nbr stop_nbr nbr_simulations nbr_slices nbr_tests")
    quit()
end

if parse(Int64, ARGS[1]) % parse(Int64, ARGS[5]) != 0
    println("The number of slices must divide the number of planets.")
    quit()
end

include("../../src/Wrapper/Julia/starpu_include.jl")
using StarPU

@debugprint "starpu_init"
starpu_init(extern_task_path = "../build/generated_tasks_nbody.so")

perfmodel = StarpuPerfmodel(
    perf_type = STARPU_HISTORY_BASED,
    symbol = "history_perf"
)

# Normal starpu codelets
claccst = StarpuCodelet(
    cpu_func = "nbody_acc",
    gpu_func = "CUDA_nbody_acc",
    modes = [STARPU_R, STARPU_RW, STARPU_R, STARPU_R, STARPU_R],
    perfmodel = perfmodel
)

clupdtst = StarpuCodelet(
    cpu_func = "nbody_updt",
    gpu_func = "CUDA_nbody_updt",
    modes = [STARPU_RW, STARPU_RW, STARPU_R, STARPU_R],
    perfmodel = perfmodel
)

# CPU_only codelets
clacccpu = StarpuCodelet(
    cpu_func = "nbody_acc",
    modes = [STARPU_R, STARPU_RW, STARPU_R, STARPU_R, STARPU_R],
    perfmodel = perfmodel
)

clupdtcpu = StarpuCodelet(
    cpu_func = "nbody_updt",
    modes = [STARPU_RW, STARPU_RW, STARPU_R,STARPU_R],
    perfmodel = perfmodel
)

# GPU_only codelets
claccgpu = StarpuCodelet(
    gpu_func = "CUDA_nbody_acc",
    modes = [STARPU_R, STARPU_RW, STARPU_R, STARPU_R, STARPU_R],
    perfmodel = perfmodel
)

clupdtgpu = StarpuCodelet(
    gpu_func = "CUDA_nbody_updt",
    modes = [STARPU_RW, STARPU_RW, STARPU_R,STARPU_R],
    perfmodel = perfmodel
)

include("nbody_def.jl")

display_times(map((x -> parse(Int64, x)), ARGS)...)

@debugprint "starpu_shutdown"
starpu_shutdown()