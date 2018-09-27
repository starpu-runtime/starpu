
__precompile__()
module StarPU


    include("starpu_define.jl")
    include("static_structures.jl")
    include("starpu_simple_functions.jl")
    include("starpu_perfmodel.jl")
    include("starpu_codelet.jl")

    include("linked_list.jl")
    include("starpu_destructible.jl")
    include("starpu_data_handle.jl")

    include("starpu_task.jl")
    include("starpu_task_submit.jl")
    include("starpu_init_shutdown.jl")

end
