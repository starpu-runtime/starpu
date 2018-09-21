

export StarpuDataAccessMode
export STARPU_NONE,STARPU_R,STARPU_W,STARPU_RW, STARPU_SCRATCH
export STARPU_REDUX,STARPU_COMMUTE, STARPU_SSEND, STARPU_LOCALITY
export STARPU_ACCESS_MODE_MAX

@enum(StarpuDataAccessMode,

    STARPU_NONE = 0,
    STARPU_R = (1 << 0),
    STARPU_W = (1 << 1),
    STARPU_RW = ((1 << 0) | (1 << 1)),
    STARPU_SCRATCH = (1 << 2),
    STARPU_REDUX = (1 << 3),
    STARPU_COMMUTE = (1 << 4),
    STARPU_SSEND = (1 << 5),
    STARPU_LOCALITY = (1 << 6),
    STARPU_ACCESS_MODE_MAX = (1 << 7)

)


export StarpuCodelet
struct StarpuCodelet

    where_to_execute :: UInt32

    cpu_func :: String
    gpu_func :: String

    modes :: Vector{StarpuDataAccessMode}

    perfmodel :: StarpuPerfmodel

    c_codelet :: Ptr{Void}


    function StarpuCodelet(;
        cpu_func :: String = "",
        gpu_func :: String = "",
        modes :: Vector{StarpuDataAccessMode} = StarpuDataAccessMode[],
        perfmodel :: StarpuPerfmodel = StarpuPerfmodel(),
        where_to_execute :: Union{Void, UInt32} = nothing
    )

        if (length(modes) > STARPU_NMAXBUFS)
            error("Codelet has too much buffers ($(length(modes)) but only $STARPU_NMAXBUFS are allowed)")
        end

        real_c_codelet_ptr = @starpucall jlstarpu_new_codelet Ptr{Void} ()
        push!(jlstarpu_allocated_structures, real_c_codelet_ptr)

        if (where_to_execute == nothing)
            real_where = ((cpu_func != "") * STARPU_CPU) | ((gpu_func != "") * STARPU_CUDA)
        else
            real_where = where_to_execute
        end

        output = new(real_where, cpu_func, gpu_func, modes, perfmodel, real_c_codelet_ptr)

        starpu_c_codelet_update(output)

        return output
    end
end



function starpu_c_codelet_update(cl :: StarpuCodelet)

    translating_cl = StarpuCodeletTranslator(cl)

    @starpucall(jlstarpu_codelet_update,
                Void, (Ptr{Void}, Ptr{Void}),
                Ref{StarpuCodeletTranslator}(translating_cl),
                cl.c_codelet
            )
end



function load_starpu_function_pointer(func_name :: String)

    if (isempty(func_name))
        return C_NULL
    end

    func_pointer = Libdl.dlsym(starpu_tasks_library_handle, func_name)

    if (func_pointer == C_NULL)
        error("Couldn't find function symbol $func_name into extern library file $starpu_tasks_library")
    end

    return func_pointer
end



mutable struct StarpuCodeletTranslator

    where_to_execute :: UInt32

    cpu_func :: Ptr{Void}
    cpu_func_name :: Cstring

    gpu_func :: Ptr{Void}

    nbuffers :: Cint
    modes :: Ptr{Void}

    perfmodel :: Ptr{Void}



    function StarpuCodeletTranslator(cl :: StarpuCodelet)

        output = new()

        if (iszero(cl.where_to_execute))
            error("StarpuCodelet field \"where_to_execute\" is empty")
        end

        output.where_to_execute = cl.where_to_execute

        cpu_func_ptr = load_starpu_function_pointer(cl.cpu_func)
        gpu_func_ptr = load_starpu_function_pointer(cl.gpu_func)

        if (cpu_func_ptr == C_NULL && gpu_func_ptr == C_NULL)
            error("No function specified inside codelet")
        end

        output.cpu_func = cpu_func_ptr
        output.cpu_func_name = Cstring_from_String(cl.cpu_func)

        output.gpu_func = gpu_func_ptr

        output.nbuffers = Cint(length(cl.modes))
        output.modes = pointer(cl.modes)

        output.perfmodel = cl.perfmodel.c_perfmodel

        return output
    end

end
