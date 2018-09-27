
export starpu_init

"""
    Must be called before any other starpu function. Field extern_task_path is the
    shared library path which will be used to find StarpuCodelet
    cpu and gpu function names
"""
function starpu_init(; extern_task_path = "")

    if (!isempty(extern_task_path))
        global starpu_tasks_library_handle = Libdl.dlopen(extern_task_path)
    else
        global starpu_tasks_library_handle = Libdl.dlopen("libjlstarpu_c_wrapper.so")
    end

    output = @starpucall jlstarpu_init Cint ()

    starpu_enter_new_block()

    return output
end


export starpu_shutdown

"""
    Must be called at the end of the program
"""
function starpu_shutdown()
    starpu_exit_block()
    @starpucall starpu_shutdown Void ()
    jlstarpu_free_allocated_structures()
    return nothing
end
