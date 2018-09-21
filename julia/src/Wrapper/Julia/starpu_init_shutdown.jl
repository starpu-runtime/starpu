
export starpu_init
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
function starpu_shutdown()
    starpu_exit_block()
    @starpucall starpu_shutdown Void ()
    jlstarpu_free_allocated_structures()
    return nothing
end
