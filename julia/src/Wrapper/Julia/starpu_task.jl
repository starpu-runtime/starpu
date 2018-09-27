
export StarpuTask
mutable struct StarpuTask

    cl :: StarpuCodelet
    handles :: Vector{StarpuDataHandle}
    handle_pointers :: Vector{StarpuDataHandlePointer}
    synchronous :: Bool
    cl_arg :: Union{Ref, Void}

    c_task :: Ptr{Void}


    """
        StarpuTask(; cl :: StarpuCodelet, handles :: Vector{StarpuDataHandle}, cl_arg :: Ref)

        Creates a new task which will run the specified codelet on handle buffers and cl_args data
    """
    function StarpuTask(; cl :: Union{Void, StarpuCodelet} = nothing, handles :: Vector{StarpuDataHandle} = StarpuDataHandle[], cl_arg :: Union{Ref, Void} = nothing)

        if (cl == nothing)
            error("\"cl\" field can't be empty when creating a StarpuTask")
        end

        output = new()

        output.cl = cl
        output.handles = handles
        output.cl_arg = cl_arg

        output.synchronous = false
        output.handle_pointers = StarpuDataHandlePointer[]

        c_task = @starpucall starpu_task_create Ptr{Void} ()

        if (c_task == C_NULL)
            error("Couldn't create new task: starpu_task_create() returned NULL")
        end

        output.c_task = c_task

        starpu_c_task_update(output)

        return output
    end

end


"""
    Updates fields of the real C structures stored at "c_task" field
"""
function starpu_c_task_update(task :: StarpuTask)

    task_translator = StarpuTaskTranslator(task)

    @starpucall(jlstarpu_task_update,
                Void, (Ptr{Void}, Ptr{Void}),
                Ref{StarpuTaskTranslator}(task_translator),
                task.c_task
            )
end


"""
    Structure used to update fields of the real C task structure 
"""
mutable struct StarpuTaskTranslator

    cl :: Ptr{Void}
    handles :: Ptr{Void}
    synchronous :: Cuint

    cl_arg :: Ptr{Void}
    cl_arg_size :: Csize_t

    function StarpuTaskTranslator(task :: StarpuTask)

        output = new()

        output.cl = task.cl.c_codelet

        task.handle_pointers = map((x -> x.object), task.handles)
        output.handles = pointer(task.handle_pointers)
        output.synchronous = Cuint(task.synchronous)

        if (task.cl_arg == nothing)
            output.cl_arg = C_NULL
            output.cl_arg_size = 0
        else
            output.cl_arg = pointer_from_objref(task.cl_arg) #TODO : Libc.malloc and cl_arg_free set to 1 ? but it should be done only when submitting
            output.cl_arg_size = sizeof(eltype(task.cl_arg))
        end

        return output
    end

end


export StarpuTag
const StarpuTag = UInt64


export starpu_tag_declare_deps
function starpu_tag_declare_deps(id :: StarpuTag, dep :: StarpuTag, other_deps :: StarpuTag...)

    v = [dep, other_deps...]

    @starpucall(starpu_tag_declare_deps_array,
                Void, (StarpuTag, Cuint, Ptr{StarpuTag}),
                id, length(v), pointer(v)
        )
end


export starpu_iteration_push
function starpu_iteration_push(iteration)

    @starpucall(starpu_iteration_push,
                Void, (Culong,), iteration
        )
end


export starpu_iteration_pop
function starpu_iteration_pop()
    @starpucall starpu_iteration_pop Void ()
end


export starpu_tag_wait
function starpu_tag_wait(id :: StarpuTag)
    @starpucall starpu_tag_wait Cint (StarpuTag,) id
end


function starpu_tag_wait(ids :: Vector{StarpuTag})

    @starpucall(starpustarpu_tag_wait_array,
                Cint, (Cuint, Ptr{StarpuTag}),
                length(ids), pointer(ids)
        )
end


export starpu_task_destroy
function starpu_task_destroy(task :: StarpuTask)
    @starpucall starpu_task_destroy Void (Ptr{Void},) task.c_task
end


export starpu_task_wait_for_n_submitted

"""
    Block until there are n submitted tasks left (to the current context or the global one if there is no current context) to
    be executed. It does not destroy these tasks.
"""
function starpu_task_wait_for_n_submitted(n)
    @starpucall starpu_task_wait_for_n_submitted Cint (Cuint,) n
end


export starpu_task_declare_deps

"""
    starpu_task_declare_deps(task :: StarpuTask, dep :: StarpuTask [, other_deps :: StarpuTask...])

    Declare task dependencies between a task and the following provided ones. This function must be called
    prior to the submission of the task, but it may called after the submission or the execution of the tasks in the array,
    provided the tasks are still valid (i.e. they were not automatically destroyed). Calling this function on a task that was
    already submitted or with an entry of task_array that is no longer a valid task results in an undefined behaviour.
"""
function starpu_task_declare_deps(task :: StarpuTask, dep :: StarpuTask, other_deps :: StarpuTask...)

    task_array = [dep.c_task, map((t -> t.c_task), other_deps)...]

    @starpucall(starpu_task_declare_deps_array,
                Void, (Ptr{Void}, Cuint, Ptr{Void}),
                task.c_task,
                length(task_array),
                pointer(task_array)
            )
end
