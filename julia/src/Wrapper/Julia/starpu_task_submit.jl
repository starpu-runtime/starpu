


export starpu_task_submit

"""
    Launches task execution, if "synchronous" task field is set to "false", call
    returns immediately
"""
function starpu_task_submit(task :: StarpuTask)

    if (length(task.handles) != length(task.cl.modes))
        error("Invalid number of handles for task : $(length(task.handles)) where given while codelet has $(output.cl.nbuffers) modes")
    end

    starpu_c_task_update(task)

    @starpucall starpu_task_submit Cint (Ptr{Void},) task.c_task
end


export @starpu_async_cl

"""
    Creates and submits an asynchronous task running cl Codelet function.
    Ex : @starpu_async_cl cl(handle1, handle2)
"""
macro starpu_async_cl(expr)

    if (!isa(expr, Expr) || expr.head != :call)
        error("Invalid task submit syntax")
    end

    cl = expr.args[1]
    handles = Expr(:vect, expr.args[2:end]...)

    quote
        task = StarpuTask(cl = $(esc(cl)), handles = $(esc(handles)))
        starpu_task_submit(task)
    end
end


export starpu_task_wait_for_all
"""
    Blocks until every submitted task has finished.
"""
function starpu_task_wait_for_all()
    @threadcall(@starpufunc(:starpu_task_wait_for_all),
                          Cint, ())
end


export @starpu_sync_tasks

"""
    Blocks until every submitted task has finished.
    Ex : @starpu_sync_tasks begin
                [...]
                starpu_task_submit(task)
                [...]
        end

    TODO : Make the macro only wait for tasks declared inside the following expression.
            (similar mechanism as @starpu_block)
"""
macro starpu_sync_tasks(expr)
    quote
        $(esc(expr))
        starpu_task_wait_for_all()
    end
end
