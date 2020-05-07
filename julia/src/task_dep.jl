function starpu_tag_declare_deps(id :: starpu_tag_t, dep :: starpu_tag_t, other_deps :: starpu_tag_t...)

    v = [dep, other_deps...]
    starpu_tag_declare_deps_array(id, length(v), pointer(v))
end

"""
    starpu_task_declare_deps(task :: StarpuTask, dep :: StarpuTask [, other_deps :: StarpuTask...])

    Declare task dependencies between a task and the following provided ones. This function must be called
    prior to the submission of the task, but it may called after the submission or the execution of the tasks in the array,
    provided the tasks are still valid (i.e. they were not automatically destroyed). Calling this function on a task that was
    already submitted or with an entry of task_array that is no longer a valid task results in an undefined behaviour.
"""
function starpu_task_declare_deps(task :: jl_starpu_task, dep :: jl_starpu_task, other_deps :: jl_starpu_task...)

    task_array = [dep.c_task, map((t -> t.c_task), other_deps)...]
    starpu_task_declare_deps_array(task.c_task, length(task_array), pointer(task_array))
end
