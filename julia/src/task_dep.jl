# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2020       Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
#
# StarPU is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation; either version 2.1 of the License, or (at
# your option) any later version.
#
# StarPU is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
# See the GNU Lesser General Public License in COPYING.LGPL for more details.
#
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

    task_array = [pointer_from_objref(dep.c_task), map((t -> pointer_from_objref(t.c_task)), other_deps)...]
    starpu_task_declare_deps_array(pointer_from_objref(task.c_task), length(task_array), task_array)
end

function starpu_task_end_dep_add(task :: jl_starpu_task, nb_deps :: Int)
    starpu_task_end_dep_add(Ref(task.c_task), nb_deps)
end

function starpu_task_end_dep_release(task :: jl_starpu_task)
    starpu_task_end_dep_release(Ref(task.c_task))
end

function starpu_task_declare_end_deps(task :: jl_starpu_task, dep :: jl_starpu_task, other_deps :: jl_starpu_task...)

    task_array = [pointer_from_objref(dep.c_task), map((t -> pointer_from_objref(t.c_task)), other_deps)...]
    starpu_task_declare_end_deps_array(pointer_from_objref(task.c_task), length(task_array), pointer(task_array))
end
