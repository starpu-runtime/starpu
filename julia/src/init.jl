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
"""
    Must be called before any other starpu function. Field extern_task_path is the
    shared library path which will be used to find StarpuCodelet
    cpu and gpu function names
"""
function starpu_init()
    debug_print("starpu_init")

    if (get(ENV,"JULIA_TASK_LIB",0)!=0)
        global starpu_tasks_library_handle= Libdl.dlopen(ENV["JULIA_TASK_LIB"])
        debug_print("Loading external codelet library")
        ff = Libdl.dlsym(starpu_tasks_library_handle,:starpu_find_function)
        dump(ff)
        for k in keys(CPU_CODELETS)
            CPU_CODELETS[k]=unsafe_string(ccall(ff,Cstring, (Cstring,Cstring),Cstring_from_String(string(k)),Cstring_from_String("cpu")))
            if STARPU_USE_CUDA == 1
                CUDA_CODELETS[k]=unsafe_string(ccall(ff,Cstring, (Cstring,Cstring),Cstring_from_String(string(k)),Cstring_from_String("gpu")))
            end
            print(k,">>>>",CPU_CODELETS[k],"\n")
        end
    else
        srcdir=get(ENV,"STARPU_JULIA_BUILD",0)
        if (srcdir == 0)
            error("Must define environment variable STARPU_JULIA_BUILD")
        end
        makefile=string(srcdir, "/src/dynamic_compiler/Makefile")
        debug_print("generating codelet library with ")
        debug_print(makefile)
        run(`make -f $makefile generated_tasks.so`)
        global starpu_tasks_library_handle=Libdl.dlopen("generated_tasks.so")
    end
    global starpu_wrapper_library_handle= Libdl.dlopen(starpu_wrapper_library_name)
    output = starpu_init(C_NULL)

    global task_pool = ThreadPools.QueuePool(2)

    starpu_enter_new_block()

    return output
end

"""
    Must be called at the end of the program
"""
function starpu_shutdown()
    debug_print("starpu_shutdown")

    starpu_exit_block()
    @starpucall starpu_shutdown Cvoid ()

    lock(mutex)
    empty!(perfmodel_list)
    empty!(codelet_list)
    empty!(task_list)
    unlock(mutex)

    return nothing
end
