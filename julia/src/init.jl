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
        for k in keys(CUDA_CODELETS)
            CPU_CODELETS[k]=unsafe_string(ccall(ff,Cstring, (Cstring,Cstring),Cstring_from_String(string(k)),Cstring_from_String("cpu")))
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
    return nothing
end
