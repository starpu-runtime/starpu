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
function fstarpu_task_library_name()
    x=get(ENV, "STARPU_JULIA_LIB", C_NULL)
    if (x == C_NULL)
        error("Environment variable STARPU_JULIA_LIB must be defined")
    end
    return x
end

function fstarpu_build_dir()
    x=get(ENV, "STARPU_BUILD_DIR", C_NULL)
    if (x == C_NULL)
        error("Environment variable STARPU_BUILD_DIR must be defined")
    end
    return x
end

function fstarpu_src_dir()
    x=get(ENV, "STARPU_SRC_DIR", C_NULL)
    if (x == C_NULL)
        error("Environment variable STARPU_SRC_DIR must be defined")
    end
    return x
end

macro starpufunc(symbol)
    :($symbol, starpu_wrapper_library_name)
end

"""
    Used to call a StarPU function compiled inside "libjlstarpu_c_wrapper.so"
    Works as ccall function
"""
macro starpucall(func, ret_type, arg_types, args...)
    return Expr(:call, :ccall, (func, starpu_wrapper_library_name), esc(ret_type), esc(arg_types), map(esc, args)...)
end

function debug_print(x...)
    println("\x1b[32m", x..., "\x1b[0m")
    flush(stdout)
end

function Cstring_from_String(str :: String)
    return Cstring(pointer(str))
end

tuple_len(::NTuple{N, Any}) where {N} = N

function starpu_find_function(name :: String, device :: String ) 
    s=ccall(:starpu_find_function,Cstring, (Cstring,Cstring),Cstring_from_String(name),Cstring_from_String(device))
    if  s == C_NULL
        print("NULL STRING\n")
        error("dead")
    end
    return s
end

function load_starpu_function_pointer(func_name :: String)

    if (isempty(func_name))
        return C_NULL
    end
    #func_pointer = ccall(:dlsym,"libdl",Ptr{Cvoid});
    func_pointer=Libdl.dlsym(starpu_tasks_library_handle, func_name)

    if (func_pointer == C_NULL)
        error("Couldn't find function symbol $func_name into extern library file $starpu_tasks_library")
    end

    return func_pointer
end

function load_wrapper_function_pointer(func_name :: String)
    if (isempty(func_name))
        return C_NULL
    end

    func_pointer=Libdl.dlsym(starpu_wrapper_library_handle, func_name)

    if (func_pointer == C_NULL)
        error("Couldn't find function symbol $func_name into extern library file $starpu_tasks_library")
    end

    return func_pointer
end


"""
    Declares a Julia function which is just calling the StarPU function
    having the same name.
"""
macro starpu_noparam_function(func_name, ret_type)

    func = Symbol(func_name)

    quote
        export $func
        global $func() = ccall(($func_name, starpu_wrapper_library_name),
                                $ret_type, ()) :: $ret_type
    end
end
