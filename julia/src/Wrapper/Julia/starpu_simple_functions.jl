

macro starpu_noparam_function(func_name, ret_type)

    func = Symbol(func_name)

    quote
        export $func
        global $func() = ccall(($func_name, "libjlstarpu_c_wrapper"),
                                $ret_type, ()) :: $ret_type
    end
end


global starpu_tasks_library_handle = C_NULL



@starpu_noparam_function "starpu_is_initialized" Cint



@starpu_noparam_function "starpu_cublas_init" Void
@starpu_noparam_function "starpu_cublas_set_stream" Void
@starpu_noparam_function "starpu_cublas_shutdown" Void
