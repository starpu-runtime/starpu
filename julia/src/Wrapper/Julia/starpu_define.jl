



STARPU_MAXIMPLEMENTATIONS = 1 # TODO : These must be the same values as defined in C macros !
STARPU_NMAXBUFS = 8 # TODO : find a way to make it automatically match


STARPU_CPU = 1 << 1
STARPU_CUDA = 1 << 3

macro starpufunc(symbol)
    :($symbol, "libjlstarpu_c_wrapper")
end

"""
    Used to call a StarPU function compiled inside "libjlstarpu_c_wrapper.so"
    Works as ccall function
"""
macro starpucall(func, ret_type, arg_types, args...)
    return Expr(:call, :ccall, (func, "libjlstarpu_c_wrapper"), esc(ret_type), esc(arg_types), map(esc, args)...)
end


export @debugprint
macro debugprint(x...)

    expr = Expr(:call, :println, "\x1b[32m", map(esc, x)..., "\x1b[0m")

    quote
        $expr
        flush(STDOUT)
    end
end



function Cstring_from_String(str :: String)
    return Cstring(pointer(str))
end



function jlstarpu_set_to_zero(x :: T) :: Ptr{Void} where {T}
    @starpucall(memset,
          Ptr{Void}, (Ptr{Void}, Cint, Csize_t),
          Ref{T}(x), 0, sizeof(x)
        )
end
