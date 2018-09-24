



STARPU_MAXIMPLEMENTATIONS = 1 # TODO : good value
STARPU_NMAXBUFS = 8 # TODO : good value


STARPU_CPU = 1 << 1
STARPU_CUDA = 1 << 3


macro starpufunc(symbol)
    :($symbol, "libjlstarpu_c_wrapper")
end

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




macro mutableview(t)
    :(unsafe_wrap( Vector{eltype($t)}, Ptr{eltype($t)}(pointer_from_objref($t)), length($t)))
end
