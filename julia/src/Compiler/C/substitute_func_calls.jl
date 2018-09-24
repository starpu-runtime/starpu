


func_substitution = Dict(
    :width => :STARPU_MATRIX_GET_NY,
    :height => :STARPU_MATRIX_GET_NX,

    :length => :STARPU_VECTOR_GET_NX
)



function substitute_func_calls(expr :: StarpuExpr)

    function func_to_apply(x :: StarpuExpr)

        if !isa(x, StarpuExprCall) || !(x.func in keys(func_substitution))
            return x
        end

        return StarpuExprCall(func_substitution[x.func], x.args)
    end

    return apply(func_to_apply, expr)
end
