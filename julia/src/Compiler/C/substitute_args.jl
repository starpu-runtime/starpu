

function substitute_argument_usage(expr :: StarpuExpr, arg_index, buffer_name :: Symbol, arg_name :: Symbol, ptr_name :: Symbol)

    function func_to_apply(x :: StarpuExpr)

        if x == StarpuExprVar(arg_name)
            return StarpuExprVar(ptr_name)
        end

        if !(isa(x, StarpuExprCall) && x.func in keys(func_substitution))
            return x
        end

        if (length(x.args) != 1)
            error("Invalid arrity for function $(x.func)")
        end

        if (x.args[1] != StarpuExprVar(ptr_name))
            return x
        end

        new_func = func_substitution[x.func]
        new_arg = starpu_parse(:($buffer_name[$arg_index]))

        return StarpuExprCall(new_func, [new_arg])
    end

    return apply(func_to_apply, expr)
end



function substitute_args(expr :: StarpuExprFunction)

    new_body = expr.body
    func_id = rand_string()
    buffer_arg_name = Symbol("buffers_", func_id)
    cl_arg_name = Symbol("cl_arg_", func_id)

    function_start_affectations = StarpuExpr[]

    for i in (1 : length(expr.args))

        var_id = rand_string()
        ptr = Symbol(:ptr_, var_id)

        if (expr.args[i].typ <: Vector)
            func_interface = :STARPU_VECTOR_GET_PTR

        elseif (expr.args[i].typ <: Matrix)
            func_interface = :STARPU_MATRIX_GET_PTR
            ld_name = Symbol("ld_", var_id)
            new_affect = starpu_parse( :($ld_name :: UInt32 = STARPU_MATRIX_GET_LD($buffer_arg_name[$i])) )
            push!(function_start_affectations, new_affect)

        else
            error("Task arguments must be either vector or matrix (got $(expr.args[i].typ))") #TODO : cl_args, variable ?
        end

        type_in_arg = eltype(expr.args[i].typ)
        new_affect = starpu_parse( :($ptr :: Ptr{$type_in_arg} = $func_interface($buffer_arg_name[$i])) )
        push!(function_start_affectations, new_affect)

        #var_to_replace = starpu_parse(expr.args[i].name)
        #replace_with = starpu_parse(ptr)
        #new_body = substitute(new_body, var_to_replace, replace_with)
        new_body = substitute_argument_usage(new_body, i, buffer_arg_name, expr.args[i].name, ptr)
    end


    new_args = [
                    starpu_parse(:($buffer_arg_name :: Matrix{Void})),
                    starpu_parse(:($cl_arg_name :: Vector{Void}))
                ]
    new_body = StarpuExprBlock([function_start_affectations..., new_body.exprs...])

    return StarpuExprFunction(expr.ret_type, expr.func, new_args, new_body)
end
