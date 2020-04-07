

"""
    Returns the list of instruction that will be added before for loop of shape
        "for for_index_var in set ..."
"""
function interval_evaluation_declarations(set :: StarpuExprInterval, for_index_var :: Symbol)

    decl_pattern = @parse € :: Int64
    affect_pattern = @parse € :: Int64 = €
    interv_size_affect_pattern = @parse € :: Int64 = jlstarpu_interval_size(€, €, €)

    id = set.id

    start_var = starpu_parse(Symbol(:start_, id))
    start_decl = replace_pattern(affect_pattern, start_var, set.start)

    index_var = starpu_parse(for_index_var)
    index_decl = replace_pattern(decl_pattern, index_var)

    if isa(set.step, StarpuExprValue)

        stop_var = starpu_parse(Symbol(:stop_, id))
        stop_decl = replace_pattern(affect_pattern, stop_var, set.stop)

        return StarpuExpr[start_decl, stop_decl, index_decl]
    end

    step_var = starpu_parse(Symbol(:step_, id))
    step_decl = replace_pattern(affect_pattern, step_var, set.step)

    dim_var = starpu_parse(Symbol(:dim_, id))
    dim_decl = replace_pattern(interv_size_affect_pattern, dim_var, start_var, step_var, set.stop)

    iter_var = starpu_parse(Symbol(:iter_, id))
    iter_decl = replace_pattern(decl_pattern, iter_var)


    return StarpuExpr[start_decl, step_decl, dim_decl, iter_decl, index_decl]
end


function add_for_loop_declarations(expr :: StarpuExpr)

    function func_to_apply(x :: StarpuExpr)

        if !isa(x, StarpuExprFor)
            return x
        end

        interval_decl = interval_evaluation_declarations(x.set, x.iter)

        return StarpuExprFor(x.iter, x.set, x.body, x.is_independant, interval_decl)
    end

    return apply(func_to_apply, expr)
end





function transform_to_cpu_kernel(expr :: StarpuExprFunction)

    output = add_for_loop_declarations(expr)
    output = substitute_args(output)
    output = substitute_func_calls(output)
    output = substitute_indexing(output)
    output = flatten_blocks(output)

    return output
end



function flatten_blocks(expr :: StarpuExpr)

    function func_to_run(x :: StarpuExpr)

        if !isa(x, StarpuExprBlock)
            return x
        end

        instrs = StarpuExpr[]

        for sub_expr in x.exprs

            if isa(sub_expr, StarpuExprBlock)
                push!(instrs, sub_expr.exprs...)
            else
                push!(instrs, sub_expr)
            end
        end

        return StarpuExprBlock(instrs)
    end

    return apply(func_to_run, expr)
end


function substitute_argument_usage(expr :: StarpuExpr, arg_index, buffer_name :: Symbol, arg_name :: Symbol, ptr_name :: Symbol)
    function func_to_apply(x :: StarpuExpr)

        if x == StarpuExprVar(arg_name)
            return StarpuExprVar(ptr_name)
        end

        if !(isa(x, StarpuExprCall) && x.func in keys(func_substitution))
            return x
        end

        if (length(x.args) != 1)
            error("Invalid arity for function $(x.func)")
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
    post = false
    function_start_affectations = StarpuExpr[]

    for i in (1 : length(expr.args))

        var_id = rand_string()
        ptr = Symbol(:ptr_, var_id)
        var_name = ptr
        
        if (expr.args[i].typ <: Vector)
            func_interface = :STARPU_VECTOR_GET_PTR
        elseif (expr.args[i].typ <: Matrix)
            func_interface = :STARPU_MATRIX_GET_PTR
            ld_name = Symbol("ld_", var_id)
            post_affect = starpu_parse( :($ld_name :: UInt32 = STARPU_MATRIX_GET_LD($buffer_arg_name[$i])) )
            post=true
            
        elseif (expr.args[i].typ <: Float32)
            func_interface = :STARPU_VARIABLE_GET_PTR
            var_name = Symbol("scal_", var_id)
            post_affect = starpu_parse( :($var_name :: Float32 = ($ptr[0])) )
            post = true
            
        end
        #else
            #error("Task arguments must be either vector or matrix (got $(expr.args[i].typ))") #TODO : cl_args, variable ?
        #end

        type_in_arg = eltype(expr.args[i].typ)
        new_affect = starpu_parse( :($ptr :: Ptr{$type_in_arg} = $func_interface($buffer_arg_name[$i])) )
        push!(function_start_affectations, new_affect)
        if (post)
            push!(function_start_affectations, post_affect)
        end
        new_body = substitute_argument_usage(new_body, i, buffer_arg_name, expr.args[i].name, var_name)

    end


    new_args = [
                    starpu_parse(:($buffer_arg_name :: Matrix{Nothing})),
                    starpu_parse(:($cl_arg_name :: Vector{Nothing}))
                ]
    new_body = StarpuExprBlock([function_start_affectations..., new_body.exprs...])

    return StarpuExprFunction(expr.ret_type, expr.func, new_args, new_body)
end



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


function substitute_indexing(expr :: StarpuExpr)

    function func_to_run(x :: StarpuExpr)

        if !isa(x, StarpuExprRef)
            return x
        end

        #if !isa(x.ref, StarpuExprVar)
        #    error("Only variable indexing is allowed") #TODO allow more ?
        #end


        nb_indexes = length(x.indexes)

        if (nb_indexes >= 3)
            error("Indexing with more than 2 indexes is not allowed") # TODO : blocks
        end

        if (nb_indexes == 0)
            return x

        elseif nb_indexes == 1
            new_index = StarpuExprCall(:-, [x.indexes[1], StarpuExprValue(1)])  #TODO : add field "offset" from STARPU_VECTOR_GET interface
                                                                            #TODO : detect when it is a matrix used with one index only
            return StarpuExprRef(x.ref, [new_index])

        elseif nb_indexes == 2

            var_name = String(x.ref.name)

            if !occursin(r"ptr_", var_name) || isempty(var_name[5:end])
                error("Invalid variable ($var_name) for multiple index dereferencing")
            end

            var_id = var_name[5:end]
            ld_name = Symbol("ld_", var_id) # TODO : check if this variable is legit (var_name must refer to a matrix)

            new_index = x.indexes[2]
            new_index = StarpuExprCall(:(-), [new_index, StarpuExprValue(1)])
            new_index = StarpuExprCall(:(*), [new_index, StarpuExprVar(ld_name)])
            new_index = StarpuExprCall(:(+), [x.indexes[1], new_index])
            new_index = StarpuExprCall(:(-), [new_index, StarpuExprValue(1)])

            return StarpuExprRef(x.ref, [new_index])
        end
    end

    return apply(func_to_run, expr)
end
