

function substitute_indexing(expr :: StarpuExpr)

    function func_to_run(x :: StarpuExpr)

        if !isa(x, StarpuExprRef)
            return x
        end

        if !isa(x.ref, StarpuExprVar)
            error("Only variable indexing is allowed") #TODO allow more ?
        end


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

            if !ismatch(r"ptr_", var_name) || isempty(var_name[5:end])
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
