
function substitute(expr :: StarpuExpr, expr_to_replace :: StarpuExpr, new_expr :: StarpuExpr)

    function func_to_apply(x :: StarpuExpr)

        if (x == expr_to_replace)
            return new_expr
        end

        return x
    end

    return apply(func_to_apply, expr)
end



function replace_pattern(expr :: StarpuExpr, replace_€ :: StarpuExpr...)

    replace_index = 0

    function func_to_apply(x :: StarpuExpr)

        if x == @parse €
            replace_index += 1
            return replace_€[replace_index]
        end

        if isa(x, StarpuExprTypedVar) && x.name == :€

            replace_index += 1

            if isa(replace_€[replace_index], StarpuExprVar)
                return StarpuExprTypedVar(replace_€[replace_index].name, x.typ)
            end

            return StarpuExprTypedExpr(replace_€[replace_index], x.typ)
        end

        if isa(x, StarpuExprFunction) && x.func == :€

            replace_index += 1

            if !(isa(replace_€[replace_index], StarpuExprVar))
                error("Can only replace a function name by a variable")
            end

            return StarpuExprFunction(x.ret_type, replace_€[replace_index].name, x.args, x.body)
        end

        return x
    end

    return apply(func_to_apply, expr)
end



import Base.any
function any(cond :: Function, expr :: StarpuExpr)

    err_to_catch = "Catch me, condition is true somewhere !"

    function func_to_apply(x :: StarpuExpr)

        if cond(x)
            error(err_to_catch) # dirty but osef
        end

        return x
    end

    try
        apply(func_to_apply, expr)
    catch err

        if (isa(err, ErrorException) && err.msg == err_to_catch)
            return true
        end

        throw(err)
    end

    return false
end


import Base.all
function all(cond :: Function, expr :: StarpuExpr)
    return !any(!cond, expr)
end
