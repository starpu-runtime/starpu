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


"""
    Returns a new expression where every occurrence of expr_to_replace into expr
    has been replaced by new_expr
"""
function substitute(expr :: StarpuExpr, expr_to_replace :: StarpuExpr, new_expr :: StarpuExpr)

    function func_to_apply(x :: StarpuExpr)

        if (x == expr_to_replace)
            return new_expr
        end

        return x
    end

    return apply(func_to_apply, expr)
end


"""
    Returns an expression where "€" symbols  in expr were replaced
    by the following expression list.

    Ex : replace_pattern((@parse € = €), (@parse x), (@parse 1 + 1))
            --> (StarpuExpr) "x = 1 + 1"
"""
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

"""
    Returns true if one of the sub-expression x in expr
    is such as cond(x) is true, otherwise, it returns false.
"""
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

"""
    Returns true if every sub-expression x in expr
    is such as cond(x) is true, otherwise, it returns false.
"""
function all(cond :: Function, expr :: StarpuExpr)
    return !any(!cond, expr)
end
