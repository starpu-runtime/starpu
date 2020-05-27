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
    Lenient comparison operator for structures and arrays.
"""
@generated function ≂(x, y)
    if x != y || x <: Type
        :(x == y)
    elseif !isempty(fieldnames(x))
        mapreduce(n -> :(x.$n ≂ y.$n), (a,b)->:($a && $b), fieldnames(x))
    elseif x <: Array
        quote
            if length(x) != length(y)
                return false
            end
            for i in 1:length(x)
                if !(x[i] ≂ y[i])
                    return false
                end
            end
            return true
        end
    else
        :(x == y)
    end
end

"""
    Returns a new expression where every occurrence of expr_to_replace into expr
    has been replaced by new_expr
"""
function substitute(expr :: StarpuExpr, expr_to_replace :: StarpuExpr, new_expr :: StarpuExpr)

    function func_to_apply(x :: StarpuExpr)
        if (x ≂ expr_to_replace)
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

function visit_preorder(func :: Function, expr :: StarpuExprAffect)
    func(expr)
    visit_preorder(func, expr.var)
    visit_preorder(func, expr.expr)
    return expr
end

function visit_preorder(func :: Function, expr :: StarpuExprBlock)
    func(expr)
    for e in expr.exprs
        visit_preorder(func, e)
    end
    return expr
end

function visit_preorder(func :: Function, expr :: StarpuExprCall)
    func(expr)
    for a in expr.args
        visit_preorder(func, a)
    end
    return expr
end

function visit_preorder(func :: Function, expr :: StarpuExprCudaCall)
    func(expr)
    func(expr.nblocks)
    func(expr.threads_per_block)
    for a in expr.args
        visit_preorder(func, a)
    end
    return expr
end

function visit_preorder(func :: Function, expr :: StarpuExprField)
    func(expr)
    func(expr.left)
    func(expr.field)
    func(expr.is_an_arrow)
    return expr
end

function visit_preorder(func :: Function, expr :: StarpuExprFor)
    func(expr)
    for d in expr.set_declarations
        visit_preorder(func, d)
    end
    visit_preorder(func, expr.set)
    visit_preorder(func, expr.body)
    return expr
end

function visit_preorder(func :: Function, expr :: StarpuExprFunction)
    func(expr)
    for a in expr.args
        visit_preorder(func, a)
    end
    visit_preorder(func, e.body)
    return expr
end

function visit_preorder(func :: Function, expr :: StarpuExprIf)
    func(expr)
    visit_preorder(func, expr.cond)
    visit_preorder(func, expr.then_statement)
    return expr
end



function visit_preorder(func :: Function, expr :: StarpuExprIfElse)
    func(expr)
    visit_preorder(func, expr.cond)
    visit_preorder(func, expr.then_statement)
    visit_preorder(func, expr.else_statement)
    return expr
end

function visit_preorder(func :: Function, expr :: StarpuExprInterval)
    func(expr)
    visit_preorder(func, expr.start)
    visit_preorder(func, expr.step)
    visit_preorder(func, expr.stop)
    return expr
end

function visit_preorder(func :: Function, expr :: StarpuExprRef)
    func(expr)
    visit_preorder(func, expr.ref)
    for i in expr.indexes
        visit_preorder(func, i)
    end
    return expr
end

function visit_preorder(func :: Function, expr :: StarpuExprAddress)
    func(expr)
    visit_preorder(func, expr.ref)
    return expr
end

function visit_preorder(func :: Function, expr :: StarpuExprBreak)
    func(expr)
    return expr
end

function visit_preorder(func :: Function, expr :: StarpuExprReturn)
    func(expr)
    visit_preorder(func, expr.value)
    return expr
end

function visit_preorder(func :: Function, expr :: StarpuExpr)
    func(expr)
    return expr
end

function visit_preorder(func :: Function, expr :: StarpuExprTypedExpr)
    func(expr)
    visit_preorder(func, expr.expr)
    return expr
end

function visit_preorder(func :: Function, expr :: StarpuExprWhile)
    func(expr)
    visit_preorder(func, expr.cond)
    visit_preorder(func, expr.body)
    return expr
end

# function substitute_preorder(expr :: StarpuExprAffect, match :: StarpuExpr, replace :: StarpuExpr)
#     if expr == match
#         return replace
#     end
#     var = substitute_preorder(func, expr.var)
#     expr = substitute_preorder(func, expr.expr)

#     if var != expr.var || expr != expr.expr
#         return StarpuExprAffect(var, expr)
#     end
#     return expr
# end

# function substitute_preorder(expr :: StarpuExprBlock, match :: StarpuExpr, replace :: StarpuExpr)
#     if expr == match
#         return replace
#     end

#     modified = false
#     new_exprs = Vector{StarpuExpr}()
#     for e in expr.exprs
#         push!(new_exprs, substitute_preorder(func, e))
#     end
#     if new_exprs != expr.exprs
#         return StarpuExprBlock(new_exprs)
#     end
#     return expr
# end

# function substitute_preorder(expr :: StarpuExprCall, match :: StarpuExpr, replace :: StarpuExpr)
#     if expr == match
#         return replace
#     end

#     new_args = Vector{StarpuExpr}()
#     for a in expr.args
#         push!(new_args, substitute_preorder(func, a))
#     end
#     if new_args != expr.args
#         return StarpuExprCall(expr.func, new_args)
#     end
#     return expr
# end

# function substitute_preorder(expr :: StarpuExprCudaCall, match :: StarpuExpr, replace :: StarpuExpr)
#     if expr == match
#         return replace
#     end

#     new_args = Vector{StarpuExpr}()
#     for a in expr.args
#         push!(new_args, substitute_preorder(func, a))
#     end
#     if new_args != expr.args
#         return new StarpuExprCudaCall(expr.ker_name, expr.nblocks, expr.threads_per_block, new_args)
#     end
#     return expr
# end

# function substitute_preorder(expr :: StarpuExprField, match :: StarpuExpr, replace :: StarpuExpr)
#     if expr == match
#         return replace
#     end

#     left = substitute_preorder(expr.left, match, replace)
#     if left != expr.left
#         return StarpuExprField(left, expr.field, expr.is_an_arrow)
#     end
#     return expr
# end

# function substitute_preorder(expr :: StarpuExprFor, match :: StarpuExpr, replace :: StarpuExpr)
#     if expr == match
#         return replace
#     end

#     new_set_declarations = Vector{StarpuExpr}()
    
#     for d in expr.set_declarations
#         substitute_preorder(func, d)
#     end
#     substitute_preorder(expr.set, match :: StarpuExpr, replace :: StarpuExpr)
#     substitute_preorder(func, expr.body)
#     return expr
# end

# function substitute_preorder(expr :: StarpuExprFunction, match :: StarpuExpr, replace :: StarpuExpr)
#         if expr == match
#         return replace
#     end

#     for a in expr.args
#         substitute_preorder(func, a)
#     end
#     substitute_preorder(e.body, match :: StarpuExpr, replace :: StarpuExpr)
#     return expr
# end

# function substitute_preorder(expr :: StarpuExprIf, match :: StarpuExpr, replace :: StarpuExpr)
#         if expr == match
#         return replace
#     end

#     substitute_preorder(func, expr.cond)
#     substitute_preorder(func, expr.then_statement)
#     return expr
# end



# function substitute_preorder(expr :: StarpuExprIfElse, match :: StarpuExpr, replace :: StarpuExpr)
#         if expr == match
#         return replace
#     end

#     substitute_preorder(func, expr.cond)
#     substitute_preorder(func, expr.then_statement)
#     substitute_preorder(func, expr.else_statement)
#     return expr
# end

# function substitute_preorder(expr :: StarpuExprInterval, match :: StarpuExpr, replace :: StarpuExpr)
#         if expr == match
#         return replace
#     end

#     substitute_preorder(func, expr.start)
#     substitute_preorder(func, expr.step)
#     substitute_preorder(func, expr.stop)
#     return expr
# end

# function substitute_preorder(expr :: StarpuExprRef, match :: StarpuExpr, replace :: StarpuExpr)
#         if expr == match
#         return replace
#     end

#     substitute_preorder(func, expr.ref)
#     for i in expr.indexes
#         substitute_preorder(func, i)
#     end
#     return expr
# end

# function substitute_preorder(expr :: StarpuExprAddress, match :: StarpuExpr, replace :: StarpuExpr)
#         if expr == match
#         return replace
#     end

#     substitute_preorder(func, expr.ref)
#     return expr
# end

# function substitute_preorder(expr :: StarpuExprBreak, match :: StarpuExpr, replace :: StarpuExpr)
#         if expr == match
#         return replace
#     end

#     return expr
# end

# function substitute_preorder(expr :: StarpuExprReturn, match :: StarpuExpr, replace :: StarpuExpr)
#         if expr == match
#         return replace
#     end

#     substitute_preorder(func, expr.value)
#     return expr
# end

# function substitute_preorder(expr :: StarpuExpr, match :: StarpuExpr, replace :: StarpuExpr)
#         if expr == match
#         return replace
#     end

#     return expr
# end

# function substitute_preorder(expr :: StarpuExprTypedExpr, match :: StarpuExpr, replace :: StarpuExpr)
#         if expr == match
#         return replace
#     end

#     substitute_preorder(func, expr.expr)
#     return expr
# end

# function substitute_preorder(expr :: StarpuExprWhile, match :: StarpuExpr, replace :: StarpuExpr)
#         if expr == match
#         return replace
#     end

#     substitute_preorder(func, expr.cond)
#     substitute_preorder(func, expr.body)
#     return expr
# end
