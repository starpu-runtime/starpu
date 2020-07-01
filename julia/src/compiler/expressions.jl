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
#======================================================
                AFFECTATION
======================================================#
abstract type StarpuExpr end
abstract type StarpuExprTyped <: StarpuExpr end


struct StarpuExprTypedVar <: StarpuExprTyped
    name :: Symbol
    typ :: Type
end

struct StarpuExprTypedExpr <: StarpuExprTyped # TODO : remove typed expression ?
    expr :: StarpuExpr
    typ :: Type
end

struct StarpuExprAffect <: StarpuExpr
    var :: StarpuExpr
    expr :: StarpuExpr
end

struct StarpuExprBlock <: StarpuExpr
    exprs :: Vector{StarpuExpr}
end

struct StarpuExprCall <: StarpuExpr
    func :: Symbol
    args :: Vector{StarpuExpr}
end
struct StarpuExprCudaCall <: StarpuExpr

    ker_name :: Symbol

    nblocks :: StarpuExpr
    threads_per_block :: StarpuExpr

    args :: Vector{StarpuExpr}

end
struct StarpuExprField <: StarpuExpr

    left :: StarpuExpr
    field :: Symbol

    is_an_arrow :: Bool
end
struct StarpuExprInterval <: StarpuExpr
    start :: StarpuExpr
    step :: StarpuExpr
    stop :: StarpuExpr

    id :: String

    function StarpuExprInterval(start :: StarpuExpr, step :: StarpuExpr, stop :: StarpuExpr ; id :: String = rand_string())
        return new(start, step, stop, id)
    end

end
struct StarpuExprFor <: StarpuExpr

    iter :: Symbol
    set:: StarpuExprInterval
    body :: StarpuExpr

    is_independant :: Bool
    set_declarations :: Vector{StarpuExpr}

end
struct StarpuExprFunction <: StarpuExpr
    ret_type :: Type
    func :: Symbol
    args :: Vector{StarpuExprTypedVar}
    body :: StarpuExpr
end
struct StarpuExprIf <: StarpuExpr
    cond :: StarpuExpr
    then_statement :: StarpuExpr
end


struct StarpuExprIfElse <: StarpuExpr
    cond :: StarpuExpr
    then_statement :: StarpuExpr
    else_statement :: StarpuExpr
end

struct StarpuExprRef <: StarpuExpr
    ref :: StarpuExpr
    indexes :: Vector{StarpuExpr}
end
struct StarpuExprReturn <: StarpuExpr
    value :: StarpuExpr
end
struct StarpuExprBreak <: StarpuExpr
end
struct StarpuExprVar <: StarpuExpr
    name :: Symbol
end
struct StarpuExprInvalid <: StarpuExpr
end

struct StarpuExprValue <: StarpuExpr
    value :: Any
end

struct StarpuExprWhile <: StarpuExpr
    cond :: StarpuExpr
    body :: StarpuExpr
end

struct StarpuExprAddress <: StarpuExpr
    ref :: StarpuExpr
end

function starpu_parse_affect(x :: Expr)

    if (x.head != :(=))
        error("Invalid \"affectation\" expression")
    end

    var = starpu_parse(x.args[1])
    expr = starpu_parse(x.args[2])

    return StarpuExprAffect(var, expr)
end


function equals(x :: StarpuExprAffect, y :: StarpuExpr)

    if typeof(y) != StarpuExprAffect
        return false
    end

    return equals(x.var, y.var) && equals(x.expr, y.expr)
end


function print(io :: IO, x :: StarpuExprAffect ; indent = 0, restrict = false)

    print(io, x.var, indent = indent)
    print(io, " = ")

    need_to_transtyp = isa(x.var, StarpuExprTypedVar) # transtyping to avoid warning (or errors for cuda) during compilation time

    if need_to_transtyp
        print(io, "(", starpu_type_traduction(x.var.typ), ") (")
    end

    print(io, x.expr, indent = indent)

    if need_to_transtyp
        print(io, ")")
    end

end

function apply(func :: Function, expr :: StarpuExprAffect)

    var = apply(func, expr.var)
    new_expr = apply(func, expr.expr)

    return func(StarpuExprAffect(var, new_expr))
end

#======================================================
                BLOCK
(series of instruction, not C variable scoping block)
======================================================#




function is_unwanted(x :: Symbol)
    return false
end

function is_unwanted(x :: LineNumberNode)
    return true
end

function is_unwanted(x :: Expr)
    return false
end

function starpu_parse_block(x :: Expr)
    if (x.head != :block)
        error("Invalid \"block\" expression")
    end    
    exprs = map(starpu_parse, filter(!is_unwanted, x.args))

    return StarpuExprBlock(exprs)
end


function print(io :: IO, x :: StarpuExprBlock ; indent = 0, restrict=false)
    for i in (1 : length(x.exprs))
        print(io, x.exprs[i], indent = indent)
        print(io, ";")
        if (i != length(x.exprs))
            print_newline(io, indent)
        end
    end
end




function apply(func :: Function, expr :: StarpuExprBlock)

    return func(StarpuExprBlock(map((x -> apply(func, x)), expr.exprs)))
end

#======================================================
                FUNCTION CALL
======================================================#




function starpu_parse_call(x :: Expr)

    if (x.head != :call)
        error("Invalid \"call\" expression")
    end

    func = starpu_parse(x.args[1])
    if (x.args[1] == Symbol(":"))
        return starpu_parse_interval(x)
    end
    if (!isa(func, StarpuExprVar))
        error("Invalid \"call\" expression : function must be a variable")
    end

    args = map(starpu_parse, x.args[2:end])

    return StarpuExprCall(func.name, args)
end


starpu_infix_operators = (:(+), :(*), :(-), :(/), :(<), :(>), :(<=), :(>=), :(!=), :(%))


function print_prefix(io :: IO, x :: StarpuExprCall ; indent = 0, restrict=false)

    print(io, x.func, "(")

    for i in (1 : length(x.args))
        if (i != 1)
            print(io, ", ")
        end
        print(io, x.args[i], indent = indent)
    end

    print(io, ")")
end


function print_infix(io :: IO, x :: StarpuExprCall ; indent = 0,restrict=false)
    for i in (1 : length(x.args))
        if (i != 1)
            print(io, " ", x.func, " ")
        end
        print(io, "(")
        print(io, x.args[i], indent = indent)
        print(io, ")")
    end
end

function print(io :: IO, x :: StarpuExprCall ; indent = 0,restrict=false)

    if (length(x.args) >= 2 && x.func in starpu_infix_operators)
        print_infix(io, x, indent = indent)
    else
        print_prefix(io, x, indent = indent)
    end
end




function apply(func :: Function, expr :: StarpuExprCall)

    return func(StarpuExprCall(expr.func, map((x -> apply(func, x)), expr.args)))
end

#======================================================
                CUDA KERNEL CALL
======================================================#





function print(io :: IO, expr :: StarpuExprCudaCall ; indent = 0,restrict=false)

    print_newline(io, indent)
    print(io, expr.ker_name)
    print_newline(io, indent + starpu_indent_size)
    print(io, "<<< ")
    print(io, expr.nblocks, indent = indent + 2 * starpu_indent_size)
    print(io, ", ")
    print(io, expr.threads_per_block, indent = indent + 2 * starpu_indent_size)
    print(io, ", 0, starpu_cuda_get_local_stream()")
    print_newline(io, indent + starpu_indent_size)
    print(io, ">>> (")

    for i in (1 : length(expr.args))

        if (i != 1)
            print(io, ", ")
            if (i % 4 == 1)
                print_newline(io, indent + 2 * starpu_indent_size + 1)
            end
        end

        print(io, expr.args[i], indent = indent + 2 * starpu_indent_size)

    end

    print(io, ");")
    print_newline(io, indent)
    print(io, "cudaError_t status = cudaGetLastError();")
    print_newline(io, indent)
    print(io, "if (status != cudaSuccess) STARPU_CUDA_REPORT_ERROR(status);")
    print_newline(io, indent)

end


function apply(func :: Function, expr :: StarpuExprCudaCall)

    nblocks = func(expr.nblocks)
    threads_per_block = func(expr.threads_per_block)
    args = map((x -> apply(func, x)), expr.args)

    return StarpuExprCudaCall(expr.ker_name, nblocks, threads_per_block, args)
end


#======================================================
                STRUCTURE FIELDS
======================================================#





function starpu_parse_field(x :: Expr)

    if x.head != :(.) || length(x.args) != 2
        error("Invalid parsing of dot expression")
    end

    left = starpu_parse(x.args[1])

    if (!isa(x.args[2], QuoteNode) || !isa(x.args[2].value, Symbol))
        error("Invalid parsing of dot expression")
    end

    return StarpuExprField(left, x.args[2].value, false)
end


function print(io :: IO, x :: StarpuExprField ; indent = 0,restrict=false)
    print(io, "(")
    print(io, x.left, indent = indent)
    print(io, ")", x.is_an_arrow ? "->" : '.', x.field)
end



function apply(func :: Function, expr :: StarpuExprField)
    return func(StarpuExprField(func(expr.left), expr.field, expr.is_an_arrow))
end

#======================================================
                FOR LOOPS
======================================================#





function starpu_parse_for(x :: Expr; is_independant = false)

    if (x.head != :for)
        error("Invalid \"for\" expression")
    end
    affect = x.args[1]

    if (affect.head != :(=))
        error("Invalid \"for\" iterator affectation")
    end

    iter = starpu_parse(affect.args[1])

    if (!isa(iter, StarpuExprVar))
        error("Invalid \"for\" iterator")
    end

    set = starpu_parse(affect.args[2])
    if (!isa(set, StarpuExprInterval))
        error("Set of values in \"for\" loop must be an interval")
    end

    body = starpu_parse(x.args[2])

    return StarpuExprFor(iter.name, set, body, is_independant, StarpuExpr[])
end





function print(io :: IO, x :: StarpuExprFor ; indent = 0,restrict=false)

    print_newline(io, indent)
    print(io, "{")
    indent += starpu_indent_size
    print_newline(io, indent)

    print(io, StarpuExprBlock(x.set_declarations), indent = indent)

    id = x.set.id

    start = "start_" * id
    stop = "stop_" * id
    step = "step_" * id
    dim = "dim_" * id
    iter = "iter_" * id

    print_newline(io, indent, 2)

    if isa(x.set.step, StarpuExprValue)
        print(io, "for ($(x.iter) = $start ; ")
        comparison_op = (x.set.step.value >= 0) ? "<=" : ">="
        print(io, "$(x.iter) $comparison_op $stop ; ")
        print(io, "$(x.iter) += $(x.set.step.value))")

    else
        print(io, "for ($iter = 0, $(x.iter) = $start ; ")
        print(io, "$iter < $dim ; ")
        print(io, "$iter += 1, $(x.iter) += $step)")

    end

    print_newline(io, indent)
    print(io, "{")
    indent += starpu_indent_size

    print_newline(io, indent)
    print(io, x.body, indent = indent)

    indent -= starpu_indent_size
    print_newline(io, indent)
    print(io, "}")

    indent -= starpu_indent_size
    print_newline(io, indent)
    print(io, "}")

    print_newline(io, indent)
end



function apply(func :: Function, expr :: StarpuExprFor)

    set_declarations = map( (x -> apply(func, x)), expr.set_declarations)
    set = apply(func, expr.set)
    body = apply(func, expr.body)

    return func(StarpuExprFor(expr.iter, set, body, expr.is_independant, set_declarations))
end


#======================================================
                FUNCTION DECLARATION
======================================================#




function starpu_parse_function(x :: Expr)

    if (x.head != :function)
        error("Invalid \"function\" expression")
    end

    typed_decl = starpu_parse(x.args[1])

    if (!isa(typed_decl, StarpuExprTypedExpr))
        error("Invalid \"function\" prototype : a return type must me explicited")
    end

    prototype = typed_decl.expr

    if (!isa(prototype, StarpuExprCall))
        error("Invalid \"function\" prototype")
    end

    arg_list = StarpuExprTypedVar[]

    for type_arg in prototype.args
        if (!isa(type_arg, StarpuExprTypedVar))
            error("Invalid \"function\" argument list")
        end
        push!(arg_list, type_arg)
    end

    body = starpu_parse(x.args[2])
    return StarpuExprFunction(typed_decl.typ, prototype.func, arg_list, body)
end



function print(io :: IO, x :: StarpuExprFunction ; indent = 0,restrict=false)

    print(io, starpu_type_traduction(x.ret_type), " ")
    print(io, x.func, '(')

    for i in (1 : length(x.args))

        if (i != 1)
            print(io, ", ")
            if (i % 4 == 1)
                print_newline(io, indent + starpu_indent_size + length(String(x.func)) + 13)
            end
        end
       print(io, x.args[i], indent = indent + starpu_indent_size, restrict = true)
    end

    print(io, ")")
    print_newline(io, indent)
    print(io, "{")
    print_newline(io, indent + starpu_indent_size)
    print(io, x.body, indent = indent + starpu_indent_size)
    print_newline(io, indent)
    print(io, "}\n\n")
    print_newline(io, indent)
end



function apply(func :: Function, expr :: StarpuExprFunction)

    args = map((x -> apply(func, x)), expr.args)
    body = apply(func, expr.body)

    return func(StarpuExprFunction(expr.ret_type, expr.func, args, body))
end


#======================================================
                IF STATEMENT
======================================================#





function starpu_parse_if(x :: Expr)

    if (x.head != :if)
        error("Invalid \"if\" expression")
    end

    len = length(x.args)

    if (len < 2)
        error("Invalid \"if\" statement")
    end

    cond = starpu_parse(x.args[1])
    then_statement = starpu_parse(x.args[2])

    if (len == 2)
        return StarpuExprIf(cond, then_statement)
    end

    else_statement = starpu_parse(x.args[3])

    return StarpuExprIfElse(cond, then_statement, else_statement)
end


function print(io :: IO, x :: Union{StarpuExprIf, StarpuExprIfElse}; indent = 0,restrict=false)

    print_newline(io, indent)
    print(io, "if (")
    print(io, x.cond, indent = indent + starpu_indent_size)
    print(io, ")")
    print_newline(io, indent)
    print(io, "{")
    print_newline(io, indent + starpu_indent_size)
    print(io, x.then_statement, indent = indent + starpu_indent_size)
    print_newline(io, indent)
    print(io, "}")

    if (!isa(x, StarpuExprIfElse))
        return
    end

    print(io, " else")
    print_newline(io, indent)
    print(io, "{")
    print_newline(io, indent + starpu_indent_size)
    print(io, x.else_statement, indent = indent + starpu_indent_size)
    print_newline(io, indent)
    print(io, "}")
    print_newline(io, indent)

end



function apply(func :: Function, expr :: StarpuExprIf)

    cond = apply(func, expr.cond)
    then_statement = apply(func, expr.then_statement)

    return func(StarpuExprIf(cond, then_statement))
end



function apply(func :: Function, expr :: StarpuExprIfElse)

    cond = apply(func, expr.cond)
    then_statement = apply(func, expr.then_statement)
    else_statement = apply(func, expr.else_statement)

    return func(StarpuExprIfElse(cond, then_statement, else_statement))
end

#======================================================
                INTERVALS
======================================================#




function starpu_parse_interval(x :: Expr)

    if (x.head != :(call))
        error("Invalid \"interval\" expression")
    end
    start = starpu_parse(x.args[2])
    steop = starpu_parse(x.args[3])

    if (length(x.args) == 3)
        return StarpuExprInterval(start, StarpuExprValue(1), steop)
    end

    stop = starpu_parse(x.args[4])

    return StarpuExprInterval(start, steop, stop)
end



function apply(func :: Function, expr :: StarpuExprInterval)

    start = apply(func, expr.start)
    step = apply(func, expr.step)
    stop = apply(func, expr.stop)

    return func(StarpuExprInterval(start, step, stop, id = expr.id))
end

#======================================================
                ARRAYS AND REFERENCES
======================================================#




function starpu_parse_ref(x :: Expr)

    if (x.head != :ref)
        error("Invalid \"reference\" expression")
    end

    ref = starpu_parse(x.args[1])
    indexes = map(starpu_parse, x.args[2:end])

    #=
    StarpuExpr[]

    for i in (2 : length(x.args))
        push!(indexes, starpu_parse(x.args[i]))
    end=#

    return StarpuExprRef(ref, indexes)
end



function equals(x :: StarpuExprRef, y :: StarpuExpr)

    if typeof(y) != StarpuExprRef
        return false
    end

    if !equals(x.ref, y.ref) || length(x.indexes) != length(y.indexes)
        return false
    end

    return all(map(equals, x.indexes, y.indexes))
end




function print(io :: IO, x :: StarpuExprRef ; indent = 0,restrict=false)

    print(io, x.ref, indent = indent)

    for i in (1 : length(x.indexes))
        print(io, "[")
        print(io, x.indexes[i], indent = indent)
        print(io, "]")
    end

end

function apply(func :: Function, expr :: StarpuExprRef)

    ref = apply(func, expr.ref)
    indexes = map((x -> apply(func, x)), expr.indexes)

    return func(StarpuExprRef(ref, indexes))
end

function print(io :: IO, x :: StarpuExprAddress ; indent = 0, restrict=false)
    print(io, "&")
    print(io, x.ref, indent = indent)
end

function apply(func :: Function, expr :: StarpuExprAddress)
    ref = apply(func, expr.ref)
    return func(StarpuExprAddress(ref))
end

#======================================================
                BREAK EXPRESSION
======================================================#

function starpu_parse_break(x :: Expr)
    if (x.head != :break)
        error("Invalid \"break\" expression")
    end

    return StarpuExprBreak()
end

function print(io :: IO, x :: StarpuExprBreak ; indent = 0)
    print(io, "break")
end

function apply(func :: Function, expr :: StarpuExprBreak)

    return func(StarpuExprBreak())
end
#======================================================
                RETURN EXPRESSION
======================================================#



function starpu_parse_return(x :: Expr)
    if (x.head != :return)
        error("Invalid \"return\" expression")
    end

    value = starpu_parse(x.args[1])
    # Remove type associated to a single, for a return
    # allows matching with ExprVar
    if (isa(value, StarpuExprTypedVar))
        value = StarpuExprVar(value.name)
    end

    return StarpuExprReturn(value)
end

function print(io :: IO, x :: StarpuExprReturn ; indent = 0,restrict=false)
    print(io, "return ")
    print(io, x.value, indent = indent)
end

function apply(func :: Function, expr :: StarpuExprReturn)

    return func(StarpuExprReturn(apply(func, expr.value)))
end

function apply(func :: Function, expr :: StarpuExpr)
    return func(expr)
end

print(io :: IO, x :: StarpuExprVar ; indent = 0, restrict = false) = print(io, x.name)

function print(io :: IO, x :: StarpuExprValue ; indent = 0,restrict=false)

    value = x.value

    if value == nothing
        return
    end

    if isa(value, AbstractString)
        print(io, '"', value, '"')
        return
    end

    if isa(value, Char)
        print(io, '\'', value, '\'')
        return
    end

    print(io, value)
end





print(io :: IO, x :: StarpuExprInvalid ; indent = 0) = print(io, "INVALID")



function starpu_parse(raw_value :: Any)
    return StarpuExprValue(raw_value)
end

function starpu_parse(sym :: Symbol)
    return StarpuExprVar(sym)
end

#======================================================
                TYPED EXPRESSION
======================================================#



function starpu_parse_typed(x :: Expr)

    if (x.head != :(::))
        error("Invalid type assigned expression")
    end

    expr = starpu_parse(x.args[1])
    typ = nothing

    try
        typ = eval(x.args[2]) :: Type
    catch
        print(x.args[2])
        error("Invalid type in type assigned expression")
    end

    if (isa(expr, StarpuExprVar))
        return StarpuExprTypedVar(expr.name, typ)
    end

    return StarpuExprTypedExpr(expr, typ)
end

function starpu_type_traduction(x)
    if x <: Array
        return starpu_type_traduction(eltype(x)) * "*"
    end

    if x <: Ptr
        depth = 1
        type = eltype(x)
        while type <: Ptr
            depth +=1
            type = eltype(type)
        end

        return starpu_type_traduction(type) * "*"^depth
    end

    return starpu_type_traduction_dict[x]

end

function print(io :: IO, x :: StarpuExprTyped ; indent = 0,restrict=false)

    if (isa(x, StarpuExprTypedVar))
        print(io,starpu_type_traduction(x.typ), " ")
        #if (restrict)
        #    print(io,"restrict ");
        #end
        print(io, x.name)
    else
        print(io, x.expr, indent = indent)
    end
end



function apply(func :: Function, expr :: StarpuExprTypedExpr)

    new_expr = apply(func, expr.expr)

    return func(StarpuExprTypedExpr(new_expr, expr.typ))
end

#======================================================
                While loop
======================================================#


function starpu_parse_while(x :: Expr)

    if (x.head != :while)
        error("Invalid \"while\" loop")
    end

    len = length(x.args)

    if (len < 2)
        error("Invalid \"while\" loop")
    end

    cond = starpu_parse(x.args[1])
    body = starpu_parse(x.args[2])

    return StarpuExprWhile(cond, body)
end


function print(io :: IO, x :: StarpuExprWhile ; indent = 0)
    print_newline(io, indent)
    print(io, "while (")
    print(io, x.cond, indent = indent + starpu_indent_size)
    print(io, ")")
    print_newline(io, indent)
    print(io, "{")
    print_newline(io, indent + starpu_indent_size)
    print(io, x.body, indent = indent + starpu_indent_size)
    print_newline(io, indent)
    print(io, "}")
    print_newline(io, indent)
end



function apply(func :: Function, expr :: StarpuExprWhile)

    cond = apply(func, expr.cond)
    body = apply(func, expr.body)

    return func(StarpuExprWhile(cond, body))
end
