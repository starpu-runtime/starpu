
#======================================================
                FOR LOOPS
======================================================#


struct StarpuExprFor <: StarpuExpr

    iter :: Symbol
    set:: StarpuExprInterval
    body :: StarpuExpr

    is_independant :: Bool
    set_declarations :: Vector{StarpuExpr}

end



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





function print(io :: IO, x :: StarpuExprFor ; indent = 0)

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
    print_newline(io, indent + starpu_indent_size)
    print(io, x.body, indent = indent + starpu_indent_size)
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
