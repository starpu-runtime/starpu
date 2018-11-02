
#======================================================
                RETURN EXPRESSION
======================================================#


struct StarpuExprReturn <: StarpuExpr
    value :: StarpuExpr
end

function starpu_parse_return(x :: Expr)

    if (x.head != :return)
        error("Invalid \"return\" expression")
    end

    value = starpu_parse(x.args[1])

    return StarpuExprReturn(value)
end


function print(io :: IO, x :: StarpuExprReturn ; indent = 0)
    print(io, "return ")
    print(io, x.value, indent = indent)
end



function apply(func :: Function, expr :: StarpuExprReturn)

    return func(StarpuExprReturn(apply(func, expr.value)))
end
