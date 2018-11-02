
#======================================================
                INTERVALS
======================================================#


struct StarpuExprInterval <: StarpuExpr
    start :: StarpuExpr
    step :: StarpuExpr
    stop :: StarpuExpr

    id :: String

    function StarpuExprInterval(start :: StarpuExpr, step :: StarpuExpr, stop :: StarpuExpr ; id :: String = rand_string())
        return new(start, step, stop, id)
    end

end


function starpu_parse_interval(x :: Expr)

    if (x.head != :(:))
        error("Invalid \"interval\" expression")
    end

    start = starpu_parse(x.args[1])
    steop = starpu_parse(x.args[2])

    if (length(x.args) == 2)
        return StarpuExprInterval(start, StarpuExprValue(1), steop)
    end

    stop = starpu_parse(x.args[3])

    return StarpuExprInterval(start, steop, stop)
end



function apply(func :: Function, expr :: StarpuExprInterval)

    start = apply(func, expr.start)
    step = apply(func, expr.step)
    stop = apply(func, expr.stop)

    return func(StarpuExprInterval(start, step, stop, id = expr.id))
end
