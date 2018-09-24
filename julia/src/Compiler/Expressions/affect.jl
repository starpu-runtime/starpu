
#======================================================
                AFFECTATION
======================================================#


struct StarpuExprAffect <: StarpuExpr
    var :: StarpuExpr
    expr :: StarpuExpr
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


function print(io :: IO, x :: StarpuExprAffect ; indent = 0)

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
