

#======================================================
                STRUCTURE FIELDS
======================================================#



struct StarpuExprField <: StarpuExpr

    left :: StarpuExpr
    field :: Symbol

    is_an_arrow :: Bool
end


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


function print(io :: IO, x :: StarpuExprField ; indent = 0)
    print(io, "(")
    print(io, x.left, indent = indent)
    print(io, ")", x.is_an_arrow ? "->" : '.', x.field)
end



function apply(func :: Function, expr :: StarpuExprField)
    return func(StarpuExprField(func(expr.left), expr.field, expr.is_an_arrow))
end
