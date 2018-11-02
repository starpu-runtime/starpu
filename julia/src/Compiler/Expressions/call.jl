
#======================================================
                FUNCTION CALL
======================================================#


struct StarpuExprCall <: StarpuExpr
    func :: Symbol
    args :: Vector{StarpuExpr}
end


function starpu_parse_call(x :: Expr)

    if (x.head != :call)
        error("Invalid \"call\" expression")
    end

    func = starpu_parse(x.args[1])

    if (!isa(func, StarpuExprVar))
        error("Invalid \"call\" expression : function must be a variable")
    end

    args = map(starpu_parse, x.args[2:end])

    return StarpuExprCall(func.name, args)
end


starpu_infix_operators = (:(+), :(*), :(-), :(/), :(<), :(>), :(<=), :(>=), :(%))


function print_prefix(io :: IO, x :: StarpuExprCall ; indent = 0)

    print(io, x.func, "(")

    for i in (1 : length(x.args))
        if (i != 1)
            print(io, ", ")
        end
        print(io, x.args[i], indent = indent)
    end

    print(io, ")")
end


function print_infix(io :: IO, x :: StarpuExprCall ; indent = 0)
    for i in (1 : length(x.args))
        if (i != 1)
            print(io, " ", x.func, " ")
        end
        print(io, "(")
        print(io, x.args[i], indent = indent)
        print(io, ")")
    end
end

function print(io :: IO, x :: StarpuExprCall ; indent = 0)

    if (length(x.args) >= 2 && x.func in starpu_infix_operators)
        print_infix(io, x, indent = indent)
    else
        print_prefix(io, x, indent = indent)
    end
end




function apply(func :: Function, expr :: StarpuExprCall)

    return func(StarpuExprCall(expr.func, map((x -> apply(func, x)), expr.args)))
end
