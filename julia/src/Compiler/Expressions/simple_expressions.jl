

abstract type StarpuExpr end


function apply(func :: Function, expr :: StarpuExpr)
    return func(expr)
end




struct StarpuExprVar <: StarpuExpr
    name :: Symbol
end

print(io :: IO, x :: StarpuExprVar ; indent = 0) = print(io, x.name)



struct StarpuExprValue <: StarpuExpr
    value :: Any
end


function print(io :: IO, x :: StarpuExprValue ; indent = 0)

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




struct StarpuExprInvalid <: StarpuExpr
end

print(io :: IO, x :: StarpuExprInvalid ; indent = 0) = print(io, "INVALID")



function starpu_parse(raw_value :: Any)
    return StarpuExprValue(raw_value)
end

function starpu_parse(sym :: Symbol)
    return StarpuExprVar(sym)
end
