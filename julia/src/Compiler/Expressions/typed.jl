
#======================================================
                TYPED EXPRESSION
======================================================#







abstract type StarpuExprTyped <: StarpuExpr end


struct StarpuExprTypedVar <: StarpuExprTyped
    name :: Symbol
    typ :: Type
end

struct StarpuExprTypedExpr <: StarpuExprTyped # TODO : remove typed expression ?
    expr :: StarpuExpr
    typ :: Type
end


function starpu_parse_typed(x :: Expr)

    if (x.head != :(::))
        error("Invalid type assigned expression")
    end

    expr = starpu_parse(x.args[1])
    typ = nothing

    try
        typ = eval(x.args[2]) :: Type
    catch
        error("Invalid type in type assigned expression")
    end

    if (isa(expr, StarpuExprVar))
        return StarpuExprTypedVar(expr.name, typ)
    end

    return StarpuExprTypedExpr(expr, typ)
end





starpu_type_traduction_dict = Dict(
    Void => "void",
    Int32 => "int32_t",
    UInt32 => "uint32_t",
    Float32 => "float",
    Int64 => "int64_t",
    UInt64 => "uint64_t",
    Float64 => "double"
)



function starpu_type_traduction(x)

    if x <: Array
        return starpu_type_traduction_array(x)
    end

    if x <: Ptr
        return starpu_type_traduction(eltype(x)) * "*"
    end

    return starpu_type_traduction_dict[x]

end


function starpu_type_traduction_array(x :: Type{Array{T,N}}) where {T,N}

    output = starpu_type_traduction(T)

    for i in (1 : N)
        output *= "*"
    end

    return output
end



function print(io :: IO, x :: StarpuExprTyped ; indent = 0)

    if (isa(x, StarpuExprTypedVar))
        print(io, starpu_type_traduction(x.typ), " ")
        print(io, x.name)
    else
        print(io, x.expr, indent = indent)
    end
end



function apply(func :: Function, expr :: StarpuExprTypedExpr)

    new_expr = apply(func, expr.expr)

    return func(StarpuExprTypedExpr(new_expr, expr.typ))
end
