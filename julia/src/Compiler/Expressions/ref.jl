
#======================================================
                ARRAYS AND REFERENCES
======================================================#


struct StarpuExprRef <: StarpuExpr
    ref :: StarpuExpr
    indexes :: Vector{StarpuExpr}
end


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




function print(io :: IO, x :: StarpuExprRef ; indent = 0)

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
