

struct StarpuIndepFor

    iters :: Vector{Symbol}
    sets :: Vector{StarpuExprInterval}

    body :: StarpuExpr
end


function assert_no_indep_for(expr :: StarpuExpr)

    function func_to_run(x :: StarpuExpr)

        if (isa(x, StarpuExprFor) && x.is_independant)
            error("Invalid usage of intricated @indep for loops")
        end

        return x
    end

    return apply(func_to_run, expr)
end


function StarpuIndepFor(expr :: StarpuExprFor)

    if !expr.is_independant
        error("For expression must be prefixed by @indep")
    end

    iters = []
    sets = []
    for_loop = expr

    while isa(for_loop, StarpuExprFor) && for_loop.is_independant

        push!(iters, for_loop.iter)
        push!(sets, for_loop.set)
        for_loop = for_loop.body

        while (isa(for_loop, StarpuExprBlock) && length(for_loop.exprs) == 1)
            for_loop = for_loop.exprs[1]
        end
    end

    return StarpuIndepFor(iters, sets, assert_no_indep_for(for_loop))
end
