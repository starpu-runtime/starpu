
#======================================================
                BLOCK
(series of instruction, not C variable scoping block)
======================================================#


struct StarpuExprBlock <: StarpuExpr
    exprs :: Vector{StarpuExpr}
end


function is_unwanted(x :: Symbol)
    return false
end

function is_unwanted(x :: Expr)

    if (x.head == :line)
        return true
    end

    return false
end


function starpu_parse_block(x :: Expr)

    if (x.head != :block)
        error("Invalid \"block\" expression")
    end

    exprs = map(starpu_parse, filter(!is_unwanted, x.args))

    #=for y in x.args

        if (is_unwanted(y))
            continue
        end

        push!(exprs, starpu_parse(y))
    end
    =#
    #if (length(exprs) == 1)
    #    return exprs[1]  #TODO : let 1 instruction blocks be a thing ?
    #end

    return StarpuExprBlock(exprs)
end


function print(io :: IO, x :: StarpuExprBlock ; indent = 0)
    for i in (1 : length(x.exprs))
        print(io, x.exprs[i], indent = indent)
        print(io, ";")
        if (i != length(x.exprs))
            print_newline(io, indent)
        end
    end
end




function apply(func :: Function, expr :: StarpuExprBlock)

    return func(StarpuExprBlock(map((x -> apply(func, x)), expr.exprs)))
end
