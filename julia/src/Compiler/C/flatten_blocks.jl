


function flatten_blocks(expr :: StarpuExpr)

    function func_to_run(x :: StarpuExpr)

        if !isa(x, StarpuExprBlock)
            return x
        end

        instrs = StarpuExpr[]

        for sub_expr in x.exprs

            if isa(sub_expr, StarpuExprBlock)
                push!(instrs, sub_expr.exprs...)
            else
                push!(instrs, sub_expr)
            end
        end

        return StarpuExprBlock(instrs)
    end

    return apply(func_to_run, expr)
end
