

#======================================================
                IF STATEMENT
======================================================#



struct StarpuExprIf <: StarpuExpr
    cond :: StarpuExpr
    then_statement :: StarpuExpr
end


struct StarpuExprIfElse <: StarpuExpr
    cond :: StarpuExpr
    then_statement :: StarpuExpr
    else_statement :: StarpuExpr
end


function starpu_parse_if(x :: Expr)

    if (x.head != :if)
        error("Invalid \"if\" expression")
    end

    len = length(x.args)

    if (len < 2)
        error("Invalid \"if\" statement")
    end

    cond = starpu_parse(x.args[1])
    then_statement = starpu_parse(x.args[2])

    if (len == 2)
        return StarpuExprIf(cond, then_statement)
    end

    else_statement = starpu_parse(x.args[3])

    return StarpuExprIfElse(cond, then_statement, else_statement)
end


function print(io :: IO, x :: Union{StarpuExprIf, StarpuExprIfElse}; indent = 0)

    print_newline(io, indent)
    print(io, "if (")
    print(io, x.cond, indent = indent + starpu_indent_size)
    print(io, ")")
    print_newline(io, indent)
    print(io, "{")
    print_newline(io, indent + starpu_indent_size)
    print(io, x.then_statement, indent = indent + starpu_indent_size)
    print_newline(io, indent)
    print(io, "}")

    if (!isa(x, StarpuExprIfElse))
        return
    end

    print(io, " else")
    print_newline(io, indent)
    print(io, "{")
    print_newline(io, indent + starpu_indent_size)
    print(io, x.else_statement, indent = indent + starpu_indent_size)
    print_newline(io, indent)
    print(io, "}")
    print_newline(io, indent)

end



function apply(func :: Function, expr :: StarpuExprIf)

    cond = apply(func, expr.cond)
    then_statement = apply(func, expr.then_statement)

    return func(StarpuExprIf(cond, then_statement))
end



function apply(func :: Function, expr :: StarpuExprIfElse)

    cond = apply(func, expr.cond)
    then_statement = apply(func, expr.then_statement)
    else_statement = apply(func, expr.else_statement)

    return func(StarpuExprIfElse(cond, then_statement, else_statement))
end
