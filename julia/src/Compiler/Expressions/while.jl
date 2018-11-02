
#======================================================
                While loop
======================================================#


struct StarpuExprWhile <: StarpuExpr
    cond :: StarpuExpr
    body :: StarpuExpr
end

function starpu_parse_while(x :: Expr)

    if (x.head != :while)
        error("Invalid \"while\" loop")
    end

    len = length(x.args)

    if (len < 2)
        error("Invalid \"while\" loop")
    end

    cond = starpu_parse(x.args[1])
    body = starpu_parse(x.args[2])

    return StarpuExprWhile(cond, body)
end


function print(io :: IO, x :: StarpuExprWhile ; indent = 0)
    print_newline(io, indent)
    print(io, "while (")
    print(io, x.cond, indent = indent + starpu_indent_size)
    print(io, ")")
    print_newline(io, indent)
    print(io, "{")
    print_newline(io, indent + starpu_indent_size)
    print(io, x.body, indent = indent + starpu_indent_size)
    print_newline(io, indent)
    print(io, "}")
    print_newline(io, indent)
end



function apply(func :: Function, expr :: StarpuExprWhile)

    cond = apply(func, expr.cond)
    body = apply(func, expr.body)

    return func(StarpuExprWhile(cond, body))
end
