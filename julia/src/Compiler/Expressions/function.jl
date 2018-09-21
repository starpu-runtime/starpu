

#======================================================
                FUNCTION DECLARATION
======================================================#


struct StarpuExprFunction <: StarpuExpr
    ret_type :: Type
    func :: Symbol
    args :: Vector{StarpuExprTypedVar}
    body :: StarpuExpr
end


function starpu_parse_function(x :: Expr)

    if (x.head != :function)
        error("Invalid \"function\" expression")
    end

    typed_decl = starpu_parse(x.args[1])

    if (!isa(typed_decl, StarpuExprTypedExpr))
        error("Invalid \"function\" prototype : a return type must me explicited")
    end

    prototype = typed_decl.expr

    if (!isa(prototype, StarpuExprCall))
        error("Invalid \"function\" prototype")
    end

    arg_list = StarpuExprTypedVar[]

    for type_arg in prototype.args
        if (!isa(type_arg, StarpuExprTypedVar))
            error("Invalid \"function\" argument list")
        end
        push!(arg_list, type_arg)
    end

    body = starpu_parse(x.args[2])

    return StarpuExprFunction(typed_decl.typ, prototype.func, arg_list, body)
end



function print(io :: IO, x :: StarpuExprFunction ; indent = 0)

    print(io, starpu_type_traduction(x.ret_type), " ")
    print(io, x.func, '(')

    for i in (1 : length(x.args))

        if (i != 1)
            print(io, ", ")
            if (i % 4 == 1)
                print_newline(io, indent + starpu_indent_size + length(String(x.func)) + 13)
            end
        end

        print(io, x.args[i], indent = indent + starpu_indent_size)
    end

    print(io, ")")
    print_newline(io, indent)
    print(io, "{")
    print_newline(io, indent + starpu_indent_size)
    print(io, x.body, indent = indent + starpu_indent_size)
    print_newline(io, indent)
    print(io, "}\n\n")
    print_newline(io, indent)
end



function apply(func :: Function, expr :: StarpuExprFunction)

    args = map((x -> apply(func, x)), expr.args)
    body = apply(func, expr.body)

    return func(StarpuExprFunction(expr.ret_type, expr.func, args, body))
end
