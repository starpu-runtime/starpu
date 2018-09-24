

#======================================================
                CUDA KERNEL CALL
======================================================#



struct StarpuExprCudaCall <: StarpuExpr

    ker_name :: Symbol

    nblocks :: StarpuExpr
    threads_per_block :: StarpuExpr

    args :: Vector{StarpuExpr}

end


function print(io :: IO, expr :: StarpuExprCudaCall ; indent = 0)

    print_newline(io, indent)
    print(io, expr.ker_name)
    print_newline(io, indent + starpu_indent_size)
    print(io, "<<< ")
    print(io, expr.nblocks, indent = indent + 2 * starpu_indent_size)
    print(io, ", ")
    print(io, expr.threads_per_block, indent = indent + 2 * starpu_indent_size)
    print(io, ", 0, starpu_cuda_get_local_stream()")
    print_newline(io, indent + starpu_indent_size)
    print(io, ">>> (")

    for i in (1 : length(expr.args))

        if (i != 1)
            print(io, ", ")
            if (i % 4 == 1)
                print_newline(io, indent + 2 * starpu_indent_size + 1)
            end
        end

        print(io, expr.args[i], indent = indent + 2 * starpu_indent_size)

    end

    print(io, ");")
    print_newline(io, indent)

end


function apply(func :: Function, expr :: StarpuExprCudaCall)

    nblocks = func(expr.nblocks)
    threads_per_block = func(expr.threads_per_block)
    args = map((x -> apply(func, x)), expr.args)

    return StarpuExprCudaCall(expr.ker_name, nblocks, threads_per_block, args)
end
