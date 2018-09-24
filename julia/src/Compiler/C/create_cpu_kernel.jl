




function transform_to_cpu_kernel(expr :: StarpuExprFunction)

    output = add_for_loop_declarations(expr)
    output = substitute_args(output)
    output = substitute_func_calls(output)
    output = substitute_indexing(output)
    output = flatten_blocks(output)

    return output
end
