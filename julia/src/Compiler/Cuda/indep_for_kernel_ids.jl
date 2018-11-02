

function translate_index_code(dims :: Vector{StarpuExprVar})

    ndims = length(dims)

    if ndims == 0
        error("No dimension specified")
    end

    prod = StarpuExprValue(1)
    output = StarpuExpr[]
    reversed_dim = reverse(dims)
    thread_index_patern = @parse € :: Int64 = (€ / €) % €
    thread_id = @parse THREAD_ID

    for i in (1 : ndims)
        index_lvalue = StarpuExprVar(Symbol(:kernel_ids__index_, ndims - i + 1))
        expr = replace_pattern(thread_index_patern, index_lvalue, thread_id, prod, reversed_dim[i])
        push!(output, expr)

        prod = StarpuExprCall(:(*), [prod, reversed_dim[i]])
    end

    thread_id_pattern = @parse begin

        € :: Int64 = blockIdx.x * blockDim.x + threadIdx.x

        if (€ >= €)
            return
        end
    end

    bound_verif = replace_pattern(thread_id_pattern, thread_id, thread_id, prod)
    push!(output, bound_verif)

    return reverse(output)
end







function kernel_index_declarations(ind_for :: StarpuIndepFor)

    pre_kernel_instr = StarpuExpr[]
    kernel_args = StarpuExprTypedVar[]
    kernel_instr = StarpuExpr[]

    decl_pattern = @parse € :: Int64 = €
    interv_size_decl_pattern = @parse € :: Int64 = jlstarpu_interval_size(€, €, €)
    iter_pattern = @parse € :: Int64 = € + € * €

    dims = StarpuExprVar[]
    ker_instr_to_add_later_on = StarpuExpr[]

    for k in (1 : length(ind_for.sets))

        set = ind_for.sets[k]

        start_var = starpu_parse(Symbol(:kernel_ids__start_, k))
        start_decl = replace_pattern(decl_pattern, start_var, set.start)

        step_var = starpu_parse(Symbol(:kernel_ids__step_, k))
        step_decl = replace_pattern(decl_pattern, step_var, set.step)

        dim_var = starpu_parse(Symbol(:kernel_ids__dim_, k))
        dim_decl = replace_pattern(interv_size_decl_pattern, dim_var, start_var, step_var, set.stop)

        push!(dims, dim_var)

        push!(pre_kernel_instr, start_decl, step_decl, dim_decl)
        push!(kernel_args, StarpuExprTypedVar(start_var.name, Int64))
        push!(kernel_args, StarpuExprTypedVar(step_var.name, Int64))
        push!(kernel_args, StarpuExprTypedVar(dim_var.name, Int64))

        iter_var = starpu_parse(ind_for.iters[k])
        index_var = starpu_parse(Symbol(:kernel_ids__index_, k))
        iter_decl = replace_pattern(iter_pattern, iter_var, start_var, index_var, step_var)

        push!(ker_instr_to_add_later_on, iter_decl)
    end


    return dims, ker_instr_to_add_later_on, pre_kernel_instr , kernel_args, kernel_instr
end



function analyse_sets(ind_for :: StarpuIndepFor)


    decl_pattern = @parse € :: Int64 = €
    nblocks_decl_pattern = @parse € :: Int64 = (€ + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK

    dims, ker_instr_to_add, pre_kernel_instr, kernel_args, kernel_instr  = kernel_index_declarations(ind_for)

    dim_prod = @parse 1

    for d in dims
        dim_prod = StarpuExprCall(:(*), [dim_prod, d])
    end

    nthreads_var = @parse nthreads
    nthreads_decl = replace_pattern(decl_pattern, nthreads_var, dim_prod)
    push!(pre_kernel_instr, nthreads_decl)

    nblocks_var = @parse nblocks
    nblocks_decl = replace_pattern(nblocks_decl_pattern, nblocks_var, nthreads_var)
    push!(pre_kernel_instr, nblocks_decl)


    index_decomposition = translate_index_code(dims)

    push!(kernel_instr, index_decomposition...)
    push!(kernel_instr, ker_instr_to_add...)

    return pre_kernel_instr, kernel_args, kernel_instr
end
