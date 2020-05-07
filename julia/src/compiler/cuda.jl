# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2020       Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
#
# StarPU is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation; either version 2.1 of the License, or (at
# your option) any later version.
#
# StarPU is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
# See the GNU Lesser General Public License in COPYING.LGPL for more details.
#


function is_indep_for_expr(x :: StarpuExpr)
    return isa(x, StarpuExprFor) && x.is_independant
end


function extract_init_indep_finish(expr :: StarpuExpr) # TODO : it is not a correct extraction (example : if (cond) {@indep for ...} else {return} would not work)
                                                            # better use apply() (NOTE :assert_no_indep_for already exists) to find recursively every for loops
    init = StarpuExpr[]
    finish = StarpuExpr[]

    if is_indep_for_expr(expr)
        return init, StarpuIndepFor(expr), finish
    end

    if !isa(expr, StarpuExprBlock)
        return [expr], nothing, finish
    end

    for i in (1 : length(expr.exprs))

        if !is_indep_for_expr(expr.exprs[i])
            continue
        end

        init = expr.exprs[1 : i-1]
        indep = StarpuIndepFor(expr.exprs[i])
        finish = expr.exprs[i+1 : end]

        if any(is_indep_for_expr, finish)
            error("Sequence of several independant loops is not allowed") #same it may be tricked by a Block(Indep_for(...))
        end

        return init, indep, finish
    end

    return expr.exprs, nothing, finish
end




function analyse_variable_declarations(expr :: StarpuExpr, already_defined :: Vector{StarpuExprTypedVar} = StarpuExprTypedVar[])

    undefined_variables = Symbol[]
    defined_variable_names = map((x -> x.name), already_defined)
    defined_variable_types = map((x -> x.typ), already_defined)

    function func_to_apply(x :: StarpuExpr)

        if isa(x, StarpuExprFunction)
            error("No function declaration allowed in this section")
        end

        if isa(x, StarpuExprVar) || isa(x, StarpuExprTypedVar)

            if !(x.name in defined_variable_names) && !(x.name in undefined_variables)
                push!(undefined_variables, x.name)
            end

            return x
        end

        if isa(x, StarpuExprAffect) || isa(x, StarpuExprFor)

            if isa(x, StarpuExprAffect)

                var = x.var

                if !isa(var, StarpuExprTypedVar)
                    return x
                end

                name = var.name
                typ = var.typ

            else
                name = x.iter
                typ = Int64
            end

            if name in defined_variable_names
                error("Multiple definition of variable $name")
            end

            filter!((sym -> sym != name), undefined_variables)
            push!(defined_variable_names, name)
            push!(defined_variable_types, typ)

            return x
        end

        return x
    end

    apply(func_to_apply, expr)
    defined_variable = map(StarpuExprTypedVar, defined_variable_names, defined_variable_types)

    return defined_variable, undefined_variables
end



function find_variable(name :: Symbol, vars :: Vector{StarpuExprTypedVar})

    for x in vars
        if x.name == name
            return x
        end
    end

    return nothing
end



function add_device_to_interval_call(expr :: StarpuExpr)

    function func_to_apply(x :: StarpuExpr)

        if isa(x, StarpuExprCall) && x.func == :jlstarpu_interval_size
            return StarpuExprCall(:jlstarpu_interval_size__device, x.args)
        end

        return x
    end

    return apply(func_to_apply, expr)
end



function transform_to_cuda_kernel(func :: StarpuExprFunction)

    cpu_func = transform_to_cpu_kernel(func)

    init, indep, finish = extract_init_indep_finish(cpu_func.body)

    if indep == nothing
        error("No independant for loop has been found") # TODO can fail because extraction is not correct yet
    end

    prekernel_instr, kernel_args, kernel_instr = analyse_sets(indep)

    kernel_call = StarpuExprCudaCall(:cudaKernel, (@parse nblocks), (@parse THREADS_PER_BLOCK), StarpuExpr[])
    prekernel_instr = vcat(init, prekernel_instr)
    kernel_instr = vcat(kernel_instr, indep.body)

    indep_for_def, indep_for_undef = analyse_variable_declarations(StarpuExprBlock(kernel_instr), kernel_args)
    prekernel_def, prekernel_undef = analyse_variable_declarations(StarpuExprBlock(prekernel_instr), cpu_func.args)

    for undef_var in indep_for_undef

        found_var = find_variable(undef_var, prekernel_def)

        if found_var == nothing # TODO : error then ?
            continue
        end

        push!(kernel_args, found_var)
    end

    call_args = map((x -> StarpuExprVar(x.name)), kernel_args)
    kernelname=Symbol("KERNEL_",func.func);
    cuda_call = StarpuExprCudaCall(kernelname, (@parse nblocks), (@parse THREADS_PER_BLOCK), call_args)
    push!(prekernel_instr, cuda_call)
    push!(prekernel_instr, @parse cudaStreamSynchronize(starpu_cuda_get_local_stream()))
    prekernel_instr = vcat(prekernel_instr, finish)

    prekernel_name = Symbol("CUDA_", func.func)
    prekernel = StarpuExprFunction(Nothing, prekernel_name, cpu_func.args, StarpuExprBlock(prekernel_instr))
    prekernel = flatten_blocks(prekernel)

    kernel = StarpuExprFunction(Nothing, kernelname, kernel_args, StarpuExprBlock(kernel_instr))
    kernel = add_device_to_interval_call(kernel)
    kernel = flatten_blocks(kernel)
    
    return prekernel, kernel
end


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
