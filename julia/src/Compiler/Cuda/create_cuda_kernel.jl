

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
    cuda_call = StarpuExprCudaCall(func.func, (@parse nblocks), (@parse THREADS_PER_BLOCK), call_args)
    push!(prekernel_instr, cuda_call)
    push!(prekernel_instr, @parse cudaStreamSynchronize(starpu_cuda_get_local_stream()))
    prekernel_instr = vcat(prekernel_instr, finish)

    prekernel_name = Symbol("CUDA_", func.func)
    prekernel = StarpuExprFunction(Void, prekernel_name, cpu_func.args, StarpuExprBlock(prekernel_instr))
    prekernel = flatten_blocks(prekernel)

    kernel = StarpuExprFunction(Void, func.func, kernel_args, StarpuExprBlock(kernel_instr))
    kernel = add_device_to_interval_call(kernel)
    kernel = flatten_blocks(kernel)
    
    return prekernel, kernel
end
