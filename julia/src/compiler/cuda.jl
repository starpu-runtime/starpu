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

function translate_cublas(expr :: StarpuExpr)
    function func_to_run(x :: StarpuExpr)
        # STARPU_BLAS => (CUBLAS, TRANS, FILLMODE, ALPHA, SIDE, DIAG)
        blas_to_cublas = Dict(:STARPU_SGEMM  => (:cublasSgemm, [1, 2], [], [6, 11], [], []),
                              :STARPU_DGEMM  => (:cublasDgemm, [1, 2], [], [6, 11], [], []),
                              :STARPU_SGEMV  => (:cublasSgemv, [1], [], [4,9], [], []),
                              :STARPU_DGEMV  => (:cublasDgemv, [1], [], [4,9], [], []),
                              :STARPU_SSCAL  => (:cublasSscal, [], [], [2], [], []),
                              :STARPU_DSCAL  => (:cublasDscal, [], [], [2], [], []),
                              :STARPU_STRSM  => (:cublasStrsm, [3], [2], [7], [1], [4]),
                              :STARPU_DTRSM  => (:cublasDtrsm, [3], [2], [7], [1], [4]),
                              :STARPU_SSYR   => (:cublasSsyr, [], [1], [3], [], []),
                              :STARPU_SSYRK  => (:cublasSsyrk, [2], [1], [5,8], [], []),
                              :STARPU_SGER   => (:cublasSger, [], [], [3], [], []),
                              :STARPU_DGER   => (:cublasDger, [], [], [3], [], []),
                              :STARPU_STRSV  => (:cublasStrsv, [2], [1], [], [], [3]),
                              :STARPU_STRMM  => (:cublasStrmm, [3], [2], [7], [1], [4]),
                              :STARPU_DTRMM  => (:cublasDtrmm, [3], [2], [7], [1], [4]),
                              :STARPU_STRMV  => (:cublasStrmv, [2], [1], [], [], [3]),
                              :STARPU_SAXPY  => (:cublasSaxpy, [], [], [2], [], []),
                              :STARPU_DAXPY  => (:cublasDaxpy, [], [], [2], [], []),
                              :STARPU_SSWAP  => (:cublasSswap, [], [], [], [], []),
                              :STARPU_DSWAP  => (:cublasDswap, [], [], [], [], []))

        if !(isa(x, StarpuExprCall) && x.func in keys(blas_to_cublas))
            return x
        end

        new_args = x.args

        # cublasOperation_t parameters (e.g. StarpuExprValue("N")
        for i in blas_to_cublas[x.func][2]
            if !isa(new_args[i], StarpuExprValue) || !isa(new_args[i].value, String)
                error("Argument $i of ", x.func, " must be a string")
            end

            value = new_args[i].value

            if value == "N" || value == "n"
                new_args[i] = StarpuExprVar(:CUBLAS_OP_N)
            elseif value == "T" || value == "t"
                new_args[i] = StarpuExprVar(:CUBLAS_OP_T)
            elseif value == "C" || value == "c"
                new_args[i] = StarpuExprVar(:CUBLAS_OP_C)
            else
                error("Unhandled value for rgument $i of ", x.func, ": ", value,
                      "expecting (\"N\", \"T\", or \"C\")")
            end
        end

        # cublasFillMode_t parameters (e.g. StarpuExprValue("L")
        for i in blas_to_cublas[x.func][3]
            if !isa(new_args[i], StarpuExprValue) || !isa(new_args[i].value, String)
                error("Argument $i of ", x.func, " must be a string")
            end

            value = new_args[i].value

            if value == "L" || value == "l"
                new_args[i] = StarpuExprVar(:CUBLAS_FILL_MODE_LOWER)
            elseif value == "U" || value == "u"
                new_args[i] = StarpuExprVar(:CUBLAS_FILL_MODE_UPPER)
            else
                error("Unhandled value for rgument $i of ", x.func, ": ", value,
                      "expecting (\"L\" or \"U\")")
            end
        end

        # scalar parameters (alpha, beta, ...):  alpha -> &alpha
        for i in blas_to_cublas[x.func][4]
            if !isa(new_args[i], StarpuExprVar)
                error("Argument $i of ", x.func, " must be a variable")
            end
            var_name = new_args[i].name
            new_args[i] = StarpuExprVar(Symbol("&$var_name"))
        end

        # cublasSideMode_t parameters (e.g. StarpuExprValue("L")
        for i in blas_to_cublas[x.func][5]
            if !isa(new_args[i], StarpuExprValue) || !isa(new_args[i].value, String)
                error("Argument $i of ", x.func, " must be a string, got: ", new_args[i])
            end

            value = new_args[i].value

            if value == "L" || value == "l"
                new_args[i] = StarpuExprVar(:CUBLAS_SIDE_LEFT)
            elseif value == "R" || value == "r"
                new_args[i] = StarpuExprVar(:CUBLAS_SIDE_RIGHT)
            else
                error("Unhandled value for rgument $i of ", x.func, ": ", value,
                      "expecting (\"L\" or \"R\")")
            end
        end

        # cublasDiag_Typet parameters (e.g. StarpuExprValue("N")
        for i in blas_to_cublas[x.func][6]
            if !isa(new_args[i], StarpuExprValue) || !isa(new_args[i].value, String)
                error("Argument $i of ", x.func, " must be a string")
            end

            value = new_args[i].value

            if value == "N" || value == "n"
                new_args[i] = StarpuExprVar(:CUBLAS_DIAG_NON_UNIT)
            elseif value == "U" || value == "u"
                new_args[i] = StarpuExprVar(:CUBLAS_DIAG_UNIT)
            else
                error("Unhandled value for rgument $i of ", x.func, ": ", value,
                      "expecting (\"N\" or \"U\")")
            end
        end

        new_args = [@parse(starpu_cublas_get_local_handle()), x.args...]

        status_varname = "status"*rand_string()
        status_var = StarpuExprVar(Symbol("cublasStatus_t "*status_varname))
        call_expr = StarpuExprCall(blas_to_cublas[x.func][1], new_args)

        return StarpuExprBlock([StarpuExprAffect(status_var, call_expr),
                                starpu_parse(Meta.parse("""if $status_varname != CUBLAS_STATUS_SUCCESS
                                                              STARPU_CUBLAS_REPORT_ERROR($status_varname)
                                                          end""")),
                                @parse cudaStreamSynchronize(starpu_cuda_get_local_stream())])
    end

    return apply(func_to_run, expr)
end

function get_all_assignments(cpu_instr)
    ret = StarpuExpr[]

    function func_to_run(x :: StarpuExpr)
        if isa(x, StarpuExprAffect)
            push!(ret, x)
        end

        return x
    end

    apply(func_to_run, cpu_instr)
    return ret
end

function get_all_buffer_vars(cpu_instr)
    ret = StarpuExprTypedVar[]
    assignments = get_all_assignments(cpu_instr)
    for x in assignments
        var = x.var
        expr = x.expr
        if isa(expr, StarpuExprCall) && expr.func in [:STARPU_MATRIX_GET_PTR, :STARPU_VECTOR_GET_PTR]
            push!(ret, var)
        end
    end

    return ret
end

function get_all_buffer_stores(cpu_instr, vars)
    ret = StarpuExprAffect[]

    function func_to_run(x :: StarpuExpr)
        if isa(x, StarpuExprAffect) && isa(x.var, StarpuExprRef) && isa(x.var.ref, StarpuExprVar) &&
            x.var.ref.name in map(x -> x.name, vars)
            push!(ret, x)
        end

        return x
    end

    apply(func_to_run, cpu_instr)
    return ret
end

function get_all_buffer_refs(cpu_instr, vars)
    ret = []

    current_instr = nothing
    InstrTy = Union{StarpuExprAffect,
                    StarpuExprCall,
                    StarpuExprCudaCall,
                    StarpuExprFor,
                    StarpuExprIf,
                    StarpuExprIfElse,
                    StarpuExprReturn,
                    StarpuExprBreak,
                    StarpuExprWhile}
    parent = nothing

    function func_to_run(x :: StarpuExpr)
        if isa(x, InstrTy) && !(isa(x, StarpuExprCall) && x.func in [:(+), :(-), :(*), :(/), :(%), :(<), :(<=), :(==), :(!=), :(>=), :(>), :sqrt])
            current_instr = x
        end

        if isa(x, StarpuExprRef) && isa(x.ref, StarpuExprVar) && x.ref.name in map(x -> x.name, vars) && # var[...]
            !isa(parent, StarpuExprAddress) && # filter &var[..]
            !(isa(current_instr, StarpuExprAffect) && current_instr.var == x) # filter lhs ref
            push!(ret, (current_instr, x))
        end

        parent = x
        return x
    end

    visit_preorder(func_to_run, cpu_instr)
    return ret
end

function transform_cuda_device_loadstore(cpu_instr :: StarpuExprBlock)
    # Get all CUDA buffer pointers
    buffer_vars = get_all_buffer_vars(cpu_instr)

    buffer_types = Dict{Symbol, Type}()
    for var in buffer_vars
        buffer_types[var.name] = var.typ
    end

    # Get all store to a CUDA buffer
    stores = get_all_buffer_stores(cpu_instr, buffer_vars)

    # Get all load from CUDA buffer
    loads = get_all_buffer_refs(cpu_instr, buffer_vars)

    # Replace each load L:
    # L: ... buffer[id]
    # With the following instruction block:
    # Type varX
    # cudaMemcpy(&varX, &buffer[id], sizeof(Type), cudaMemcpyDeviceToHost)
    # L: ... varX
    for l in loads
        (instr, ref) = l
        block = []
        buffer = ref.ref.name
        varX = "var"*rand_string()
        type = buffer_types[Symbol(buffer)]
        ctype = starpu_type_traduction(eltype(type))
        push!(block, StarpuExprTypedVar(Symbol(varX), eltype(type)))
        push!(block, StarpuExprCall(:cudaMemcpy,
                                    [StarpuExprAddress(StarpuExprVar(Symbol(varX))),
                                     StarpuExprAddress(ref),
                                     StarpuExprVar(Symbol("sizeof($ctype)")),
                                     StarpuExprVar(:cudaMemcpyDeviceToHost)]))
        push!(block, substitute(instr, ref, StarpuExprVar(Symbol("$varX"))))

        cpu_instr = substitute(cpu_instr, instr, StarpuExprBlock(block))
    end

    # Replace each Store S:
    # S: buffer[id] = expr
    # With the following instruction block:
    # Type varX
    # varX = expr
    # cudaMemcpy(&buffer[id], &varX, sizeof(Type), cudaMemcpyHostToDevice)
    for s in stores
        block = []
        buffer = s.var.ref.name
        varX = "var"*rand_string()
        type = buffer_types[Symbol(buffer)]
        ctype = starpu_type_traduction(eltype(type))
        push!(block, StarpuExprTypedVar(Symbol(varX), eltype(type)))
        push!(block, StarpuExprAffect(StarpuExprVar(Symbol("$varX")), s.expr))
        push!(block, StarpuExprCall(:cudaMemcpy,
                                    [StarpuExprAddress(s.var),
                                     StarpuExprAddress(StarpuExprVar(Symbol(varX))),
                                     StarpuExprVar(Symbol("sizeof($ctype)")),
                                     StarpuExprVar(:cudaMemcpyHostToDevice)]))

        cpu_instr = substitute(cpu_instr, s, StarpuExprBlock(block))
    end

    return cpu_instr
end

function transform_to_cuda_kernel(func :: StarpuExprFunction)

    cpu_func = transform_to_cpu_kernel(func)

    init, indep, finish = extract_init_indep_finish(cpu_func.body)

    cpu_instr = init
    kernel = nothing

    # Generate a CUDA kernel only if there is an independent loop (@parallel macro).
    if (indep != nothing)
        prekernel_instr, kernel_args, kernel_instr = analyse_sets(indep)

        kernel_call = StarpuExprCudaCall(:cudaKernel, (@parse nblocks), (@parse THREADS_PER_BLOCK), StarpuExpr[])
        cpu_instr = vcat(cpu_instr, prekernel_instr)
        kernel_instr = vcat(kernel_instr, indep.body)

        indep_for_def, indep_for_undef = analyse_variable_declarations(StarpuExprBlock(kernel_instr), kernel_args)
        prekernel_def, prekernel_undef = analyse_variable_declarations(StarpuExprBlock(cpu_instr), cpu_func.args)

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
        push!(cpu_instr, cuda_call)
        push!(cpu_instr, @parse cudaStreamSynchronize(starpu_cuda_get_local_stream()))
        kernel = StarpuExprFunction(Nothing, kernelname, kernel_args, StarpuExprBlock(kernel_instr))
        kernel = add_device_to_interval_call(kernel)
        kernel = flatten_blocks(kernel)
    end

    cpu_instr = vcat(cpu_instr, finish)
    cpu_instr = StarpuExprBlock(cpu_instr)
    cpu_instr = transform_cuda_device_loadstore(cpu_instr)

    prekernel_name = Symbol("CUDA_", func.func)
    prekernel = StarpuExprFunction(Nothing, prekernel_name, cpu_func.args, cpu_instr)
    prekernel = translate_cublas(prekernel)
    prekernel = flatten_blocks(prekernel)

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
