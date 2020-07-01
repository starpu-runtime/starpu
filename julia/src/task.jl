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
using ThreadPools

mutable struct jl_starpu_codelet
    c_codelet :: starpu_codelet
    perfmodel :: starpu_perfmodel
    cpu_func :: Union{String, STARPU_BLAS}
    cuda_func :: Union{String, STARPU_BLAS}
    opencl_func :: String
    modes
end

global codelet_list = Vector{jl_starpu_codelet}()

function starpu_codelet(;
                        cpu_func :: Union{String, STARPU_BLAS, Cvoid} = "",
                        cuda_func :: Union{String, STARPU_BLAS, Cvoid} = "",
                        opencl_func :: String = "",
                        modes = [],
                        perfmodel :: starpu_perfmodel,
                        where_to_execute :: Union{Cvoid, UInt32} = nothing,
                        color :: UInt32 = 0x00000000
                        )

    if (length(modes) > STARPU_NMAXBUFS)
        error("Codelet has too much buffers ($(length(modes)) but only $STARPU_NMAXBUFS are allowed)")
    end


    if (where_to_execute == nothing)
        real_where = ((cpu_func != nothing) * STARPU_CPU) | ((cuda_func != nothing) * STARPU_CUDA)
    else
        real_where = where_to_execute
    end

    output = jl_starpu_codelet(starpu_codelet(zero), perfmodel, cpu_func, cuda_func, opencl_func, modes)
    ## TODO: starpu_codelet_init

    output.c_codelet.where = real_where

    for i in 1:length(modes)
        output.c_codelet.modes[i] = modes[i]
    end
    output.c_codelet.nbuffers = length(modes)
    output.c_codelet.model = pointer_from_objref(perfmodel)
    output.c_codelet.color = color

    if typeof(cpu_func) == STARPU_BLAS
        output.cpu_func = cpu_blas_codelets[cpu_func]
        output.c_codelet.cpu_func = load_wrapper_function_pointer(output.cpu_func)
    else
        output.c_codelet.cpu_func = load_starpu_function_pointer(get(CPU_CODELETS, cpu_func, ""))
    end

    if typeof(cuda_func) == STARPU_BLAS
        output.cuda_func = cuda_blas_codelets[cuda_func]
        output.c_codelet.cuda_func = load_wrapper_function_pointer(output.cuda_func)
        output.c_codelet.cuda_flags[1] = STARPU_CUDA_ASYNC
    else
        output.c_codelet.cuda_func = load_starpu_function_pointer(get(CUDA_CODELETS, cuda_func, ""))
    end

    output.c_codelet.opencl_func = load_starpu_function_pointer("")

    # Codelets must not be garbage collected before starpu shutdown is called.
    lock(mutex)
    push!(codelet_list, output)
    unlock(mutex)

    return output
end

mutable struct jl_starpu_task

    cl :: jl_starpu_codelet
    handles :: Vector{StarpuDataHandle}
    handle_pointers :: Vector{StarpuDataHandlePointer}
    synchronous :: Bool
    cl_arg # type depends on codelet
    callback_signal :: Vector{Cint}
    callback_function :: Union{Cvoid, Function}
    callback_arg
    c_task :: starpu_task
end

task_list = Vector{jl_starpu_task}()

"""
            starpu_task(; cl :: jl_starpu_codelet, handles :: Vector{StarpuDataHandle}, cl_arg :: Ref)

            Creates a new task which will run the specified codelet on handle buffers and cl_args data
        """
function starpu_task(;
                     cl :: Union{Cvoid, jl_starpu_codelet} = nothing,
                     handles :: Vector{StarpuDataHandle} = StarpuDataHandle[],
                     cl_arg = (),
                     callback :: Union{Cvoid, Function} = nothing,
                     callback_arg = nothing,
                     tag :: Union{Cvoid, starpu_tag_t} = nothing,
                     tag_only :: Union{Cvoid, starpu_tag_t} = nothing,
                     sequential_consistency = true,
                     detach = 1,
                     color :: Union{Cvoid, UInt32} = nothing,
                     where :: Union{Cvoid, Int32} = nothing)
    if (cl == nothing)
        error("\"cl\" field can't be empty when creating a StarpuTask")
    end

    output = jl_starpu_task(cl, handles, map((x -> x.object), handles), false, nothing, Vector{Cint}(undef, 1), callback, callback_arg, starpu_task(zero))

    # handle scalar_parameters
    codelet_name = ""
    if isa(cl.cpu_func, String) && cl.cpu_func != ""
        codelet = cl.cpu_func
    elseif isa(cl.gpu_func, String) && cl.gpu_func != ""
        codelet = cl.gpu_func
    end
    scalar_parameters = get(CODELETS_SCALARS, codelet_name, nothing)
    if scalar_parameters != nothing
        nb_scalar_required = length(scalar_parameters)
        nb_scalar_provided = tuple_len(cl_arg)
        if (nb_scalar_provided != nb_scalar_required)
            error("$nb_scalar_provided scalar parameters provided but $nb_scalar_required are required by $codelet_name.")
        end
        output.cl_arg = create_param_struct_from_clarg(codelet_name, cl_arg)
    else
        output.cl_arg = cl_arg
    end

    starpu_task_init(Ref(output.c_task))
    output.c_task.cl = pointer_from_objref(cl.c_codelet)
    output.c_task.synchronous = false
    output.c_task.sequential_consistency = sequential_consistency
    output.c_task.detach = detach

    ## TODO: check num handles equals num codelet buffers
    for i in 1:length(handles)
        output.c_task.handles[i] = output.handle_pointers[i]
    end
    if tuple_len(cl_arg) > 0
        output.c_task.cl_arg = Base.unsafe_convert(Ptr{Cvoid}, Ref(output.cl_arg))
        output.c_task.cl_arg_size = sizeof(output.cl_arg)
    end

    # callback
    if output.callback_function != nothing
        output.callback_signal[1] = 0
        output.c_task.callback_arg = Base.unsafe_convert(Ptr{Cvoid}, output.callback_signal)
        output.c_task.callback_func = load_wrapper_function_pointer("julia_callback_func")
    end

    if tag != nothing
        output.c_task.tag_id = tag
        output.c_task.use_tag = 1
    end

    if tag_only != nothing
        output.c_task.tag_id = tag_only
    end

    if color != nothing
        output.c_task.color = color
    end

    if where != nothing
        output.c_task.where = where
    end

    # Tasks must not be garbage collected before starpu_task_wait_for_all is called.
    # This is necessary in particular for tasks created inside callback functions.
    lock(mutex)
    push!(task_list, output)
    unlock(mutex)

    return output
end


function create_param_struct_from_clarg(codelet_name, cl_arg)
    struct_params_name = CODELETS_PARAMS_STRUCT[codelet_name]

    if struct_params_name == false
        error("structure name not found in CODELET_PARAMS_STRUCT")
    end

    nb_scalar_provided = length(cl_arg)
    create_struct_param_str = "output = $struct_params_name("
    for i in 1:nb_scalar_provided-1
        arg = cl_arg[i]
        create_struct_param_str *= "$arg, "
        end
    if (nb_scalar_provided > 0)
        arg = cl_arg[nb_scalar_provided]
        create_struct_param_str *= "$arg"
    end
    create_struct_param_str *= ")"
    eval(Meta.parse(create_struct_param_str))
    return output
end

"""
    Launches task execution, if "synchronous" task field is set to "false", call
    returns immediately
"""
function starpu_task_submit(task :: jl_starpu_task)
    if (length(task.handles) != length(task.cl.modes))
        error("Invalid number of handles for task : $(length(task.handles)) where given while codelet has $(task.cl.modes) modes")
    end

    starpu_task_submit(Ref(task.c_task))

    if task.callback_function != nothing
        callback_arg = task.callback_arg
        callback_signal = task.callback_signal
        callback_function = task.callback_function

        lock(mutex)
        put!(task_pool) do

            # Active waiting loop
            @starpucall(julia_wait_signal, Cvoid, (Ptr{Cvoid},), Base.unsafe_convert(Ptr{Cvoid}, callback_signal))

            # We've received the signal from the pthread, now execute the callback.
            callback_function(callback_arg)

            # Tell the pthread that the callback is done.
            callback_signal[1] = 0
        end
        unlock(mutex)
    end
end

function starpu_modes(x :: Symbol)
    if (x == Symbol("STARPU_RW"))
        return STARPU_RW
    elseif (x == Symbol("STARPU_R"))
        return STARPU_R
    else return STARPU_W
    end
end

default_codelet = Dict{String, jl_starpu_codelet}()
default_perfmodel = Dict{String, starpu_perfmodel}()

function get_default_perfmodel(name)
    if name in keys(default_perfmodel)
        return default_perfmodel[name]
    end

    perfmodel = starpu_perfmodel(
        perf_type = starpu_perfmodel_type(STARPU_HISTORY_BASED),
        symbol = name
    )
    default_perfmodel[name] = perfmodel
    return perfmodel
end

function get_default_codelet(codelet_name, perfmodel, modes) :: jl_starpu_codelet
    if codelet_name in keys(default_codelet)
        return default_codelet[codelet_name]
    end

    cl = starpu_codelet(
        cpu_func  = codelet_name in keys(CPU_CODELETS) ? codelet_name : "",
        cuda_func = codelet_name in keys(CUDA_CODELETS) ? codelet_name : "",
        modes = modes,
        perfmodel = perfmodel,
    )
    default_codelet[codelet_name] = cl
    return cl
end

function starpu_task_insert(;
                            codelet_name :: Union{Cvoid, String} = nothing,
                            cl :: Union{Cvoid, jl_starpu_codelet} = nothing,
                            perfmodel :: Union{starpu_perfmodel, Cvoid} = nothing,
                            handles :: Vector{StarpuDataHandle} = StarpuDataHandle[],
                            cl_arg = (),
                            callback :: Union{Cvoid, Function} = nothing,
                            callback_arg = nothing,
                            tag :: Union{Cvoid, starpu_tag_t} = nothing,
                            tag_only :: Union{Cvoid, starpu_tag_t} = nothing,
                            sequential_consistency = true,
                            detach = 1,
                            where :: Union{Cvoid, Int32} = nothing,
                            color :: Union{Cvoid, UInt32} = nothing,
                            modes = nothing)
    if cl == nothing && codelet_name == nothing
        error("At least one of the two parameters codelet_name or cl must be provided when calling starpu_task_insert.")

    end
    if cl == nothing && modes == nothing
        error("Modes must be defined when calling starpu_task_insert without a codelet.")
    end

    if perfmodel == nothing
        perfmodel = get_default_perfmodel(codelet_name == nothing ? "default" : codelet_name)
    end

    if cl == nothing
        cl = get_default_codelet(codelet_name, perfmodel, modes)
    end

    task = starpu_task(cl = cl, handles = handles, cl_arg = cl_arg, callback = callback,
                       callback_arg = callback_arg, tag = tag, tag_only = tag_only,
                       sequential_consistency = sequential_consistency,
                       detach = detach, color = color, where = where)

    starpu_task_submit(task)
end

"""
    Creates and submits an asynchronous task running cl Codelet function.
    Ex : @starpu_async_cl cl(handle1, handle2)
"""
macro starpu_async_cl(expr, modes, cl_arg=(), color ::UInt32=0x00000000)

    if (!isa(expr, Expr) || expr.head != :call)
        error("Invalid task submit syntax")
    end
    if (!isa(expr, Expr)||modes.head != :vect)
        error("Invalid task submit syntax")
    end
    perfmodel = starpu_perfmodel(
        perf_type = starpu_perfmodel_type(STARPU_HISTORY_BASED),
        symbol = "history_perf"
    )
    println(CPU_CODELETS[string(expr.args[1])])
    cl = starpu_codelet(
        cpu_func  = string(expr.args[1]),
        cuda_func = string(expr.args[1]),
        #opencl_func="ocl_matrix_mult",
        ### TODO: CORRECT !
        modes = map((x -> starpu_modes(x)),modes.args),
        perfmodel = perfmodel,
        color = color
    )
    handles = Expr(:vect, expr.args[2:end]...)
    #dump(handles)
    quote
        task = starpu_task(cl = $(esc(cl)), handles = $(esc(handles)), cl_arg=$(esc(cl_arg)))
        starpu_task_submit(task)
    end
end

function starpu_task_wait(task :: jl_starpu_task)
    @threadcall(@starpufunc(:starpu_task_wait),
                Cint, (Ptr{Cvoid},), Ref(task.c_task))

    # starpu_task_wait(Ref(task.c_task))
end


"""
    Blocks until every submitted task has finished.
"""
function starpu_task_wait_for_all()
    @threadcall(@starpufunc(:starpu_task_wait_for_all),
                Cint, ())

    lock(mutex)
    empty!(task_list)
    unlock(mutex)
end

"""
    Blocks until every submitted task has finished.
    Ex : @starpu_sync_tasks begin
                [...]
                starpu_task_submit(task)
                [...]
        end

    TODO : Make the macro only wait for tasks declared inside the following expression.
            (similar mechanism as @starpu_block)
"""
macro starpu_sync_tasks(expr)
    quote
        $(esc(expr))
        starpu_task_wait_for_all()
    end
end

function starpu_task_destroy(task :: jl_starpu_task)
    starpu_task_destroy(Ref(task.c_task))
end
