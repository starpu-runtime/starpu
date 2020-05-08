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

struct jl_starpu_codelet
    c_codelet :: starpu_codelet
    perfmodel :: starpu_perfmodel
    cpu_func :: String
    cuda_func :: String
    opencl_func :: String
    modes
end

function starpu_codelet(;
                        cpu_func :: String = "",
                        cuda_func :: String = "",
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
        real_where = ((cpu_func != "") * STARPU_CPU) | ((cuda_func != "") * STARPU_CUDA)
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
    output.c_codelet.cpu_func = load_starpu_function_pointer(cpu_func)
    output.c_codelet.cuda_func = load_starpu_function_pointer(cuda_func)
    output.c_codelet.opencl_func = load_starpu_function_pointer(opencl_func)

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

"""
            starpu_task(; cl :: jl_starpu_codelet, handles :: Vector{StarpuDataHandle}, cl_arg :: Ref)

            Creates a new task which will run the specified codelet on handle buffers and cl_args data
        """
function starpu_task(; cl :: Union{Cvoid, jl_starpu_codelet} = nothing, handles :: Vector{StarpuDataHandle} = StarpuDataHandle[], cl_arg = (),
                     callback :: Union{Cvoid, Function} = nothing, callback_arg = nothing)
    if (cl == nothing)
        error("\"cl\" field can't be empty when creating a StarpuTask")
    end

    output = jl_starpu_task(cl, handles, map((x -> x.object), handles), false, nothing, Vector{Cint}(undef, 1), callback, callback_arg, starpu_task(zero))

    # handle scalar_parameters
    codelet_name = cl.cpu_func
    if isempty(codelet_name)
        codelet_name = cl.cuda_func
    end
    if isempty(codelet_name)
        codelet_name = cl.opencl_func
    end
    if isempty(codelet_name)
        error("No function provided with codelet.")
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

    return output
end


function create_param_struct_from_clarg(name, cl_arg)
    struct_params_name = CODELETS_PARAMS_STRUCT[name]

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

    # Prevent task from being garbage collected. This is necessary for tasks created
    # inside callbacks.
    starpu_task_submit(Ref(task.c_task))

    if task.callback_function != nothing
        callback_arg = task.callback_arg
        callback_signal = task.callback_signal
        callback_function = task.callback_function

        @qbthreads for x in 1:1
            begin
                # Active waiting loop
                # We're doing a fake computation on tmp to prevent optimization.
                tmp = 0
                while task.callback_signal[1] == 0
                    tmp += 1
                end

                # We've received the signal from the pthread, now execute the callback.
                callback_function(callback_arg)

                # Tell the pthread that the callback is done.
                callback_signal[1] = 0

                return callback_signal[1]
            end
        end
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
        cpu_func = CPU_CODELETS[string(expr.args[1])],
        # cuda_func = CUDA_CODELETS[string(expr.args[1])],
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

"""
    Blocks until every submitted task has finished.
"""
function starpu_task_wait_for_all()
    @threadcall(@starpufunc(:starpu_task_wait_for_all),
                Cint, ())
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
