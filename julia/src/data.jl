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
const StarpuDataHandlePointer = Ptr{Cvoid}
StarpuDataHandle = StarpuDestructible{StarpuDataHandlePointer}

@enum(StarpuDataFilterFunc,
      STARPU_MATRIX_FILTER_VERTICAL_BLOCK = 0,
      STARPU_MATRIX_FILTER_BLOCK = 1,
      STARPU_VECTOR_FILTER_BLOCK = 2,
)

export starpu_data_filter
function starpu_data_filter(filter_func ::StarpuDataFilterFunc, nchildren ::Integer)
    output = starpu_data_filter(zero)
    output.nchildren = UInt32(nchildren)

    if filter_func == STARPU_MATRIX_FILTER_VERTICAL_BLOCK
        output.filter_func = Libdl.dlsym(starpu_wrapper_library_handle, "starpu_matrix_filter_vertical_block")
    elseif filter_func == STARPU_MATRIX_FILTER_BLOCK
        output.filter_func = Libdl.dlsym(starpu_wrapper_library_handle, "starpu_matrix_filter_block")
    else filter_func == STARPU_VECTOR_FILTER_BLOCK
        output.filter_func = Libdl.dlsym(starpu_wrapper_library_handle, "starpu_vector_filter_block")
    end

    return output
end

function starpu_memory_pin(data :: Union{Vector{T}, Matrix{T}}) where T
    starpu_memory_pin(data, sizeof(data))::Cint
end

function starpu_memory_unpin(data :: Union{Vector{T}, Matrix{T}}) where T
    starpu_memory_unpin(data, sizeof(data))::Cint
end

function StarpuNewDataHandle(ptr :: StarpuDataHandlePointer, destr :: Function...) :: StarpuDataHandle
    return StarpuDestructible(ptr, destr...)
end



function starpu_data_unregister_pointer(ptr :: StarpuDataHandlePointer)
    starpu_data_unregister(ptr)
end

function starpu_data_unregister(handles :: StarpuDataHandle...)
    for h in handles
        starpu_execute_destructor!(h, starpu_data_unregister_pointer)
    end
end

function starpu_data_register(v :: Vector{T}) where T
    output = Ref{Ptr{Cvoid}}(0)
    data_pointer = pointer(v)

    starpu_vector_data_register(output, STARPU_MAIN_RAM, data_pointer, length(v), sizeof(T))
    return StarpuNewDataHandle(output[], starpu_data_unregister_pointer)#, [starpu_data_unregister_pointer])
end

function starpu_data_register(m :: Matrix{T}) where T

    output = Ref{Ptr{Cvoid}}(0)
    data_pointer = pointer(m)
    (height, width) = size(m)

    starpu_matrix_data_register(output, STARPU_MAIN_RAM, data_pointer, height, height, width, sizeof(T))
    return StarpuNewDataHandle(output[], starpu_data_unregister_pointer)#, [starpu_data_unregister_pointer])
end

function starpu_data_register(block :: Array{T,3}) where T

    output = Ref{Ptr{Cvoid}}(0)
    data_pointer = pointer(block)
    (height, width, depth) = size(block)

    starpu_block_data_register(output, STARPU_MAIN_RAM, data_pointer, height, height * width, height, width, depth, sizeof(T))
    return StarpuNewDataHandle(output[], starpu_data_unregister_pointer)
end

function starpu_data_register(ref :: Ref{T}) where T

    output = Ref{Ptr{Cvoid}}(0)

    starpu_variable_data_register(output, STARPU_MAIN_RAM, ref, sizeof(T))
    return StarpuNewDataHandle(output[], starpu_data_unregister_pointer)
end

function starpu_data_register(x1, x2, next_args...)

    handle_1 = starpu_data_register(x1)
    handle_2 = starpu_data_register(x2)

    next_handles = map(starpu_data_register, next_args)

    return [handle_1, handle_2, next_handles...]
end

import Base.getindex
function Base.getindex(handle :: StarpuDataHandle, indexes...)
    output = starpu_data_get_sub_data(handle.object, length(indexes),
                                      map(x->x-1, indexes)...)
    return StarpuNewDataHandle(output)
end

function starpu_data_unpartition_pointer(ptr :: StarpuDataHandlePointer)
    starpu_data_unpartition(ptr, STARPU_MAIN_RAM)
end

function starpu_data_partition(handle :: StarpuDataHandle, filter :: starpu_data_filter)

    starpu_add_destructor!(handle, starpu_data_unpartition_pointer)
    starpu_data_partition(handle.object, pointer_from_objref(filter))
end

function starpu_data_unpartition(handles :: StarpuDataHandle...)

    for h in handles
        starpu_execute_destructor!(h, starpu_data_unpartition_pointer)
    end

    return nothing
end

function starpu_data_map_filters(handle :: StarpuDataHandle, filter :: starpu_data_filter)
    starpu_add_destructor!(handle, starpu_data_unpartition_pointer)
    starpu_data_map_filters(handle.object, 1, pointer_from_objref(filter))
end

function starpu_data_map_filters(handle :: StarpuDataHandle, filter_1 :: starpu_data_filter, filter_2 :: starpu_data_filter)
    starpu_add_destructor!(handle, starpu_data_unpartition_pointer)
    starpu_data_map_filters(handle.object, 2, pointer_from_objref(filter_1), pointer_from_objref(filter_2))
end

function starpu_data_get_sequential_consistency_flag(handle :: StarpuDataHandle)
    return starpu_data_get_sequential_consistency_flag(handle.object)
end

function starpu_data_set_sequential_consistency_flag(handle :: StarpuDataHandle, flag :: Int)
    starpu_data_set_sequential_consistency_flag(handle.object, flag)
end

function starpu_data_acquire_on_node(handle :: StarpuDataHandle, node :: Int, mode)
    starpu_data_acquire_on_node(handle.object, node, mode)
end

function starpu_data_release_on_node(handle :: StarpuDataHandle, node :: Int)
    starpu_data_release_on_node(handle.object, node)
end

function starpu_data_wont_use(handle :: StarpuDataHandle)
    starpu_data_wont_use(handle.object)
end

function repl(x::Symbol)
    return x
end
function repl(x::Number)
    return x
end
function repl(x :: Expr)
    if (x.head == :call && x.args[1] == :+)
        if (x.args[2] == :_)
            return x.args[3]
        elseif (x.args[3] == :_)
            return x.args[2]
        else return Expr(:call,:+,repl(x.args[2]),repl(x.args[3]))
        end
    elseif (x.head == :call && x.args[1] == :-)
        if (x.args[2] == :_)
            return Expr(:call,:-,x.args[3])
        elseif (x.args[3] == :_)
            return x.args[2]
        else return Expr(:call,:-,repl(x.args[2]),repl(x.args[3]))
        end
    else return Expr(:call,x.args[1],repl(x.args[2]),repl(x.args[3]))
    end
end
"""
    Declares a subarray.
    Ex : @starpu_filter ha = A[ _:_+1, : ] 
 
"""
macro starpu_filter(expr)
    #dump(expr, maxdepth=20)
    if (expr.head==Symbol("="))
        region = expr.args[2]
        if (region.head == Symbol("ref"))
            farray = expr.args[1]
            println("starpu filter")
            index = 0
            filter2=nothing
            filter3=nothing
            if (region.args[2]==Symbol(":"))
                index = 3
                filter2=:(STARPU_MATRIX_FILTER_BLOCK)
            elseif (region.args[3] == Symbol(":"))
                index = 2
                filter3=:(STARPU_MATRIX_FILTER_VERTICAL_BLOCK)
            else
            end
            ex = repl(region.args[index].args[3])
            if (region.args[index].args[2] != Symbol("_"))
                throw(AssertionError("LHS must be _"))
            end
            ret = quote
                # escape and not global for farray!
                $(esc(farray)) = starpu_data_register($(esc(region.args[1])))
                starpu_data_partition( $(esc(farray)),starpu_data_filter($(esc(filter)),$(esc(ex))))
            end
            return ret
        else
            ret = quote
                $(esc(farray))= starpu_data_register($(esc(region.args[1])))
            end
            
            dump("coucou"); #dump(region.args[2])
            #                dump(region.args[2])
            #                dump(region.args[3])
            return ret
        end
    end
end
