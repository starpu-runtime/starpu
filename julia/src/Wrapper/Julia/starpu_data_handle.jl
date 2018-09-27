

STARPU_MAIN_RAM = 0 #TODO: ENUM


const StarpuDataHandlePointer = Ptr{Void}



StarpuDataHandle = StarpuDestructible{StarpuDataHandlePointer}



function StarpuNewDataHandle(ptr :: StarpuDataHandlePointer, destr :: Function...) :: StarpuDataHandle
    return StarpuDestructible(ptr, destr...)
end



function starpu_data_unregister_pointer(ptr :: StarpuDataHandlePointer)
    @starpucall(starpu_data_unregister, Void, (Ptr{Void},), ptr)
end


export starpu_data_unregister
function starpu_data_unregister(handles :: StarpuDataHandle...)
    for h in handles
        starpu_execute_destructor!(h, starpu_data_unregister_pointer)
    end
end



export starpu_data_register

function starpu_data_register(v :: Vector{T}) where T

    output = Ref{Ptr{Void}}(0)
    data_pointer = pointer(v)

    @starpucall(starpu_vector_data_register,
                Void,
                (Ptr{Void}, Cint, Ptr{Void}, UInt32, Csize_t),
                output, STARPU_MAIN_RAM, data_pointer,
                length(v), sizeof(T)
            )

    return StarpuNewDataHandle(output[], starpu_data_unregister_pointer)#, [starpu_data_unregister_pointer])
end


function starpu_data_register(m :: Matrix{T}) where T

    output = Ref{Ptr{Void}}(0)
    data_pointer = pointer(m)
    (height, width) = size(m)

    @starpucall(starpu_matrix_data_register,
                Void,
                (Ptr{Void}, Cint, Ptr{Void},
                    UInt32, UInt32, UInt32, Csize_t),
                output, STARPU_MAIN_RAM, data_pointer,
                height, height, width, sizeof(T)
            )

    return StarpuNewDataHandle(output[], starpu_data_unregister_pointer)#, [starpu_data_unregister_pointer])
end


function starpu_data_register(block :: Array{T,3}) where T

    output = Ref{Ptr{Void}}(0)
    data_pointer = pointer(block)
    (height, width, depth) = size(block)

    @starpucall(starpu_block_data_register,
                Void,
                (Ptr{Void}, Cint, Ptr{Void},
                    UInt32, UInt32, UInt32, UInt32,
                    UInt32, Csize_t),
                output, STARPU_MAIN_RAM, data_pointer,
                height, height * width,
                height, width, depth,
                sizeof(T)
            )

    return StarpuNewDataHandle(output[], starpu_data_unregister_pointer)
end



function starpu_data_register(ref :: Ref{T}) where T

    output = Ref{Ptr{Void}}(0)

    @starpucall(starpu_variable_data_register,
                Void,
                (Ptr{Void}, Cint, Ptr{Void}, Csize_t),
                output, STARPU_MAIN_RAM, ref, sizeof(T)
            )

    return StarpuNewDataHandle(output[], starpu_data_unregister_pointer)
end



function starpu_data_register(x1, x2, next_args...)

    handle_1 = starpu_data_register(x1)
    handle_2 = starpu_data_register(x2)

    next_handles = map(starpu_data_register, next_args)

    return [handle_1, handle_2, next_handles...]
end




export starpu_data_get_sub_data

function starpu_data_get_sub_data(root_data :: StarpuDataHandle, id)

    output = @starpucall(starpu_data_get_sub_data,
                        Ptr{Void}, (Ptr{Void}, Cuint, Cuint),
                        root_data.object, 1, id - 1
                    )

    return StarpuNewDataHandle(output)
end


function starpu_data_get_sub_data(root_data :: StarpuDataHandle, idx, idy)

    output = @starpucall(starpu_data_get_sub_data,
                        Ptr{Void}, (Ptr{Void}, Cuint, Cuint, Cuint),
                        root_data.object, 2, idx - 1, idy - 1
                    )

    return StarpuNewDataHandle(output)
end

import Base.getindex



function Base.getindex(handle :: StarpuDataHandle, indexes...)
     starpu_data_get_sub_data(handle, indexes...)
 end




export StarpuDataFilterFunc
export STARPU_MATRIX_FILTER_VERTICAL_BLOCK, STARPU_MATRIX_FILTER_BLOCK

@enum(StarpuDataFilterFunc,

    STARPU_MATRIX_FILTER_VERTICAL_BLOCK = 0,
    STARPU_MATRIX_FILTER_BLOCK = 1
)

export StarpuDataFilter
"""
    TODO : use real function pointers loaded from starpu shared library
"""
mutable struct StarpuDataFilter

    filter_func :: StarpuDataFilterFunc
    nchildren :: Cuint

    function StarpuDataFilter(filter_func, nchildren)
        output = new()
        output.filter_func = filter_func
        output.nchildren = Cuint(nchildren)
        return output
    end

end


function starpu_data_unpartition_pointer(ptr :: StarpuDataHandlePointer)
    @starpucall(starpu_data_unpartition, Void, (Ptr{Void}, Cuint), ptr, STARPU_MAIN_RAM)
end

export starpu_data_partition
function starpu_data_partition(handle :: StarpuDataHandle, filter :: StarpuDataFilter)

    starpu_add_destructor!(handle, starpu_data_unpartition_pointer)

    @starpucall(jlstarpu_data_partition,
            Void, (Ptr{Void}, Ptr{Void}),
            handle.object, Ref{StarpuDataFilter}(filter)
        )
end


export starpu_data_unpartition
function starpu_data_unpartition(handles :: StarpuDataHandle...)

    for h in handles
        starpu_execute_destructor!(h, starpu_data_unpartition_pointer)
    end

    return nothing
end



export starpu_data_map_filters

function starpu_data_map_filters(handle :: StarpuDataHandle, filter :: StarpuDataFilter)

    starpu_add_destructor!(handle, starpu_data_unpartition_pointer)

    @starpucall(jlstarpu_data_map_filters_1_arg,
            Void, (Ptr{Void}, Ptr{Void}),
            handle.object, Ref{StarpuDataFilter}(filter)
    )
end


function starpu_data_map_filters(handle :: StarpuDataHandle, filter_1 :: StarpuDataFilter, filter_2 :: StarpuDataFilter)

    starpu_add_destructor!(handle, starpu_data_unpartition_pointer)

    @starpucall(jlstarpu_data_map_filters_2_arg,
            Void, (Ptr{Void}, Ptr{Void}, Ptr{Void}),
            handle.object,
            Ref{StarpuDataFilter}(filter_1),
            Ref{StarpuDataFilter}(filter_2)
    )

end
