


const jlstarpu_allocated_structures = Vector{Ptr{Void}}([])



function jlstarpu_allocate_and_store(x_c :: T) where {T}

    allocated_ptr = Ptr{T}(Libc.malloc(sizeof(T)))

    if (allocated_ptr == C_NULL)
        error("Base.Libc.malloc returned NULL")
    end

    unsafe_store!(allocated_ptr, x_c)
    push!(jlstarpu_allocated_structures, Ptr{Void}(allocated_ptr))

    return allocated_ptr
end



function jlstarpu_free_allocated_structures()
    map(Libc.free, jlstarpu_allocated_structures)
    empty!(jlstarpu_allocated_structures)
    return nothing
end
