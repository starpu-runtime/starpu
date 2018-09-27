


"""
    Object used to store a lost of function which must
    be applied to and object
"""
mutable struct StarpuDestructible{T}

    object :: T
    destructors :: LinkedList{Function}

end

starpu_block_list = Vector{LinkedList{StarpuDestructible}}()



function StarpuDestructible(obj :: T, destructors :: Function...) where T

    if (isempty(starpu_block_list))
        error("Creation of a StarpuDestructible object while not beeing in a @starpu_block")
    end

    l = LinkedList{Function}()

    for destr in destructors
        add_to_tail!(l, destr)
    end

    output = StarpuDestructible{T}(obj, l)
    add_to_head!(starpu_block_list[end], output)

    return output
end




function starpu_enter_new_block()

    push!(starpu_block_list, LinkedList{StarpuDestructible}())
end

"""
    Applies every stored destructores to the StarpuDestructible stored object
"""
function starpu_destruct!(x :: StarpuDestructible)

    for destr in x.destructors
        destr(x.object)
    end

    empty!(x.destructors)

    return nothing
end


function starpu_exit_block()

    destr_list = pop!(starpu_block_list)

    for x in destr_list
        starpu_destruct!(x)
    end

end

"""
    Adds new destructors to the list of function. They will be executed before
        already stored ones when calling starpu_destruct!
"""
function starpu_add_destructor!(x :: StarpuDestructible, destrs :: Function...)

    for d in destrs
        add_to_head!(x.destructors, d)
    end

    return nothing
end

"""
    Removes detsructor without executing it
"""
function starpu_remove_destructor!(x :: StarpuDestructible, destr :: Function)

    @foreach_asc x.destructors lnk begin

        if (lnk.data == destr)
            remove_link!(lnk)
            break
        end
    end

    return nothing
end


"""
    Executes "destr" function. If it was one of the stored destructors, it
    is removed.
    This function can be used to allow user to execute a specific action manually
        (ex : explicit call to starpu_data_unpartition() without unregistering)
"""
function starpu_execute_destructor!(x :: StarpuDestructible, destr :: Function)

    starpu_remove_destructor!(x, destr)
    return destr(x.object)
end


export @starpu_block

"""
    Declares a block of code. Every declared StarpuDestructible in this code
    will execute its destructors on its object, once the block is exited
"""
macro starpu_block(expr)
    quote
        starpu_enter_new_block()
        $(esc(expr))
        starpu_exit_block()
    end
end
