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
"""
        Object used to store a lot of function which must
        be applied to and object
    """
mutable struct StarpuDestructible{T}

    object :: T
    destructors :: LinkedList{Function}

end

starpu_block_list = Vector{LinkedList{StarpuDestructible}}()

"""
    Declares a block of code. Every declared StarpuDestructible in this code
    will execute its destructors on its object, once the block is exited
"""
macro starpu_block(expr)
    quote
        starpu_enter_new_block()
        local z=$(esc(expr))
        starpu_exit_block()
        z
    end
end


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

function starpu_destruct!(x :: StarpuDestructible)

    @foreach_asc  x.destructors destr begin
        destr.data(x.object)
    end

    empty!(x.destructors)

    return nothing
end


function starpu_exit_block()

    destr_list = pop!(starpu_block_list)

    @foreach_asc destr_list x begin
        starpu_destruct!(x.data)
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
