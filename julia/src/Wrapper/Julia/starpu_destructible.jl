



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


function starpu_add_destructor!(x :: StarpuDestructible, destrs :: Function...)

    for d in destrs
        add_to_head!(x.destructors, d)
    end

    return nothing
end


function starpu_remove_destructor!(x :: StarpuDestructible, destr :: Function)

    @foreach_asc x.destructors lnk begin

        if (lnk.data == destr)
            remove_link!(lnk)
            break
        end
    end

    return nothing
end

function starpu_execute_destructor!(x :: StarpuDestructible, destr :: Function)

    starpu_remove_destructor!(x, destr)
    return destr(x.object)
end


export @starpu_block
macro starpu_block(expr)
    quote
        starpu_enter_new_block()
        $(esc(expr))
        starpu_exit_block()
    end
end



if false

@starpu_block let
    println("Begining of block")
    x = StarpuDestructible(1, println)
    println("End of block")
end



@starpu_block let
    println("Begining of block")
    x = StarpuDestructible(2, (x -> @show x), println)
    println("End of block")
end


@starpu_block let
    println("Begining of block")
    x = StarpuDestructible(3, (x -> @show x), println)
    starpu_add_destructor!(x, (x -> @show x+1))
    println("End of block")
end

@starpu_block let
    println("Begining of block")
    x = StarpuDestructible(4, (x -> @show x), println)
    starpu_add_destructor!(x, (x -> @show x+1))
    starpu_remove_destructor!(x, println)
    println("End of block")
end

@starpu_block let
    println("Begining of block")
    x = StarpuDestructible(4, (x -> @show x), println)
    starpu_add_destructor!(x, (x -> @show x+1))
    starpu_execute_destructor!(x, println)
    println("End of block")
end

end
