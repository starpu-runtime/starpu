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
    export Link
    mutable struct Link{T}

        data :: T

        previous :: Union{Nothing, Link{T}}
        next :: Union{Nothing, Link{T}}

        list

        function Link{T}(x :: T, l) where {T}
            output = new()
            output.data = x
            output.previous = Nothing()
            output.next = Nothing()
            output.list = l
            return output
        end
    end


    export LinkedList
    mutable struct LinkedList{T}

        nelement :: Int64

        first :: Union{Nothing, Link{T}}
        last :: Union{Nothing, Link{T}}

        function LinkedList{T}() where {T}
            output = new()
            output.nelement = 0
            output.first = Nothing()
            output.last = Nothing()

            return output
        end

    end

    export add_to_head!
    function add_to_head!(l :: LinkedList{T}, el :: T) where {T}

        new_first = Link{T}(el, l)
        old_first = l.first

        l.first = new_first
        new_first.next = old_first

        if (isnothing(old_first))
            l.last = new_first
        else
            old_first.previous = new_first
        end

        l.nelement += 1

        return new_first
    end


    export add_to_tail!
    function add_to_tail!(l :: LinkedList{T}, el :: T) where {T}

        new_last = Link{T}(el, l)
        old_last = l.last

        l.last = new_last
        new_last.previous = old_last

        if (isnothing(old_last))
            l.first = new_last
        else
            old_last.next = new_last
        end

        l.nelement += 1

        return new_last
    end


    function LinkedList(v :: Union{Array{T,N}, NTuple{N,T}}) where {N,T}

        output = LinkedList{T}()

        for x in v
            add_to_tail!(output, x)
        end

        return output
    end


    export remove_link!
    function remove_link!(lnk :: Link{T}) where {T}

        if (lnk.list == nothing)
            return lnk.data
        end

        l = lnk.list
        next = lnk.next
        previous = lnk.previous

        if (isnothing(next))
            l.last = previous
        else
            next.previous = previous
        end

        if (isnothing(previous))
            l.first = next
        else
            previous.next = next
        end

        l.nelement -= 1
        lnk.list = nothing

        return lnk.data
    end


    export is_linked
    function is_linked(lnk :: Link)
        return (lnk.list != nothing)
    end





    export foreach_asc
    macro foreach_asc(list, lnk_iterator, expression)

        quote
            $(esc(lnk_iterator)) = $(esc(list)).first

            while (!isnothing($(esc(lnk_iterator))))
                __next_lnk_iterator = $(esc(lnk_iterator)).next
                $(esc(expression))
                $(esc(lnk_iterator)) = __next_lnk_iterator
            end
        end
    end


    export foreach_desc
    macro foreach_desc(list, lnk_iterator, expression)

        quote
            $(esc(lnk_iterator)) = $(esc(list)).last

            while (!isnothing($(esc(lnk_iterator))))
                __next_lnk_iterator = $(esc(lnk_iterator)).previous
                $(esc(expression))
                $(esc(lnk_iterator)) = __next_lnk_iterator
            end
        end
    end




    function Base.show(io :: IO, lnk :: Link{T}) where {T}

        print(io, "Link{$T}{data: ")
        print(io, lnk.data)

        print(io, " ; previous: ")

        if (isnothing(lnk.previous))
            print(io, "NONE")
        else
            print(io, lnk.previous.data)
        end

        print(io, " ; next: ")

        if (isnothing(lnk.next))
            print(io, "NONE")
        else
            print(io, lnk.next.data)
        end

        print(io, "}")

    end



    function Base.show(io :: IO, l :: LinkedList{T}) where {T}

        print(io, "LinkedList{$T}{")

        @foreach_asc l lnk begin

            if (!isnothing(lnk.previous))
                print(io, ", ")
            end

            print(io, lnk.data)

        end

        print(io, "}")

    end



    #import Base.start
    function start(l :: LinkedList)
        return nothing
    end


    #import Base.done
    function done(l :: LinkedList, state)

        if (state == nothing)
            return isnothing(l.first)
        end

        return isnothing(state.next)
    end


    #import Base.next
    function next(l :: LinkedList, state)

        if (state == nothing)
            next_link = l.first
        else
            next_link = state.next
        end

        return (next_link.data, next_link)
    end


    #import Base.endof
    function endof(l :: LinkedList)
        return l.nelement
    end

    export index_to_link
    function index_to_link(l :: LinkedList, ind)

        if (ind > l.nelement || ind <= 0)
            error("Invalid index")
        end

        lnk = l.first

        for i in (1:(ind - 1))
            lnk = lnk.next
        end

        return lnk
    end


    import Base.getindex
    function getindex(l :: LinkedList, ind)
        return index_to_link(l,ind).data
    end

    import Base.setindex!
    function setindex!(l :: LinkedList{T}, ind, value :: T) where T
        lnk = index_to_link(l,ind)
        lnk.data = value
    end





    import Base.eltype
    function eltype(l :: LinkedList{T}) where T
        return T
    end


    import Base.isempty
    function isempty(l :: LinkedList)
        return (l.nelement == 0)
    end


    import Base.empty!
    function empty!(l :: LinkedList)
        @foreach_asc l lnk remove_link!(lnk)
    end


    import Base.length
    function length(l :: LinkedList)
        return l.nelement
    end
