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
import Base.print

function print_newline(io :: IO, indent = 0, n_lines = 1)
    for i in (1 : n_lines)
        print(io, "\n")
    end

    for i in (1 : indent)
        print(io, " ")
    end
end

starpu_indent_size = 4

function rand_char()
    r = rand(UInt) % 62

    if (0 <= r < 10)
        return '0' + r
    elseif (10 <= r < 36)
        return 'a' + (r - 10)
    else
        return 'A' + (r - 36)
    end
end

function rand_string(size = 8)
    output = ""

    for i in (1 : size)
        output *= string(rand_char())
    end
    return output
end

function system(cmd :: String)
    ccall((:system, "libc"), Cint, (Cstring,), cmd)
end
