#!/bin/bash

# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2011  Centre National de la Recherche Scientifique
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

stcolor=$(tput sgr0)
redcolor=$(tput setaf 1)

functions=$(grep 'starpu.*(.*);' include/*.h | awk -F':' '{print $2}' | sed 's/(.*//' | sed 's/.* //'| tr -d ' ' | tr -d '*')

for func in $functions ; do
    #echo Processing function $func
    x=$(grep $func doc/starpu.texi doc/chapters/*texi | grep deftypefun)
    if test "$x" == "" ; then
        echo "Error. Function ${redcolor}${func}${stcolor} is not (or incorrectly) documented"
    fi
done


