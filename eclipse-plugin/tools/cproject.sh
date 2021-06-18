#!/bin/bash
#
# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2021       Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

type=$1
shift

if test "$type" == "module"
then
    input='@STARPU_LIB@="true"'
elif test "$type" == "option"
then
     input='<listOptionValue builtIn="false" srcPrefixMapping="" srcRootPath="" value="@STARPU_LIB@"/>'
else
    echo Unknown type $type
    exit 1
fi

for x in $*
do
    echo $input | sed -e 's/@STARPU_LIB@/'$x'/'
done | tr '\012' ' '
