#!/bin/bash
# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2009-2025   University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

for v in $(grep -i '<dt>' doc/doxygen/chapters/starpu_installation/environment_variables.doxy | sed -e 's/<dt>//' -e 's/<\/dt>//')
do
    l=$(grep -rs $v . | grep starpu_getenv | wc -l)
    if test "$l" == "0"
    then
	echo $v $l
    else
	echo $v
    fi
done
