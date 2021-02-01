#!/bin/bash
# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2017-2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
DIR=$(echo $(dirname $0)/../../../)

SGREP="grep --exclude-dir=.git --binary-files=without-match"

for m in $(grep undef $DIR/src/common/config.h.in | awk '{print $2}' | grep -v "^PACKAGE")
do
    #echo Check macro $m
    used=$($SGREP -rsl $m | grep -v Makefile | grep -v "^src" | grep -v configure | grep -v autom4 | grep -v "mpi/src" | grep -v "tests/helper.h" | grep -v m4 | grep -v doc )
    if test -n "$used"
    then
	#echo "Checking $m is defined in include config"
	count=$(grep -c $m $DIR/include/starpu_config.h.in)
	if test $count == 0
	then
	    echo "Error: Macro $m is not defined in include/starpu_config.h.in"
	    echo "but is used in"
	    echo $used |  tr ' ' '\012'
	    echo
	fi
    fi
done


