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
dirname=$(dirname $0)
cd $dirname/../../../

DIRS="src tools mpi/src"
#sc_hypervisor/src"

for d in $DIRS
do
    STARPU_C_FILES=$(find $d  -name '*.c' | tr '\012' ' ')
    #echo "Checking $STARPU_C_FILES"
    for e in "\bmalloc(" "\bcalloc(" "\brealloc("
    do
	#echo $e
	grep -n "$e" $STARPU_C_FILES | while read line
	do
	    #echo "----------------------------------------------------------------"
	    #echo "$line"
	    file=$(echo $line | awk -F':' '{print $1}')
	    count=$(echo $line | awk -F':' '{print $2}')
	    count1=$(( count + 1 ))
	    line1=$(grep -n "" $file | grep "^$count1:")
	    #    echo "$line1"
	    c=$(echo "$line1" | grep -c -E "\!|NULL")
	    #    echo $c
	    if test $c -eq 0
	    then
		echo "$line"
	    fi
	done
    done
done

