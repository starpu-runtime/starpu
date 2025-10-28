#!/bin/bash
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

DIRS="tools src tests examples examples/stencil mpi/src mpi/tests mpi/examples"
for d in ${1:-$DIRS}
do
    #echo processing $d
    cd $d
    for ext in c h cl cu doxy
    do
	#echo processing $ext
	for f in $(find -name "*.$ext")
	do
	    #echo processing $f
	    x1=$(grep -c $(basename $f) Makefile.am)
	    if test $ext == "c"
	    then
		x2=$(grep -c $(basename $f .$ext) Makefile.am)
	    else
		x2=0
	    fi
	    if test "$x1" == "0" -a "$x2" == "0"
	    then
		echo $d/$f not in $d/Makefile.am
	    fi
	done
	##grep -rsn "{" $d |grep ".${ext}:" | grep -v "}" > /tmp/braces
	#my_grep "{" /tmp/braces $ext
	#grep -rsn "}" $d |grep ".${ext}:" | grep -v "{" | grep -v "};" > /tmp/braces
	#my_grep "}" /tmp/braces $ext
    done
    cd -
done
