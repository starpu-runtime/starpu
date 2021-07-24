#!/bin/bash
# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2020-2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

dir=$(dirname $0)

cd $dir/../../../
for d in $(find . -name include -not -wholename "*/build/*")
do
    for f in $(find $d -name "*h")
    do
	for i in doxygen-config.cfg.in Makefile.am
	do
	    x=`grep $f $dir/../$i`
	    if test -z "$x"
	    then
		echo $f missing in $i
	    fi
	done
    done
done
