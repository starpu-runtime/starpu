#!/bin/bash
# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2016-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

INC_DIRS=$(find $dirname/../../../ -name include -type d)
STARPU_H_FILES=$(find $INC_DIRS -name '*.h')
STARPU_CONFIG=$dirname/../../../include/starpu_config.h.in
STARPU_CONFIG_DISPLAY=$(python3 -c "import os.path; print(os.path.relpath('$STARPU_CONFIG', '.'))")
macros1=$(grep 'ifdef' $STARPU_H_FILES|grep STARPU|awk '{print $NF}')
macros2=$(grep defined $STARPU_H_FILES | tr ' (' '\012' | grep STARPU | sed 's/defined//' | tr -d '()!,')
macros=$(echo $macros1 $macros2 | tr ' ' '\012' | sort | uniq)

for m in $macros
do
    count=$(grep -c $m $STARPU_CONFIG)
    if test "$count" -eq "0"
    then
	echo $m missing in $STARPU_CONFIG_DISPLAY
    fi
done


