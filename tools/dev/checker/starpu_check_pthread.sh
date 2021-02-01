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
sgrep="grep --exclude-dir=.git --binary-files=without-match"

macros=$(grep define $dirname/../../../include/starpu_thread_util.h | grep "define STARPU" | awk -F'(' '{print $1}' |awk '{print $2}')
for m in $macros
do
    func=$(echo $m | tr '[A-Z]' '[a-z]')
    echo processing $func
    pthread=$(echo $func | sed 's/starpu_//')
    $sgrep -rsl $func
    echo
    $sgrep -rs $pthread | grep -v $func
    read
done
