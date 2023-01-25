#!/bin/sh
# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2019-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

for i in bcsr block coo csr matrix multiformat ndim tensor variable vector void
do
    $MS_LAUNCHER $STARPU_LAUNCH ./tests/datawizard/interfaces/$i/${i}_interface
    ret=$?
    if test "$ret" = "0"
    then
	echo "Interface $i: success"
    else
	echo "Interface $i: failure"
    fi
done
