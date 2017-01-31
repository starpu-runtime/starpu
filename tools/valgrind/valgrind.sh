#!/bin/bash
#
# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2016 CNRS
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

EXEC=$(basename $0 .sh)
if test "$EXEC" == "valgrind"
then
    RUN="valgrind"
else
    RUN="valgrind --tool=$EXEC"
fi
SUPPRESSIONS=$(for f in $(dirname $0)/*.suppr ; do echo "--suppressions=$f" ; done)
$RUN -v --num-callers=42 --error-exitcode=42 --track-origins=yes --leak-check=full --show-reachable=yes --errors-for-leak-kinds=all --show-leak-kinds=all --gen-suppressions=all $SUPPRESSIONS $*
