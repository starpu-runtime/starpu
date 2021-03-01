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

exampledir=/home/gonthier/starpu/./starpupy/examples

modpath=/home/gonthier/starpu/src/.libs:
pypath=/home/gonthier/starpu/starpupy/src/build:$PYTHONPATH

valgrind=""
gdb=""
if test "$1" == "--valgrind"
then
    valgrind=1
    shift
fi
if test "$1" == "--gdb"
then
    gdb=1
    shift
fi

examplefile=$1
if test -f $examplefile
then
    pythonscript=$examplefile
elif test -f $exampledir/$examplefile
then
    pythonscript=$exampledir/$examplefile
else
    echo "Error. Python script $examplefile not found in current directory or in $exampledir"
    exit 1
fi
shift

set -x
if test "$valgrind" == "1"
then
    PYTHONPATH=$pypath LD_LIBRARY_PATH=$modpath PYTHONMALLOC=malloc valgrind --track-origins=yes  $pythonscript $*
elif test "$gdb" == "1"
then
    PYTHONPATH=$pypath LD_LIBRARY_PATH=$modpath gdb --args  $pythonscript $*
else
    PYTHONPATH=$pypath LD_LIBRARY_PATH=$modpath  $pythonscript $*
fi

