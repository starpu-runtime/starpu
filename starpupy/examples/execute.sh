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

exampledir=/home/gonthier/starpu/./starpupy/examples

modpath=/home/gonthier/starpu/src/.libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
pypath=/home/gonthier/starpu/starpupy/src/build:$PYTHONPATH

LOADER=

if test -z "$MPI_LAUNCHER"
then
    MPI_LAUNCHER="mpiexec -np 2"
fi
mpi=""
gdb=""

read_arg()
{
    do_shift=0
    if test "$1" == "--valgrind"
    then
	export PYTHONMALLOC=malloc
	LOADER="valgrind --track-origins=yes "
	do_shift=1
    elif test "$1" == "--gdb"
    then
	gdb="gdb"
	if test "$mpi" == "mpi"
	then
	    LOADER="$MPI_LAUNCHER xterm -sl 10000 -hold -e gdb --args "
	else
	    LOADER="gdb --args "
	fi
	do_shift=1
    elif test "$1" == "--mpirun"
    then
	mpi="mpi"
	if test "$gdb" == "gdb"
	then
	    LOADER="$MPI_LAUNCHER xterm -sl 10000 -hold -e gdb --args "
	else
	    LOADER="$MPI_LAUNCHER "
	fi
	do_shift=1
    fi
}

for x in $*
do
    read_arg $x
    if test $do_shift == 1
    then
	shift
    fi
done
for x in $LOADER_ARGS
do
    read_arg $x
done

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
PYTHONPATH=$pypath LD_LIBRARY_PATH=$modpath $LOADER $pythonscript $*

