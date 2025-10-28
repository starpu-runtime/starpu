#!/bin/bash
# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2020-2025   University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
DIR=$(realpath $(dirname $0))
ROOTDIR=$DIR/../..

TRACEDIR=$ROOTDIR/tests/traces/fxt.traces
mkdir -p $TRACEDIR
if test ! -f $ROOTDIR/tests/traces/fxt
then
    echo "Example not available"
    exit 77
fi

export STARPU_FXT_PREFIX=$TRACEDIR
export STARPU_FXT_TRACE=1
export STARPU_GENERATE_TRACE_OPTIONS="-no-acquire -c -label-deps"
export STARPU_GENERATE_TRACE=1
$STARPU_MS_LAUNCHER $STARPU_LOADER $ROOTDIR/tests/traces/fxt

prof_file=prof_file_${USER}_0
if test -z "$USER"
then
    prof_file=prof_file_0
fi

if test ! -f $STARPU_FXT_PREFIX/$prof_file
then
    echo "FxT file not generated"
    exit 77
fi
echo "Trace FxT generated"
rm -rf $TRACEDIR
exit 0
