#!/bin/bash

# StarPU --- Runtime system for heterogeneous multicore architectures.
# 
# Copyright (C) 2011  INRIA
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


DIR=$PWD
ROOTDIR=$DIR/../..
TIMINGDIR=$DIR/timings-sched/$1
mkdir -p $TIMINGDIR
BENCH_NAME=cholesky/cholesky_implicit
nsamples=5

filename=$TIMINGDIR/cholesky_no_ctxs


nmaxcpus=12
nmincpus=1
blocks1=40
blocks2=40

size1=20000
size2=10000


for j in `seq $nmincpus 1 $nmaxcpus`
do
    if [ $j -le 3 ]
    then
	export STARPU_NCUDA=$j
    else
	export STARPU_NCPUS=$(($j-3))
    fi
    
    OPTIONS="$2 -with_noctxs -nblocks1 $blocks1 -size1 $size1 -nblocks2 $blocks2 -size2 $size2"

    source evaluate_expression.sh "$BENCH_NAME" "$OPTIONS" "$filename" "$j"

done
    




