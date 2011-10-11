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
TIMINGDIR=$DIR/timings-sched/
mkdir -p $TIMINGDIR
BENCH_NAME=cholesky/cholesky_implicit

filename=$TIMINGDIR/$1

gpu=$2
gpu1=$3
gpu2=$4

nmaxcpus=$STARPU_NCPUS
echo $nmaxcpus

nmincpus1=1
nmincpus2=1

if [ $gpu1 -gt 0 ]
then
    nmincpus1=0
fi

if [ $gpu2 -gt 0 ]
then
    nmincpus2=0
fi


blocks1=40
blocks2=40

size1=20000
size2=10000

for j in `seq $nmincpus1 1 $(($nmaxcpus-1))`
do
    if [ $j -gt $(($nmaxcpus-$nmincpus2)) ]
    then
	break
    fi

    ncpus1=$j
    ncpus2=$(($nmaxcpus-$j))    
    
    OPTIONS="-with_ctxs -nblocks1 $blocks1 -size1 $size1 -nblocks2 $blocks2 -size2 $size2 -gpu $gpu -gpu1 $gpu1 -gpu2 $gpu2 -cpu1 $ncpus1 -cpu2 $ncpus2"

    source evaluate_expression.sh "$BENCH_NAME" "$OPTIONS" "$filename" "$gpu $gpu1 $gpu2 $ncpus1 $ncpus2"

done


