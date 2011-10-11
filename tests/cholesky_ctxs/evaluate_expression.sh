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

nsamples=3

BENCH_NAME=$1
OPTIONS=$2
filename=$3
print_options=$4

gflops1_avg=0
gflops2_avg=0

t1_avg=0
t2_avg=0
t_total_avg=0

for s in `seq 1 $nsamples`
do
    echo "$ROOTDIR/examples/$BENCH_NAME $OPTIONS"
    
    val=`$ROOTDIR/examples/$BENCH_NAME $OPTIONS`
    
    echo "$val"
    
    results=($val)
    
    gflops1_avg=$(echo "$gflops1_avg+${results[0]}"|bc -l)
    gflops2_avg=$(echo "$gflops2_avg+${results[1]}"|bc -l)
    t1_avg=$(echo "$t1_avg+${results[2]}"|bc -l)
    t2_avg=$(echo "$t2_avg+${results[3]}"|bc -l)
    t_total_avg=$(echo "$t_total_avg+${results[4]}"|bc -l)
    
done

gflops1_avg=$(echo "$gflops1_avg / $nsamples"|bc -l)
gflops2_avg=$(echo "$gflops2_avg / $nsamples"|bc -l)
t1_avg=$(echo "$t1_avg / $nsamples"|bc -l)
t2_avg=$(echo "$t2_avg / $nsamples"|bc -l)
t_total_avg=$(echo "$t_total_avg / $nsamples"|bc -l)


echo "$print_options `printf '%2.2f %2.2f %2.2f %2.2f %2.2f' $gflops1_avg $gflops2_avg $t1_avg $t2_avg $t_total_avg`"
echo "$print_options `printf '%2.2f %2.2f %2.2f %2.2f %2.2f' $gflops1_avg $gflops2_avg $t1_avg $t2_avg $t_total_avg`" >> $filename
