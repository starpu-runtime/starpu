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


infilename=$1
outfilename=$2
withctx=$3
compute_effic=$4
ninstr=$5
best_gflops_withoutctxs=$6

rm -rf $outfilename

while read line
do 
    results=($line)

    gflops1=0
    gflops2=0

    t1=0
    t2=0

    if [ $withctx -eq 1 ]
    then
	gpu=${results[0]}
	gpu1=${results[1]}
	gpu2=${results[2]}
	ncpus1=${results[3]}
	ncpus2=${results[4]}
	gflops1=${results[5]}
	gflops2=${results[6]}
	t1=${results[7]}
	t2=${results[8]}

	maxtime=$(echo "$t1/$t2"|bc -l)
	maxtime=${maxtime/.*}

 	if [ "$maxtime" == "" ]
	then
	    maxtime=$t2
	else
	    maxtime=$t1
	fi

	gflops=$(echo "$ninstr/$maxtime"|bc -l)
	if [ $compute_effic -eq 1 ]
	then
	    gflops_norm=$(echo "$gflops/$best_gflops_withoutctxs"|bc -l)
	    
	    echo "$gpu $gpu1 $gpu2 $ncpus1 $ncpus2 `printf '%2.2f %2.2f' $gflops $gflops_norm`" >> $outfilename$gpu1$gpu2
	else
	    nres=$(echo "$gpu+$gpu1+$gpu2+$ncpus1+$ncpus2"|bc -l)
	    best_gflops_rate=$(echo "$best_gflops_withoutctxs/$nres"|bc -l)

	    gflop_rate=$(echo "$gflops/$nres"|bc -l)
	    gflop_norm_rate=$(echo "$gflop_rate/$best_gflops_rate"|bc -l)
	    
	    echo "$ncpus1 $ncpus2 `printf '%2.2f %2.2f %2.2f' $gflops $gflop_rate $gflop_norm_rate`" >> $outfilename  
	fi
    else

	nres=${results[0]}
	gflops1=${results[1]}
	gflops2=${results[2]}
	t1=${results[3]}
	t2=${results[4]}


	maxtime=$(echo "$t1/$t2"|bc -l)
	maxtime=${maxtime/.*}

 	if [ "$maxtime" == "" ]
	then
	    maxtime=$t2
	else
	    maxtime=$t1
	fi

	gflops=$(echo "$ninstr/$maxtime"|bc -l)

	if [ $compute_effic -eq 1 ]
	then
	    echo "$nres `printf '%2.2f' $gflops`" >> $outfilename
	else
	    gflop_rate=$(echo "$gflops/$nres"|bc -l)
	    echo "$nres `printf '%2.2f %2.2f' $gflops $gflop_rate`" >> $outfilename
	fi
	
    fi


done < $infilename

