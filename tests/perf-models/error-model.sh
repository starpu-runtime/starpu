#!/bin/bash

#
# StarPU
# Copyright (C) INRIA 2008-2009 (see AUTHORS file)
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation; either version 2.1 of the License, or (at
# your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
# See the GNU Lesser General Public License in COPYING.LGPL for more details.
#

# for bc
scale=8

#nblockslist="2 4 8 16 16 16 16 16 16 16 16 16 16 16 16"
nblockslist="4 8 16 16 16 16 16 16 16 16 16"
niter=5

#nblockslist="4 4"
#niter=2


rm -f .sampling/*
rm -f log

echo "#iter cpu0 (#tasks0) cpu1 (#tasks1) cpu2 (#tasks2) gpu0 (#tasksgpu0) #totaltask gflops" > gnuplot.data

i=0
for nblocks in $nblockslist
do
	i=$(($i + 1))

	sumcore[$i]='0'
	ntaskcore[$i]='0'
	sumcuda[$i]='0'
	ntaskcuda[$i]='0'
	cpu_ntasktotal[$i]='0'
	gpu_ntasktotal[$i]='0'
	sumgflops[$i]='0'
done

for iter in `seq 1 $niter`
do
cpu_taskcnt=0
gpu_taskcnt=0
i=0
for nblocks in $nblockslist
do
	i=$(($i + 1))

	ntheta=$(($((32 * $nblocks)) + 2))

	echo "ITER $iter -> I $i NBLOCKS $nblocks"

	CALIBRATE=1 SCHED="dm" ./examples/heat/heat -nblocks $nblocks -nthick 34 -ntheta $ntheta -pin 2> output.log.err > output.log
	gflops=`grep "Synthetic GFlops :" output.log.err| sed -e "s/Synthetic GFlops ://"`

	sumgflops[$i]=$(echo "${sumgflops[$i]} + $gflops"|bc -l)

	# retrieve ratio for core 0, 1 and 2
	avgcore0=`grep "MODEL ERROR: CORE 0" starpu.log | sed -e "s/^.*RATIO \(.*\) NTASKS\(.*\)$/\1/"`
	avgcore1=`grep "MODEL ERROR: CORE 1" starpu.log | sed -e "s/^.*RATIO \(.*\) NTASKS\(.*\)$/\1/"`
	avgcore2=`grep "MODEL ERROR: CORE 2" starpu.log | sed -e "s/^.*RATIO \(.*\) NTASKS\(.*\)$/\1/"`
	avgcuda0=`grep "MODEL ERROR: CUDA 0" starpu.log | sed -e "s/^.*RATIO \(.*\) NTASKS\(.*\)$/\1/"`

	ntaskcore0=`grep "MODEL ERROR: CORE 0" starpu.log | sed -e "s/^.*RATIO \(.*\) NTASKS\(.*\)$/\2/"`
	ntaskcore1=`grep "MODEL ERROR: CORE 1" starpu.log | sed -e "s/^.*RATIO \(.*\) NTASKS\(.*\)$/\2/"`
	ntaskcore2=`grep "MODEL ERROR: CORE 2" starpu.log | sed -e "s/^.*RATIO \(.*\) NTASKS\(.*\)$/\2/"`
	ntaskcuda0=`grep "MODEL ERROR: CUDA 0" starpu.log | sed -e "s/^.*RATIO \(.*\) NTASKS\(.*\)$/\2/"`

	sumcore[$i]=$(echo "${sumcore[$i]} + ( $avgcore0 * $ntaskcore0 ) + ( $avgcore1 * $ntaskcore1 ) +  ( $avgcore2 * $ntaskcore2 )"| bc -l)
	ntaskcore[$i]=$(echo "${ntaskcore[$i]} + $ntaskcore0 + $ntaskcore1 + $ntaskcore2"|bc -l)
	sumcuda[$i]=$(echo "${sumcuda[$i]} + ( $avgcuda0 * $ntaskcuda0 )"| bc -l)
	ntaskcuda[$i]=$(echo "${ntaskcuda[$i]} + $ntaskcuda0"|bc -l)

	cpu_taskcnt=$(($cpu_taskcnt + $ntaskcore0 + $ntaskcore1 + $ntaskcore2 ))
	gpu_taskcnt=$(($gpu_taskcnt + $ntaskcuda0))

	cpu_ntasktotal[$i]=$( echo "$cpu_taskcnt + ${cpu_ntasktotal[$i]}" | bc -l) 
	gpu_ntasktotal[$i]=$( echo "$gpu_taskcnt + ${gpu_ntasktotal[$i]}" | bc -l)
done
done

i=0
echo "#ntaskscpu #avg. error cpu #ntaskgpu #avg. error gpu #avg. gflops" > gnuplot.data
for nblocks in $nblockslist
do
	i=$(($i + 1))
	
	avggflops=$(echo "${sumgflops[$i]}/$niter"|bc -l)

	cpu_ntasks=$(echo "${cpu_ntasktotal[$i]}/$niter" | bc -l)
	gpu_ntasks=$(echo "${gpu_ntasktotal[$i]}/$niter" | bc -l)

	avgcpu=$(echo "${sumcore[$i]}/${ntaskcore[$i]}"|bc -l)
	avgcuda=$(echo "${sumcuda[$i]}/${ntaskcuda[$i]}"|bc -l)

	echo "$cpu_ntasks $avgcpu $gpu_ntasks $avgcuda $avggflops" >> gnuplot.data
done

./error-model.gp
