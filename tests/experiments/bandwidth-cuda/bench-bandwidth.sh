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


mkdir -p .results

rm -f .results/htod-pin.data
rm -f .results/dtoh-pin.data

echo "H -> D"

for log in `seq 1 13`
do
	size=$((2**$log))
	echo "$size	`./cuda-bandwidth -pin -HtoD -size $size -cpu-ld $size -gpu-ld $size -iter 50`" >> .results/htod-pin.data 
done

echo "D -> H"

for log in `seq 1 13`
do
	size=$((2**$log))
	echo "$size	`./cuda-bandwidth -pin -size $size -cpu-ld $size -gpu-ld $size -iter 50`" >> .results/dtoh-pin.data 
done

./bench-bandwidth.gp

echo "STRIDED H -> D"

for stridelog in `seq 1 13`
do
	stridesize=$((2**$stridelog))
	rm -f .results/htod-pin.$stridesize.data
	echo "	STRIDE $stridesize"
	for log in `seq 1 $stridelog`
	do
		size=$((2**$log))
		echo "$size	`./cuda-bandwidth -pin -HtoD -size $size -cpu-ld $stridesize -gpu-ld $stridesize -iter 50`" >> .results/htod-pin.$stridesize.data 
	done
done

echo "STRIDED D -> H"

for stridelog in `seq 1 13`
do
	stridesize=$((2**$stridelog))
	rm -f .results/dtoh-pin.$stridesize.data
	echo "	STRIDE $stridesize"
	for log in `seq 1 $stridelog`
	do
		size=$((2**$log))
		echo "$size	`./cuda-bandwidth -pin -size $size -cpu-ld $stridesize -gpu-ld $stridesize -iter 50`" >> .results/dtoh-pin.$stridesize.data 
	done
done
