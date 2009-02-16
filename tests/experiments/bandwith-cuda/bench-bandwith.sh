#!/bin/bash

mkdir -p .results

rm -f .results/htod-pin.data
rm -f .results/dtoh-pin.data

echo "H -> D"

for log in `seq 1 13`
do
	size=$((2**$log))
	echo "$size	`./cuda-bandwith -pin -HtoD -size $size -cpu-ld $size -gpu-ld $size -iter 50`" >> .results/htod-pin.data 
done

echo "D -> H"

for log in `seq 1 13`
do
	size=$((2**$log))
	echo "$size	`./cuda-bandwith -pin -size $size -cpu-ld $size -gpu-ld $size -iter 50`" >> .results/dtoh-pin.data 
done

./bench-bandwith.gp

echo "STRIDED H -> D"

for stridelog in `seq 1 13`
do
	stridesize=$((2**$stridelog))
	rm -f .results/htod-pin.$stridesize.data
	echo "	STRIDE $stridesize"
	for log in `seq 1 $stridelog`
	do
		size=$((2**$log))
		echo "$size	`./cuda-bandwith -pin -HtoD -size $size -cpu-ld $stridesize -gpu-ld $stridesize -iter 50`" >> .results/htod-pin.$stridesize.data 
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
		echo "$size	`./cuda-bandwith -pin -size $size -cpu-ld $stridesize -gpu-ld $stridesize -iter 50`" >> .results/dtoh-pin.$stridesize.data 
	done
done
