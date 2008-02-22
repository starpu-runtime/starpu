#!/bin/bash

#just in case
make clean
make

echo "GRIDDIMX\tGRIDDIMY\tBLOCKDIMX\tBLOCKDIMY\tTIME\tBW" > perf.log

# we only bench the "good" kernel
for griddimy in 1 2 4 8 16
do
	for griddimx in 1 2 4 8 16 32 
	do
		for blockdimy in 1 2 4 8 16
		do
			for blockdimx in 1 2 4 8 16 32 
			do
				echo "grid ($griddimx,$griddimy) block ($blockdimx, $blockdimy)"
				./bandwith-test 3 $griddimx $griddimy $blockdimx $blockdimy >> perf.log
			done
		done
	done
done
