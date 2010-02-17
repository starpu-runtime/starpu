#!/bin/bash

# 4G x np = 4 * (k*1K) ^ 2
# A G * np = 4 * k^2 * 1M
# A * 250 * np = k^2
# A = 6
# k = sqrt(1500*np)
# np = 1 => k = 32
# np = 2 => k = 48
# np = 3 => k = 64 
# np = 4 => k = 64

# Problem size
NBLOCKS=16
BLOCKSIZE=1024
SIZE=$(($NBLOCKS*$BLOCKSIZE))

echo "JOB ID ${PBS_JOBID}"

nnodes=$(cat machinefile.${PBS_JOBID}|wc -l)
echo "got $nnodes mpi nodes"

# Calibrate
ncalibrate=0
for i in `seq 1 $ncalibrate`
do
echo "CALIBRATE $i/$ncalibrate"
CALIBRATE=1 SCHED="dmda" PREFETCH=1 mpirun -machinefile machinefile.${PBS_JOBID} -np $nnodes ./mpi_lu/plu_example_float -p 2 -q 2 -nblocks 32 -size $((32*$BLOCKSIZE)) -numa
done

func()
{
ngpus=$1
np=$2
p=$3
q=$4
nblocks=$5

echo "*******************************************"> log
echo "*************** NGPUS $ngpus - np $np - nblocks $nblocks **************">> log
echo "*******************************************">> log
cat log
cat log >> log.all

STARPU_NCPUS=0 STARPU_NCUDA=$ngpus SCHED="dmda" PREFETCH=1 mpirun -machinefile machinefile.${PBS_JOBID} -np $np ./mpi_lu/plu_example_float -p $p -q $q -nblocks $nblocks -size $(($nblocks * $BLOCKSIZE)) -numa > log.out 2> log.err
cat log.out > log
cat log.err >> log
cat log
cat log >> log.all
}

rm -f log.all

#how many time do we repeat each experiment ?
nloops=3

per_node_max_memory=7000

for np in 1 2 4
do
	for nblocks in 16 32 48 64 80
	do
		for ngpus_per_node in 1 2 3 4
		do
			for loop in `seq 1 $nloops`
			do
				# Compute p and q from np
				case $np in
				  1) p=1; q=1;;
				  2) p=2; q=1;;
				  4) p=2; q=2;;
				  *) echo -n "does not support $np nodes yet";;
				esac

				# Does the problem fit into memory ?
				matrix_size=$(($nblocks * $BLOCKSIZE))
				per_node_memory=$(($((4*$matrix_size*$matrix_size/(1024*1024))) / $np))

				echo "NP $np P $p Q $q SIZE $per_node_memory NBLOCKS $nblocks"

				if test $per_node_memory -ge $per_node_max_memory; then
						echo "Problem is too large !"
				else
					func $ngpus_per_node $np $p $q $nblocks
					echo "go !"
				fi
			done
		done
	done
done
