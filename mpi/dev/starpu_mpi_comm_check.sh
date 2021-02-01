#!/bin/bash
# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2017-2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
# Script to check MPI communications are done properly
# The application should be launched with STARPU_MPI_COMM=1
# e.g
#    $ export STARPU_MPI_COMM=1
#    $ mpirun --output-filename starpu_mpi.log appli parameters
# and then the script can be launched with the output files
#    $ starpu_mpi_comm_check.sh starpu_mpi.log.*

if test -z "$1"
then
    echo Syntax error: parameter missing
    exit 1
fi

# Get the nodes identifiers
nodes=$(for f in $*
	do
	    grep starpu_mpi $f | grep '\[' | awk '{print $1}'| sed 's/\[\(.*\)\]\[starpu_mpi\]/\1/' | grep "^[[:digit:]]*$"
	done |sort|uniq
     )
echo nodes $nodes

DIR=/tmp

# for each node, extract send and receive communications
for node in $nodes
do
    for f in $*
    do
	grep starpu_mpi $f |grep "\[$node"
    done > $DIR/starpu_mpi_node$node.log
    grep -- "-->" $DIR/starpu_mpi_node$node.log > $DIR/starpu_mpi_node${node}_send.log
    grep -- "<--" $DIR/starpu_mpi_node$node.log > $DIR/starpu_mpi_node${node}_recv.log
done

# count the number of traced lines
#for node in $nodes
#do
#    wc -l $DIR/starpu_mpi_node${node}_recv.log
#    lines=$(grep :42:42 $DIR/starpu_mpi_node${node}_recv.log | wc -l)
#    lines2=$(( lines + lines ))
#    echo $lines2
#    lines3=$(( lines2 + lines ))
#    echo $lines3
#done

# for each pair of nodes, check tags are sent and received in the same order
for src in $nodes
do
    for dst in $nodes
    do
	if test $src != $dst
	then
	    grep ":$dst:42:" $DIR/starpu_mpi_node${src}_send.log| awk -F':' '{print $6}' > $DIR/node${src}_send_to_${dst}.log
	    grep ":$src:42:" $DIR/starpu_mpi_node${dst}_recv.log|awk -F ':' '{print $6}'> $DIR/node${dst}_recv_from_${src}.log
 	    diff --side-by-side  --suppress-common-lines $DIR/node${src}_send_to_${dst}.log $DIR/node${dst}_recv_from_${src}.log  > $DIR/check_$$
	    if test -s $DIR/check_$$
	    then
		echo $src $dst
		less $DIR/check_$$
	    fi
	fi
    done
done

# check each envelope reception is followed by the appropriate data reception
# first line: MPI_Recv of the envelope
# second line: display envelope information
# third line: MPI_Recv of the data
for node in $nodes
do
    echo processing $DIR/starpu_mpi_node${node}_recv.log
    (
	while read line
	do
	    read line2
	    read line3
	    #echo processing
	    tag2=$(echo $line2 | awk -F ':' '{print $6}')
	    tag3=$(echo $line3 | awk -F ':' '{print $6}')
	    if test "$tag2" != "$tag3"
	    then
		echo erreur
		echo $tag2 $tag3
		echo $line
		echo $line2
		echo $line3
	    fi
	done
    ) < $DIR/starpu_mpi_node${node}_recv.log
done

