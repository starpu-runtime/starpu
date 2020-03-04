#!/bin/bash
# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2012-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
CFLAGS += $(shell pkg-config --cflags starpu-1.0) -g -O0 -Wall
LDFLAGS += $(shell pkg-config --libs starpu-1.0) -g -O0 -Wall
 
all: bfs
 
bfs : bfs.o bfs_omp_func.o
	g++ bfs.o bfs_omp_func.o $(CFLAGS) $(LDFLAGS) -fopenmp -O3 -o bfs
bfs.o : bfs.cpp
	g++ bfs.cpp $(CFLAGS) -fopenmp -O3 -c -o bfs.o
bfs_omp_func.o : ./bfs_func/bfs_omp_func.cpp
	g++ ./bfs_func/bfs_omp_func.cpp $(CFLAGS) -fopenmp -O3 -c -o bfs_omp_func.o
 
clean:
	rm -f bfs *.o *~
