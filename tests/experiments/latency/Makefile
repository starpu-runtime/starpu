#
# StarPU
# Copyright (C) INRIA 2008-2010 (see AUTHORS file)
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

CUDASDKDIR?=/home/gonnet/NVIDIA_CUDA_SDK/
CUDAINSTALLDIR=/usr/local/cuda/

CFLAGS += -I$(CUDAINSTALLDIR)/include -Wall
LDFLAGS += -lcuda -L/usr/local/cuda/lib64/ -lcudart
LDFLAGS += -lpthread

CFLAGS+= -g #-pg
#LDFLAGS+= -pg

all:  cuda-latency

cuda-latency:  cuda-latency.c

clean:
	rm -f cuda-latency
	rm -f *.o
