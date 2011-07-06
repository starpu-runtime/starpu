# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2009-2011  Université de Bordeaux 1
# Copyright (C) 2010, 2011  Centre National de la Recherche Scientifique
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

CFLAGS          +=      $$(pkg-config --cflags libstarpu)
LDFLAGS         +=      $$(pkg-config --libs libstarpu)

%.o: %.cu
	nvcc $(CFLAGS) $< -c $

all: hello_world vector_scal
vector_scal: vector_scal.o vector_scal_cpu.o vector_scal_cuda.o vector_scal_opencl.o

clean:
	rm -f hello_world vector_scal *.o

