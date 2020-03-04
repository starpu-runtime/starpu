# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2016-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

# This makefile gives an example on how to build the testcase outside StarPU

PRG	= mm

CC	= mpicc
CFLAGS	= $(shell pkg-config --cflags starpumpi-1.3) -g -Wall
LDFLAGS	= $(shell pkg-config --libs starpumpi-1.3) -lm

.phony: all clean

all: $(PRG)

clean:
	rm -f $(PRG) *.o starpu*.log
