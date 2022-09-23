# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2015-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
# Copyright (C) 2015       ONERA
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
PROG = f90_example

STARPU_VERSION=1.3
FSTARPU_MOD = $(shell pkg-config --variable=starpu_includedir starpu-$(STARPU_VERSION))/fstarpu_mod.f90

SRCSF = mod_types.f90		\
	mod_interface.f90	\
	mod_compute.f90		\
	f90_example.f90
SRCSC = marshalling.c

FC = gfortran
CC = gcc

CFLAGS = -g $(shell pkg-config --cflags starpu-$(STARPU_VERSION))
FCFLAGS = -fdefault-real-8 -J. -g
LDLIBS =  $(shell pkg-config --libs starpu-$(STARPU_VERSION))

OBJS = $(SRCSC:%.c=%.o) starpu_mod.o $(SRCSF:%.f90=%.o)

.phony: all clean
all: $(PROG)

$(PROG): $(OBJS)
	$(FC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

starpu_mod.o: $(STARPU_MOD)
	$(FC) $(FCFLAGS) -c -o $@ $<

%.o: %.f90
	$(FC) $(FCFLAGS) -c -o $@ $<

clean:
	rm -fv *.o *.mod $(PROG)

# modfiles generation dependences
mod_compute.o: mod_compute.f90 mod_types.o mod_interface.o starpu_mod.o
f90_example.o: f90_example.f90 mod_types.o mod_interface.o mod_compute.o
