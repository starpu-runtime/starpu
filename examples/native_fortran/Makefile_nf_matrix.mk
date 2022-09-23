# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2016-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
PROG = nf_matrix

STARPU_VERSION=1.3
FSTARPU_MOD = $(shell pkg-config --cflags-only-I starpu-$(STARPU_VERSION)|sed -e 's/^\([^ ]*starpu\/$(STARPU_VERSION)\).*$$/\1/;s/^.* //;s/^-I//')/fstarpu_mod.f90

SRCSF = nf_matrix.f90		\
	nf_codelets.f90

FC = gfortran
CC = gcc

CFLAGS = -g $(shell pkg-config --cflags starpu-$(STARPU_VERSION))
FCFLAGS = -fdefault-real-8 -J. -g
LDLIBS =  $(shell pkg-config --libs starpu-$(STARPU_VERSION))

OBJS = $(SRCSC:%.c=%.o) fstarpu_mod.o $(SRCSF:%.f90=%.o)

.phony: all clean
all: $(PROG)

$(PROG): $(OBJS)
	$(FC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

fstarpu_mod.o: $(FSTARPU_MOD)
	$(FC) $(FCFLAGS) -c -o $@ $<

%.o: %.f90
	$(FC) $(FCFLAGS) -c -o $@ $<

clean:
	rm -fv *.o *.mod $(PROG)

nf_matrix.o: nf_matrix.f90 nf_codelets.o fstarpu_mod.o
nf_codelets.o: fstarpu_mod.o
