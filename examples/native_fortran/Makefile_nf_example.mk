# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2015-2025   University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
# Copyright (C) 2015-2015   ONERA
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
PROG = nf_example

STARPU_VERSION=1.3
FSTARPU_MOD = $(shell pkg-config --variable=starpu_includedir starpu-$(STARPU_VERSION))/fstarpu_mod.f90

SRCSF = nf_types.f90		\
	nf_compute.f90		\
	nf_example.f90

FC = gfortran

FCFLAGS = -fdefault-real-8 -J. -g
LDLIBS =  $(shell pkg-config --libs starpu-$(STARPU_VERSION))

OBJS = fstarpu_mod.o $(SRCSF:%.f90=%.o)

.phony: all clean
all: $(PROG)

$(PROG): $(OBJS)
	$(FC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

fstarpu_mod.o: $(FSTARPU_MOD)
	$(FC) $(FCFLAGS) -c -o $@ $<

%.o: %.f90
	$(FC) $(FCFLAGS) -c -o $@ $<

clean:
	rm -fv *.o *.mod $(PROG)

# modfiles generation dependences
nf_compute.o: nf_compute.f90 nf_types.o fstarpu_mod.o
nf_example.o: nf_example.f90 nf_types.o nf_compute.o fstarpu_mod.o
