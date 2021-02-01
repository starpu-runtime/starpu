# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2016-2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
PROG = add_vectors

SRCCXX = add_vectors.cpp

CXX = g++

CXXFLAGS = -g -DPRINT_OUTPUT $(shell pkg-config --cflags starpu-1.3)
LDLIBS =  $(shell pkg-config --libs starpu-1.3)

OBJS = $(SRCCXX:%.cpp=%.o)

.phony: all clean
all: $(PROG)

$(PROG): $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LDLIBS)

clean:
	rm -fv *.o $(PROG)
