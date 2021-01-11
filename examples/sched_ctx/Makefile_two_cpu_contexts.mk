# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2015-2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
PROG = two_cpu_contexts

SRCSC = two_cpu_contexts.c

CC = gcc

CFLAGS = -Wall -g $(shell pkg-config --cflags starpu-1.3)
LDLIBS =  $(shell pkg-config --libs starpu-1.3)

OBJS = $(SRCSC:%.c=%.o)

.phony: all clean
all: $(PROG)

$(PROG): $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -fv *.o $(PROG)
