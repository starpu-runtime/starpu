#
# StarPU
# Copyright (C) INRIA 2008-2009 (see AUTHORS file)
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

export PKG_CONFIG_PATH=../../

LDFLAGS+=$$(pkg-config --libs libstarpu)
CFLAGS+=$$(pkg-config --cflags libstarpu)

all: async-tasks-overhead sync-tasks-overhead

async-tasks-overhead: async-tasks-overhead.o
	$(CC) -o async-tasks-overhead async-tasks-overhead.o $(LDFLAGS)

async-tasks-overhead.o: async-tasks-overhead.c 
	$(CC) $(CFLAGS) async-tasks-overhead.c -c -o async-tasks-overhead.o

sync-tasks-overhead: sync-tasks-overhead.o
	$(CC) -o sync-tasks-overhead sync-tasks-overhead.o $(LDFLAGS)

sync-tasks-overhead.o: sync-tasks-overhead.c 
	$(CC) $(CFLAGS) sync-tasks-overhead.c -c -o sync-tasks-overhead.o

check: async-tasks-overhead sync-tasks-overhead
	./starpu_check.sh

clean:
	rm -f *.o
	rm -f sync-tasks-overhead
	rm -f async-tasks-overhead
