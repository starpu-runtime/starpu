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

CFLAGS+= -I../.. -g

all: example_read

example_read: mm_to_bcsr.o example_read.o mmio.o
	$(CC) example_read.o mm_to_bcsr.o mmio.o -o example_read $(LDFLAGS)

clean:
	@rm -f *.o *.d *.gcno *.gcda
	@rm -f example_read
