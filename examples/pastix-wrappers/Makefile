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

STARPU= ../../libstarpu.a

all: starpu-blas-wrapper.a

starpu-blas-wrapper.o: starpu-blas-wrapper.c starpu-blas-wrapper.h
	$(CC) $(CFLAGS) starpu-blas-wrapper.c -c -o starpu-blas-wrapper.o

starpu-blas-wrapper.a: starpu-blas-wrapper.o
	$(AR) rcs starpu-blas-wrapper.a starpu-blas-wrapper.o

clean:
	@rm -f *.a *.o *.d *.gcno *.gcda
