#
# StarPU
# Copyright (C) Université Bordeaux 1, CNRS 2008-2009 (see AUTHORS file)
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

OBJS := bcsr_interface.o csr_interface.o blas_filters.o blas_interface.o vector_interface.o bcsr_filters.o csr_filters.o vector_filters.o

all: $(OBJS)

ifeq ($(filter ${MAKECMDGOALS},clean distclean),)
%.d: %.c
	$(CC) $(CFLAGS) $< -MM -o $*.d

-include $(OBJS:.o=.d)
endif

clean:
	@rm -f *.o *.d *.gcno *.gcda
