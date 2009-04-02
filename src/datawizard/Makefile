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

.PHONY: interfaces common

OBJS := write_back.o coherency.o data_request.o progress.o copy-driver.o hierarchy.o memalloc.o footprint.o datastats.o

DWOBJDEPS += ../common/hash.o
DWOBJDEPS += ../common/timing.o 
DWOBJDEPS += ../common/htable32.o 
DWOBJDEPS += ../common/mutex.o 
DWOBJDEPS += ../common/rwlock.o 
DWOBJDEPS += progress.o
DWOBJDEPS += write_back.o
DWOBJDEPS += copy-driver.o
DWOBJDEPS += data_request.o
DWOBJDEPS += coherency.o 
DWOBJDEPS += hierarchy.o 
DWOBJDEPS += memalloc.o
DWOBJDEPS += footprint.o
DWOBJDEPS += interfaces/blas_filters.o
DWOBJDEPS += interfaces/csr_filters.o
DWOBJDEPS += interfaces/bcsr_filters.o
DWOBJDEPS += interfaces/vector_filters.o
DWOBJDEPS += interfaces/blas_interface.o
DWOBJDEPS += interfaces/csr_interface.o
DWOBJDEPS += interfaces/bcsr_interface.o
DWOBJDEPS += interfaces/vector_interface.o


all: datawizard.a interfaces $(SPE_TARGET) $(OBJS)

datawizard.so: common interfaces $(SPE_TARGET) $(OBJS) 
	gcc --shared -o datawizard.so $(DWOBJDEPS)

datawizard.a: common interfaces $(SPE_TARGET) $(OBJS)
	$(AR) rcs $@ $(DWOBJDEPS)

common:
	@make -C ../common/

interfaces:
	@make -C interfaces

ifeq ($(filter ${MAKECMDGOALS},clean distclean),)
%.d: %.c
	$(CC) $(CFLAGS) $< -MM -o $*.d

-include $(OBJS:.o=.d)
endif

clean:
	@make -C interfaces clean
	@rm -f *.o *.d *.gcno *.gcda
	@rm -f *.a *.so
