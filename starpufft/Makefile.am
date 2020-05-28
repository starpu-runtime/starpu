# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2009-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
include $(top_srcdir)/starpu.mk

SUBDIRS=src

if STARPU_BUILD_STARPUFFT_EXAMPLES
if STARPU_BUILD_TESTS
SUBDIRS += tests
endif
endif

versincludedir = $(includedir)/starpu/$(STARPU_EFFECTIVE_VERSION)
versinclude_HEADERS = 				\
	include/starpufft.h

pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = libstarpufft.pc starpufft-1.0.pc starpufft-1.1.pc starpufft-1.2.pc starpufft-1.3.pc


recheck:
	RET=0 ; \
	for i in $(SUBDIRS) ; do \
		make -C $$i recheck || RET=1 ; \
	done ; \
	exit $$RET
