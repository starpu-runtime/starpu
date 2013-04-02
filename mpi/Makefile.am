# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2009-2013  Université de Bordeaux 1
# Copyright (C) 2010, 2011, 2012  Centre National de la Recherche Scientifique
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

SUBDIRS=src tests examples

pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = libstarpumpi.pc starpumpi-1.0.pc starpumpi-1.1.pc

versincludedir = $(includedir)/starpu/$(STARPU_EFFECTIVE_VERSION)
versinclude_HEADERS = 					\
	include/starpu_mpi.h

showcheck:
	for i in $(SUBDIRS) ; do \
		make -C $$i showcheck ; \
	done
