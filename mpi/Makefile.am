# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2009-2015                                Université de Bordeaux
# Copyright (C) 2011,2012,2016                           Inria
# Copyright (C) 2010-2013,2015,2017                      CNRS
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
SUBDIRS=src tests examples

pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = libstarpumpi.pc starpumpi-1.0.pc starpumpi-1.1.pc starpumpi-1.2.pc starpumpi-1.3.pc

versincludedir = $(includedir)/starpu/$(STARPU_EFFECTIVE_VERSION)
versinclude_HEADERS = 					\
	include/starpu_mpi.h				\
	include/starpu_mpi_lb.h				\
	include/fstarpu_mpi_mod.f90

recheck:
	RET=0 ; \
	for i in $(SUBDIRS) ; do \
		make -C $$i recheck || RET=1 ; \
	done ; \
	exit $$RET

showcheck:
	RET=0 ; \
	for i in $(SUBDIRS) ; do \
		make -C $$i showcheck || RET=1 ; \
	done ; \
	exit $$RET
