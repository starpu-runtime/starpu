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

include $(top_srcdir)/starpu-subdirtests.mk

SUBDIRS=src

if STARPU_BUILD_EXAMPLES
SUBDIRS += examples
endif
if STARPU_BUILD_TESTS
SUBDIRS += tests
endif

pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = libstarpumpi.pc starpumpi-1.0.pc starpumpi-1.1.pc starpumpi-1.2.pc starpumpi-1.3.pc

versincludedir = $(includedir)/starpu/$(STARPU_EFFECTIVE_VERSION)
versinclude_HEADERS = 					\
	include/starpu_mpi.h				\
	include/starpu_mpi_lb.h				\
	include/fstarpu_mpi_mod.f90

if !STARPU_SIMGRID
check-recursive:
	RET=0 ; \
	NJOBS=`printf %s "$(MAKEFLAGS)" | sed -ne 's/.*-j \?\([0-9]\+\).*/\1/p'` ; \
	JOBS="" ; \
	if [ -n "$$NJOBS" ] ; then \
		if [ "$$NJOBS" -ge 4 ] ; then \
			JOBS="-j$$(($$NJOBS / 4))" ; \
		else \
			JOBS="-j1" ; \
		fi ; \
	fi ; \
	for i in $(SUBDIRS) ; do \
		$(MAKE) check -C $$i MAKEFLAGS="$(MAKEFLAGS) $$JOBS" || RET=1; \
	done ; \
	exit $$RET
endif
