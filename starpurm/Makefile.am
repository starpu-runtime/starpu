# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2017,2018                                Inria
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
SUBDIRS=src

if BUILD_STARPURM_EXAMPLES
SUBDIRS += examples
endif

pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = starpurm-1.3.pc

versincludedir = $(includedir)/starpu/$(STARPU_EFFECTIVE_VERSION)
versinclude_HEADERS = 					\
	include/starpurm.h

nodist_versinclude_HEADERS = \
	include/starpurm_config.h

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

showsuite:
	RET=0 ; \
	for i in $(SUBDIRS) ; do \
		make -C $$i showsuite || RET=1 ; \
	done ; \
	exit $$RET
