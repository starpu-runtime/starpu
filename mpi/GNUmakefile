# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2009-2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

SUBDIRS=

SUBDIRS += examples
SUBDIRS += tests

check: check-recursive

# divide by 4 the number of jobs to run in parallel, since mpirun will start 4
# processes in the tests and examples
#check-recursive:
#	RET=0 ; \
#	NJOBS=`printf %s "$(MAKEFLAGS)" | sed -ne 's/.*-j \?\([0-9]\+\).*/\1/p'` ; \
#	JOBS="" ; \
#	if [ -n "$$NJOBS" ] ; then \
#		if [ "$$NJOBS" -ge 4 ] ; then \
#			JOBS="-j$$(($$NJOBS / 4))" ; \
#		else \
#			JOBS="-j1" ; \
#		fi ; \
#	fi ; \
#	for i in $(SUBDIRS) ; do \
#		$(MAKE) check -C $$i MAKEFLAGS="$(MAKEFLAGS) $$JOBS" || RET=1; \
#	done ; \
#	exit $$RET

%: force
	@$(MAKE) -f Makefile $@

force: ;
