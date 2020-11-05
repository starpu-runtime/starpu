# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2020       Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

SUBDIRS =

all:
	$(MKDIR_P) starpu
	for f in $(top_srcdir)/starpupy/src/starpu/*py ; do cp $$f starpu ; done
	for f in $(top_srcdir)/starpupy/src/starpu/*c ; do cp $$f starpu ; done
	$(PYTHON) setup.py build

install:
	@if test -d $(prefix)/lib/python* ; \
	then	\
		chmod u+w $(prefix)/lib/python* ; \
		chmod u+w $(prefix)/lib/python*/site-packages ; \
	fi
	$(PYTHON) setup.py install

clean-local:
	$(PYTHON) setup.py clean
	rm -f starpu/*py starpu/*c

distclean-local:
	rm -rf build

uninstall:
	rm -rf $(prefix)/lib/python*/site-packages/starpu*
	rm -rf $(prefix)/lib/python*/site-packages/tmp/starpu*

EXTRA_DIST	=		\
	starpu/delay.py		\
	starpu/__init__.py	\
	starpu/intermedia.py	\
	starpu/joblib.py	\
	starpu/starpu_task_wrapper.c

