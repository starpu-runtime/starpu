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

include $(top_srcdir)/starpu-notests.mk

SUBDIRS =

PYTHON_PY_SRC	=	$(wildcard $(top_srcdir)/starpupy/src/*py)
PYTHON_PY_BUILD	=	$(addprefix $(top_builddir)/starpupy/src/starpu/,$(notdir $(PYTHON_PY_SRC)))

PYTHON_C_SRC	=	$(wildcard $(top_srcdir)/starpupy/src/*c)
PYTHON_C_BUILD	=	$(addprefix $(top_builddir)/starpupy/src/starpu/,$(notdir $(PYTHON_C_SRC)))

$(top_builddir)/starpupy/src/starpu/%.py: $(abs_top_srcdir)/starpupy/src/%.py
	$(MKDIR_P) starpu
	$(V_ln) $(LN_S) $< $@
$(top_builddir)/starpupy/src/starpu/%.c: $(abs_top_srcdir)/starpupy/src/%.c
	@$(MKDIR_P) starpu
	$(V_ln) $(LN_S) $< $@

all: $(PYTHON_PY_BUILD) $(PYTHON_C_BUILD)
	$(PYTHON) setup.py build $(PYTHON_DEBUG)

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
	delay.py		\
	__init__.py	\
	intermedia.py	\
	joblib.py	\
	starpu_task_wrapper.c
