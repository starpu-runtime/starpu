# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2020-2025   University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

include $(top_srcdir)/make/starpu-notests.mk

SUBDIRS =

PYTHON_PY_SRC	=	$(wildcard $(top_srcdir)/starpupy/src/*py)
PYTHON_PY_BUILD	=	$(addprefix $(top_builddir)/starpupy/src/starpu/,$(notdir $(PYTHON_PY_SRC)))

PYTHON_C_SRC	=	$(wildcard $(top_srcdir)/starpupy/src/*c)
PYTHON_C_BUILD	=	$(addprefix $(top_builddir)/starpupy/src/starpu/,$(notdir $(PYTHON_C_SRC)))

PYTHON_H_SRC	=	$(wildcard $(top_srcdir)/starpupy/src/*h)
PYTHON_H_BUILD	=	$(addprefix $(top_builddir)/starpupy/src/starpu/,$(notdir $(PYTHON_H_SRC)))

$(top_builddir)/starpupy/src/starpu/%.py: $(abs_top_srcdir)/starpupy/src/%.py
	$(MKDIR_P) starpu
	$(V_ln) $(LN_S) $< $@
$(top_builddir)/starpupy/src/starpu/%.c: $(abs_top_srcdir)/starpupy/src/%.c
	@$(MKDIR_P) starpu
	$(V_ln) $(LN_S) $< $@
$(top_builddir)/starpupy/src/starpu/%.h: $(abs_top_srcdir)/starpupy/src/%.h
	@$(MKDIR_P) starpu
	$(V_ln) $(LN_S) $< $@

all: $(PYTHON_PY_BUILD) $(PYTHON_C_BUILD) $(PYTHON_H_BUILD)
	LDFLAGS=$${LDFLAGS/-no-pie/} $(PYTHON) setup.py build $(PYTHON_SETUP_OPTIONS)

# FIXME and enable
dist-hook:
	nm -n build/starpu/starpupy.cpython-*.so | grep -v " [Ua-z] " | grep -ve " _\?_\?_\?f\?starpu" | grep -ve " \(_init\|_fini\|_edata\|__bss_start\|_end\|PyInit_starpupy\|__gcov_\|mangle_path\)" | (! grep .)

check: all

PYTHONPATH=$(DESTDIR)$(prefix)/lib/python$(PYTHON_VERSION)/site-packages

install-exec-local: all
	LDFLAGS=$${LDFLAGS/-no-pie/} $(PYTHON) setup.py clean
	@if test -d $(DESTDIR)$(prefix)/lib/python$(PYTHON_VERSION) ; \
	then	\
		chmod u+w $(DESTDIR)$(prefix)/lib/python$(PYTHON_VERSION) ; \
		chmod u+w $(DESTDIR)$(prefix)/lib/python$(PYTHON_VERSION)/site-packages ; \
	fi
	$(MKDIR_P) $(PYTHONPATH)/starpu
	chmod u+w $(PYTHONPATH)/starpu
	$(PYTHON) setup.py install --prefix '$(prefix)' --root '$(DESTDIR)$(prefix)'
	rm -fr build/bdist*

if STARPU_BUILD_STARPUPY
clean-local:
	rm -rf lib dist build starpupy.egg-info
	$(PYTHON) setup.py clean -a
endif

distclean-local:
	rm -rf build dist lib

uninstall-local: clean-local
	rm -rf $(DESTDIR)$(prefix)/lib/python$(PYTHON_VERSION)/site-packages/starpu*
	rm -rf $(DESTDIR)$(prefix)/lib/python$(PYTHON_VERSION)/site-packages/tmp/starpu*

EXTRA_DIST	=			\
	delay.py			\
	__init__.py			\
	intermedia.py			\
	joblib.py			\
	handle_access.py 		\
	starpu_task_wrapper.c 		\
	starpupy_cloudpickle.h  	\
	starpupy_interface.c 		\
	starpupy_interface.h    	\
	starpupy_buffer_interface.c 	\
	starpupy_buffer_interface.h     \
	starpupy_numpy_filters.c 	\
	starpupy_numpy_filters.h    	\
	starpupy_handle.c 		\
	starpupy_handle.h		\
	starpupy_private.h
