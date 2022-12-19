# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2016-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

noinst_PROGRAMS		=

if STARPU_HAVE_WINDOWS
LOADER_BIN		=	$(LAUNCHER) $(EXTERNAL)
else
LOADER			?=	./loader
loader_CPPFLAGS 	= 	$(AM_CPPFLAGS) -I$(top_builddir)/src/
LOADER_BIN		=	$(LAUNCHER) $(LOADER) $(EXTERNAL)
noinst_PROGRAMS		+=	loader
endif

LSAN_OPTIONS ?= suppressions=$(abs_top_srcdir)/tools/dev/lsan/suppressions
TSAN_OPTIONS ?= suppressions=$(abs_top_srcdir)/tools/dev/tsan/starpu.suppr
export LSAN_OPTIONS
export TSAN_OPTIONS

if STARPU_HAVE_AM111
TESTS_ENVIRONMENT	=	$(LAUNCHER_ENV) top_builddir="$(abs_top_builddir)" top_srcdir="$(abs_top_srcdir)"
LOG_COMPILER	 	=	$(LOADER_BIN)
else
TESTS_ENVIRONMENT 	=	$(LAUNCHER_ENV) top_builddir="$(abs_top_builddir)" top_srcdir="$(abs_top_srcdir)" $(LOADER_BIN)
endif

AM_TESTS_FD_REDIRECT = 9>&2
