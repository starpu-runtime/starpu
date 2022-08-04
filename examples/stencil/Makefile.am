# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2010-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
include $(top_srcdir)/starpu-tests.mk
include $(top_srcdir)/starpu-loader.mk
if STARPU_SIMGRID
LOADER_BIN = $(LAUNCHER)
endif

AM_CFLAGS +=  $(APP_CFLAGS)

AM_CPPFLAGS = -I$(top_srcdir)/include/ -I$(top_srcdir)/examples/ -I$(top_builddir)/include $(STARPU_H_CPPFLAGS)
AM_LDFLAGS = @STARPU_EXPORT_DYNAMIC@
LIBS += $(top_builddir)/src/@LIBSTARPU_LINK@
LIBS += $(STARPU_OPENCL_LDFLAGS) $(STARPU_CUDA_LDFLAGS)
LIBS += $(STARPU_EXPORTED_LIBS)

if STARPU_USE_MPI
LIBS += $(top_builddir)/mpi/src/libstarpumpi-@STARPU_EFFECTIVE_VERSION@.la
AM_CPPFLAGS += -I$(top_srcdir)/mpi/include
LAUNCHER		+= $(STARPU_MPIEXEC)
LAUNCHER_ENV	+=	$(MPI_RUN_ENV)
if STARPU_SIMGRID
NVCCFLAGS		+= --compiler-options -fPIC
endif
endif

CC = $(CC_OR_MPICC)

#####################################
# What to install and what to check #
#####################################

check_PROGRAMS	=	$(STARPU_EXAMPLES)

if !STARPU_SIMGRID
if STARPU_USE_MPI
if STARPU_MPI_CHECK
TESTS		=	$(STARPU_EXAMPLES)
endif
else
TESTS		=	$(STARPU_EXAMPLES)
endif
endif

###################
# stencil example #
###################
STARPU_EXAMPLES =				\
	stencil					\
	implicit_stencil

examplebindir = $(libdir)/starpu/examples/stencil

examplebin_PROGRAMS =				\
	stencil					\
	implicit_stencil

stencil_SOURCES =				\
	life.c					\
	stencil-kernels.c			\
	stencil-tasks.c				\
	stencil-blocks.c			\
	stencil.c

noinst_HEADERS =				\
	stencil.h				\
	implicit-stencil.h			\
	shadow.h

if STARPU_USE_CUDA
stencil_SOURCES +=				\
	life_cuda.cu				\
	shadow.cu
endif

if STARPU_USE_OPENCL
stencil_SOURCES +=				\
	life_opencl.c				\
	shadow_opencl.c
endif

implicit_stencil_SOURCES =			\
	life.c					\
	implicit-stencil-kernels.c		\
	implicit-stencil-tasks.c		\
	implicit-stencil-blocks.c		\
	implicit-stencil.c

if STARPU_USE_CUDA
implicit_stencil_SOURCES +=			\
	life_cuda.cu				\
	shadow.cu
endif

if STARPU_USE_OPENCL
implicit_stencil_SOURCES +=			\
	life_opencl.c				\
	shadow_opencl.c
endif

outs =						\
	0.5.out					\
	0.out					\
	1.out					\
	2.out					\
	3.out					\
	4.out					\
	6.out					\
	mpi.out

EXTRA_DIST = $(outs) results run README

pics: $(outs:.out=.xpm)

CLEANFILES = *.gcno *.gcda *.xpm starpu_idle_microsec.log


.out.out2:
	$(GREP) '^|' $< | tr -d ' ' > $@

.out2.xpm:
	( width=$$(expr $$(head -n 1 < $< | wc -c) - 1) ; \
	height=`wc -l < $<` ; \
	echo "/* XPM */" ; \
	echo "static char * test_xpm[] = {" ; \
	echo "\"$$width $$height 9 1\"," ; \
	echo "\"_	c None\"," ; \
	echo "\"0	c #FF0000\"," ; \
	echo "\"1	c #00FF00\"," ; \
	echo "\"2	c #0000FF\"," ; \
	echo "\"3	c #FFFF00\"," ; \
	echo "\"4	c #FF00FF\"," ; \
	echo "\"5	c #00FFFF\"," ; \
	echo "\"|	c #FFFFFF\"," ; \
	echo "\"*	c #000000\"," ; \
	< $< $(SED) -e 's/^/"/' -e 's/$$/",/' | $(SED) -e '$$s/",$$/"};/' ) > $@

view:
	feh --zoom 800 -F 0.xpm 0.5.xpm 1.xpm 2.xpm 3.xpm 4.xpm 6.xpm mpi.xpm
