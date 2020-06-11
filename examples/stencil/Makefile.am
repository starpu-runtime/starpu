# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2010-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
include $(top_srcdir)/starpu.mk

AM_CFLAGS = $(HWLOC_CFLAGS) $(STARPU_CUDA_CPPFLAGS) $(STARPU_OPENCL_CPPFLAGS) $(GLOBAL_AM_CFLAGS) -Wno-unused
LIBS = $(top_builddir)/src/@LIBSTARPU_LINK@ $(HWLOC_LIBS) @LIBS@ $(FXT_LIBS)
AM_CPPFLAGS = -I$(top_srcdir)/include/ -I$(top_srcdir)/examples/ -I$(top_builddir)/include
AM_LDFLAGS = @STARPU_EXPORT_DYNAMIC@ $(STARPU_OPENCL_LDFLAGS) $(STARPU_CUDA_LDFLAGS) $(STARPU_COI_LDFLAGS) $(STARPU_SCIF_LDFLAGS)

if STARPU_USE_MPI
LIBS += $(top_builddir)/mpi/src/libstarpumpi-@STARPU_EFFECTIVE_VERSION@.la
AM_CPPFLAGS += -I$(top_srcdir)/mpi/include
if STARPU_SIMGRID
MPI			=	$(abs_top_builddir)/tools/starpu_smpirun -np 4 -platform $(abs_top_srcdir)/tools/perfmodels/cluster.xml -hostfile $(abs_top_srcdir)/tools/perfmodels/hostfile
NVCCFLAGS		+= --compiler-options -fPIC
else
MPI			=	$(MPIEXEC) $(MPIEXEC_ARGS) -np 4
endif
else
MPI			=
endif

CC = $(CC_OR_MPICC)

if STARPU_USE_CUDA

if STARPU_COVERITY
include $(top_srcdir)/starpu-mynvcc.mk
else
NVCCFLAGS += --compiler-options -fno-strict-aliasing  -I$(top_srcdir)/include/ -I$(top_builddir)/include/ -I$(top_srcdir)/examples/  $(HWLOC_CFLAGS)

.cu.cubin:
	$(V_nvcc) $(NVCC) -cubin $< -o $@ $(NVCCFLAGS)

.cu.o:
	$(V_nvcc) $(NVCC) $< -c -o $@ $(NVCCFLAGS)
endif


endif

#####################################
# What to install and what to check #
#####################################

check_PROGRAMS	=	$(STARPU_EXAMPLES)
noinst_PROGRAMS	=

if !STARPU_SIMGRID
if STARPU_USE_MPI
if STARPU_MPI_CHECK
TESTS		=	$(STARPU_EXAMPLES)
endif
else
TESTS		=	$(STARPU_EXAMPLES)
endif
endif

if !STARPU_HAVE_WINDOWS
## test loader program
if !STARPU_CROSS_COMPILING
LOADER			=	loader
loader_CPPFLAGS 	= 	$(AM_CFLAGS) $(AM_CPPFLAGS) -I$(top_builddir)/src/
LOADER_BIN		=	./$(LOADER)
loader_SOURCES		=	../../tests/loader.c
noinst_PROGRAMS		+=	loader
else
LOADER			=
LOADER_BIN		=	$(top_builddir)/examples/stencil/loader-cross.sh
endif

if STARPU_USE_MPI_MASTER_SLAVE
MPI = $(MPIEXEC) $(MPIEXEC_ARGS) -np 4
endif

if STARPU_HAVE_AM111
TESTS_ENVIRONMENT	=	$(MPI_RUN_ARGS) top_builddir="$(abs_top_builddir)" top_srcdir="$(abs_top_srcdir)"
LOG_COMPILER		=	$(MPI) $(LOADER_BIN)
else
TESTS_ENVIRONMENT	=	$(MPI_RUN_ARGS) top_builddir="$(abs_top_builddir)" top_srcdir="$(abs_top_srcdir)" $(MPI) $(LOADER_BIN)
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
