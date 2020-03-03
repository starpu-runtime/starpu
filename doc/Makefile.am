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

include $(top_srcdir)/starpu-notests.mk

SUBDIRS = doxygen
SUBDIRS += doxygen_dev
DIST_SUBDIRS = doxygen
DIST_SUBDIRS += doxygen_dev

EXTRA_DIST =    tutorial/hello_world.c \
		tutorial/hello_world_msvc.c \
		tutorial/Makefile \
		tutorial/README \
		tutorial/vector_scal.c \
		tutorial/vector_scal_cpu.c \
		tutorial/vector_scal_cuda.cu \
		tutorial/vector_scal_opencl.c \
		tutorial/vector_scal_opencl_kernel.cl

txtdir = ${docdir}/tutorial
txt_DATA = $(EXTRA_DIST)
