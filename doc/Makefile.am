# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2013  Centre National de la Recherche Scientifique
#
# Permission is granted to copy, distribute and/or modify this document
# under the terms of the GNU Free Documentation License, Version 1.3
# or any later version published by the Free Software Foundation;
# with no Invariant Sections, no Front-Cover Texts, and no Back-Cover Texts.
#
# See the GNU Free Documentation License in COPYING.GFDL for more details.

if BUILD_DOC
SUBDIRS = doxygen
endif

EXTRA_DIST =    tutorial/hello_world.c \
		tutorial/hello_world_plugin.c \
		tutorial/hello_world_msvc.c \
		tutorial/Makefile \
		tutorial/README \
		tutorial/vector_scal.c \
		tutorial/vector_scal_cpu.c \
		tutorial/vector_scal_cuda.cu \
		tutorial/vector_scal_opencl.c \
		tutorial/vector_scal_opencl_kernel.cl \
		tutorial/vector_scal_plugin.c \
		tutorial/vector_scal_plugin_cuda.cu

txtdir = ${docdir}/tutorial
txt_DATA = $(EXTRA_DIST)
