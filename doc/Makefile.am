# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2009-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
SUBDIRS	+= doxygen_web_introduction
SUBDIRS	+= doxygen_web_installation
SUBDIRS	+= doxygen_web_basics
SUBDIRS	+= doxygen_web_performances
SUBDIRS	+= doxygen_web_faq
SUBDIRS	+= doxygen_web_languages
SUBDIRS	+= doxygen_web_extensions
DIST_SUBDIRS = doxygen
DIST_SUBDIRS += doxygen_dev
DIST_SUBDIRS += doxygen_web_introduction
DIST_SUBDIRS += doxygen_web_installation
DIST_SUBDIRS += doxygen_web_basics
DIST_SUBDIRS += doxygen_web_performances
DIST_SUBDIRS += doxygen_web_faq
DIST_SUBDIRS += doxygen_web_languages
DIST_SUBDIRS += doxygen_web_extensions

EXTRA_DIST =	\
	tutorial/hello_world.c \
	tutorial/hello_world_msvc.c \
	tutorial/Makefile \
	tutorial/README \
	tutorial/vector_scal.c \
	tutorial/vector_scal_cpu.c \
	tutorial/vector_scal_cuda.cu \
	tutorial/vector_scal_opencl.c \
	tutorial/vector_scal_opencl_kernel.cl	\
	tutorial/vector_scal_task_insert.c \
	title.tex \
	sectionNumbering.py \
	extractHeadline.sh

txtdir = ${docdir}/tutorial
txt_DATA = $(EXTRA_DIST)
