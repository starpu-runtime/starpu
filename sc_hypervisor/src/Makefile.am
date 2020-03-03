# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2011-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

AM_CFLAGS = -Wall $(STARPU_CUDA_CPPFLAGS) $(STARPU_OPENCL_CPPFLAGS) $(FXT_CFLAGS) $(MAGMA_CFLAGS) $(HWLOC_CFLAGS) $(GLOBAL_AM_CFLAGS)
LIBS = $(top_builddir)/src/libstarpu-@STARPU_EFFECTIVE_VERSION@.la
AM_CPPFLAGS = -I$(top_builddir)/include -I$(top_srcdir)/include -I$(top_srcdir)/sc_hypervisor/include/ -I$(top_srcdir)/sc_hypervisor/src
AM_LDFLAGS = $(STARPU_CUDA_LDFLAGS) $(STARPU_OPENCL_LDFLAGS) $(STARPU_COI_LDFLAGS) $(STARPU_SCIF_LDFLAGS)

lib_LTLIBRARIES = libsc_hypervisor.la

libsc_hypervisor_la_LIBADD = $(top_builddir)/src/libstarpu-@STARPU_EFFECTIVE_VERSION@.la

libsc_hypervisor_la_SOURCES = 				\
	sc_hypervisor.c					\
	sc_config.c					\
	policies_utils/policy_tools.c			\
	policies_utils/speed.c				\
	policies_utils/task_pool.c			\
	policies_utils/lp_tools.c			\
	policies_utils/lp_programs.c			\
	policies_utils/dichotomy.c			\
	hypervisor_policies/idle_policy.c		\
	hypervisor_policies/app_driven_policy.c		\
	hypervisor_policies/gflops_rate_policy.c	\
	hypervisor_policies/feft_lp_policy.c		\
	hypervisor_policies/teft_lp_policy.c		\
	hypervisor_policies/ispeed_policy.c		\
	hypervisor_policies/ispeed_lp_policy.c		\
	hypervisor_policies/throughput_lp_policy.c	\
	hypervisor_policies/hard_coded_policy.c		\
	hypervisor_policies/perf_count_policy.c

noinst_HEADERS = sc_hypervisor_intern.h
