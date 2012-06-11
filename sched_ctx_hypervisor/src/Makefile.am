# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2011, 2012  INRIA
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

AM_CFLAGS = -Wall $(STARPU_CUDA_CPPFLAGS) $(STARPU_OPENCL_CPPFLAGS)
LIBS = $(top_builddir)/src/libstarpu-@STARPU_EFFECTIVE_VERSION@.la

AM_CPPFLAGS = -I$(top_srcdir)/include/ -I$(top_builddir)/include/ -I$(top_builddir)/src/ -I$(top_srcdir)/src/ -I$(top_srcdir)/sched_ctx_hypervisor/include/

AM_LDFLAGS = $(STARPU_CUDA_LDFLAGS) $(STARPU_OPENCL_LDFLAGS)

lib_LTLIBRARIES = libsched_ctx_hypervisor.la

libsched_ctx_hypervisor_la_LIBADD = $(top_builddir)/src/libstarpu-@STARPU_EFFECTIVE_VERSION@.la

libsched_ctx_hypervisor_la_SOURCES = 			\
	sched_ctx_hypervisor.c				\
	sched_ctx_config.c				\
	hypervisor_policies/policy_tools.c		\
	hypervisor_policies/lp_tools.c			\
	hypervisor_policies/idle_policy.c		\
	hypervisor_policies/app_driven_policy.c		\
	hypervisor_policies/gflops_rate_policy.c	\
	hypervisor_policies/lp_policy.c			\
	hypervisor_policies/lp2_policy.c		\
	hypervisor_policies/lp3_policy.c		

noinst_HEADERS = sched_ctx_hypervisor_intern.h		\
	hypervisor_policies/policy_tools.h		\
	hypervisor_policies/lp_tools.h
