# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2009-2025   University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
ACLOCAL_AMFLAGS=-I m4
CLEANFILES = *.gcno *.gcda *.linkinfo

SUBDIRS =

if STARPU_USE_MIN_DGELS
SUBDIRS += min-dgels
endif

SUBDIRS += src

SUBDIRS += tools

if STARPU_BUILD_TESTS
SUBDIRS += tests
endif

SUBDIRS += doc

if STARPU_USE_MPI
SUBDIRS += mpi
endif

if STARPU_BUILD_EXAMPLES
SUBDIRS += examples
endif

if STARPU_BUILD_SOCL
SUBDIRS += socl
endif

if STARPU_BUILD_STARPUFFT
SUBDIRS += starpufft
endif

if STARPU_BUILD_STARPURM
SUBDIRS += starpurm
endif

if STARPU_OPENMP_LLVM
SUBDIRS += starpu_openmp_llvm
endif

if STARPU_BUILD_STARPUPY
if STARPU_USE_CPU
SUBDIRS += starpupy
else
if STARPU_USE_MPI_MASTER_SLAVE
SUBDIRS += starpupy
else
if STARPU_USE_TCPIP_MASTER_SLAVE
SUBDIRS += starpupy
endif
endif
endif
endif

if STARPU_BUILD_SC_HYPERVISOR
SUBDIRS += sc_hypervisor
endif

if STARPU_USE_JULIA
SUBDIRS += julia
endif

if STARPU_BUILD_ECLIPSE_PLUGIN
SUBDIRS += eclipse-plugin
endif

pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = packages/libstarpu.pc packages/starpu-1.0.pc packages/starpu-1.1.pc packages/starpu-1.2.pc packages/starpu-1.3.pc packages/starpu-1.4.pc

versincludedir = $(includedir)/starpu/$(STARPU_EFFECTIVE_VERSION)
versinclude_HEADERS = 				\
	include/starpu.h			\
	include/starpu_helper.h			\
	include/starpu_bitmap.h			\
	include/starpu_data_filters.h		\
	include/starpu_data_interfaces.h	\
	include/starpu_worker.h			\
	include/starpu_task.h			\
	include/starpu_task_dep.h		\
	include/starpu_task_bundle.h		\
	include/starpu_task_list.h		\
	include/starpu_task_util.h		\
	include/starpu_data.h			\
	include/starpu_perfmodel.h		\
	include/starpu_util.h			\
	include/starpu_fxt.h			\
	include/starpu_cuda.h			\
	include/starpu_hip.h			\
	include/starpu_opencl.h			\
	include/starpu_max_fpga.h		\
	include/starpu_openmp.h			\
	include/starpu_sink.h			\
	include/starpu_expert.h			\
	include/starpu_profiling.h		\
	include/starpu_profiling_tool.h		\
	include/starpu_bound.h			\
	include/starpu_scheduler.h		\
	include/schedulers/starpu_heteroprio.h	\
	include/starpu_sched_component.h	\
	include/starpu_sched_ctx.h		\
	include/starpu_sched_ctx_hypervisor.h	\
	include/starpu_deprecated_api.h         \
	include/starpu_hash.h			\
	include/starpu_rand.h			\
	include/starpu_disk.h			\
	include/starpu_cublas.h			\
	include/starpu_cublas_v2.h		\
	include/starpu_cublasLt.h		\
	include/starpu_cusolver.h		\
	include/starpu_cusparse.h		\
	include/starpu_hipblas.h		\
	include/starpu_driver.h			\
	include/starpu_stdlib.h			\
	include/starpu_thread.h			\
	include/starpu_thread_util.h		\
	include/starpu_tree.h			\
	include/starpu_simgrid_wrap.h		\
	include/starpu_mod.f90			\
	include/fstarpu_mod.f90			\
	include/starpu_parallel_worker.h	\
	include/starpu_perf_monitoring.h	\
	include/starpu_perf_steering.h		\
	include/schedulers/starpu_scheduler_toolbox.h

if STARPU_OPENMP_LLVM
versinclude_HEADERS += 				\
	include/openmp/omp.h
endif

nodist_versinclude_HEADERS = 			\
	include/starpu_config.h

noinst_HEADERS = \
	include/pthread_win32/pthread.h		\
	include/pthread_win32/semaphore.h

if STARPU_DEVEL
all-local:
	@if $(GREP) -r sys/time.h $$( find $(srcdir)/examples $(srcdir)/tests $(srcdir)/src $(srcdir)/mpi/src $(srcdir)/include -name \*.[ch] -a \! -name starpu_util.h -a \! -name timer.h -a \! -name loader.c ) ; \
	then \
		echo "Please do not include sys/time, it is not available on Windows, include starpu_util.h and use starpu_timing_now() instead" ; \
		false ; \
	fi
	@if $(GREP) -re '\<ssize_t' $$( find $(srcdir)/examples $(srcdir)/tests $(srcdir)/src $(srcdir)/mpi/src $(srcdir)/include -name \*.[ch] -a \! -name starpu_config.h ) ; \
	then \
		echo "Please do not use ssize_t, it is not available on Windows, use starpu_ssize_t instead"; \
		false ; \
	fi
	@if $(GREP) -re '\<getenv\>' $$( find $(srcdir)/src $(srcdir)/mpi/src $(srcdir)/include -name \*.[ch] -a \! -name starpu_util.h -a \! -name utils.c -a \! -name simgrid.h) ; \
	then \
		echo "Please do not use getenv, use starpu_getenv instead, which catches unsafe uses"; \
		false ; \
	fi
# we count the number of files which include unistd.h
# we count the number of files which properly include unistd.h i.e by first detecting if it's available
# and then we check both numbers are the same ...a
	@UNISTD_ALL_LINES=$(shell $(GREP) -B1 -rs "^#include <unistd.h>" $(srcdir)/src/ $(srcdir)/include/ $(srcdir)/mpi/src $(srcdir)/mpi/include  |$(GREP) -v dolib|$(GREP) -v -e "--" | tr '\012' '@' | $(SED) 's/unistd.h>@/unistd.h>\n/g' | wc -l | tr -d ' ') ;\
	UNISTD_CORRECT_LINES=$(shell $(GREP) -B1 -rs "^#include <unistd.h>" $(srcdir)/src/ $(srcdir)/include/ $(srcdir)/mpi/src $(srcdir)/mpi/include  |$(GREP) -v dolib|$(GREP) -v -e "--" | tr '\012' '@' | $(SED) 's/unistd.h>@/unistd.h>\n/g' | $(GREP) '#ifdef .*HAVE_UNISTD_H.*:#include <unistd.h>' | wc -l | tr -d ' ') ;\
	if test $$UNISTD_ALL_LINES -ne $$UNISTD_CORRECT_LINES ; \
	then \
		echo "Please do not unconditionally include unistd.h, it is not available on Windows, include config.h and test for HAVE_UNISTD_H" ; \
		false ; \
	fi
endif

if STARPU_HAVE_WINDOWS
txtdir = ${prefix}
else
txtdir = ${docdir}
endif
txt_DATA = AUTHORS COPYING.LGPL README.md README.dev STARPU-REVISION
EXTRA_DIST = autogen.sh AUTHORS COPYING.LGPL README.md README.dev STARPU-VERSION STARPU-REVISION

EXTRA_DIST += .gitlab-ci.yml
EXTRA_DIST += ci/gitlab/build.yml
EXTRA_DIST += ci/gitlab/daily.yml
EXTRA_DIST += ci/gitlab/extended_build.yml
EXTRA_DIST += ci/gitlab/releases.yml
EXTRA_DIST += ci/profiles/genere_ci.py
EXTRA_DIST += ci/profiles/tests.py
EXTRA_DIST += ci/scripts/check_header.sh
EXTRA_DIST += ci/scripts/build.sh
EXTRA_DIST += ci/scripts/check.sh
EXTRA_DIST += ci/scripts/chameleon.sh
EXTRA_DIST += ci/scripts/coverity.sh
EXTRA_DIST += ci/scripts/mpi_ulfm.sh
EXTRA_DIST += ci/scripts/mpich_struct.sh
EXTRA_DIST += ci/scripts/mpich.sh
EXTRA_DIST += ci/scripts/run_profile.sh
EXTRA_DIST += ci/scripts/clean_profile.sh
EXTRA_DIST += ci/scripts/simgrid.sh
EXTRA_DIST += ci/scripts/releases/head.org
EXTRA_DIST += ci/scripts/releases/set-release.sh
EXTRA_DIST += ci/scripts/releases/new-release.sh
EXTRA_DIST += ci/scripts/windows/build-windows.sh
EXTRA_DIST += ci/scripts/windows/check-windows.bat

moduledir = ${libdir}/modules
module_DATA = packages/starpu-1.4

DISTCLEANFILES = STARPU-REVISION

include ./make/starpu-subdirtests.mk

ctags-local:
	cd $(top_srcdir) ; $(CTAGS) -R -I LIST_TYPE
	$(SED) -i $(top_srcdir)/tags -e '/^[^	]* [^	]*	/d' -e '/^[^	]*$$/d'

etags:
	cd $(top_srcdir) ; find . -type f -iname "*.[ch]" | $(ETAGS) -


# Cyclomatic complexity reports.

# The pmccabe tool, see <http://www.parisc-linux.org/~bame/pmccabe/>.
PMCCABE = pmccabe

VC_URL = "https://gitlab.inria.fr/starpu/starpu/-/blob/master/%FILENAME%"

# Generate a cyclomatic complexity report.  Note that examples and tests are
# excluded because they're not particularly relevant, and more importantly
# they all have a function called `main', which clobbers the report.
cyclomatic-complexity.html:
	$(PMCCABE)								\
	  `find \( -name examples -o -name tests -o -path ./tools/dev/experimental \) -prune -o -name \*.c` \
	  | sort -nr								\
	  | $(AWK) -f ${top_srcdir}/build-aux/pmccabe2html			\
		   -v lang=html -v name="$(PACKAGE_NAME)"			\
		   -v vcurl=$(VC_URL)						\
		   -v url="$(PACKAGE_URL)"					\
		   -v css=${top_srcdir}/build-aux/pmccabe.css			\
		   -v cut_dir=${top_srcdir}/					\
		   > $@-tmp
	mv $@-tmp $@
