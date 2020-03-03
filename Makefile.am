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
ACLOCAL_AMFLAGS=-I m4
CLEANFILES = *.gcno *.gcda *.linkinfo

SUBDIRS = src
SUBDIRS += tools

if BUILD_TESTS
SUBDIRS += tests
endif

SUBDIRS += doc

if USE_MPI
SUBDIRS += mpi
endif

if BUILD_EXAMPLES
SUBDIRS += examples
endif

if BUILD_SOCL
SUBDIRS += socl
endif

if BUILD_GCC_PLUGIN
SUBDIRS += gcc-plugin
endif

if BUILD_STARPUFFT
SUBDIRS += starpufft
endif

if STARPU_BUILD_SC_HYPERVISOR
SUBDIRS += sc_hypervisor
endif

pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = libstarpu.pc starpu-1.0.pc starpu-1.1.pc starpu-1.2.pc

versincludedir = $(includedir)/starpu/$(STARPU_EFFECTIVE_VERSION)
versinclude_HEADERS = 				\
	include/starpu.h			\
	include/starpu_bitmap.h			\
	include/starpu_data_filters.h		\
	include/starpu_data_interfaces.h	\
	include/starpu_worker.h			\
	include/starpu_task.h			\
	include/starpu_task_bundle.h		\
	include/starpu_task_list.h		\
	include/starpu_task_util.h		\
	include/starpu_data.h			\
	include/starpu_perfmodel.h		\
	include/starpu_util.h			\
	include/starpu_fxt.h			\
	include/starpu_cuda.h			\
	include/starpu_opencl.h			\
	include/starpu_openmp.h			\
	include/starpu_sink.h			\
	include/starpu_mic.h			\
	include/starpu_scc.h			\
	include/starpu_expert.h			\
	include/starpu_profiling.h		\
	include/starpu_bound.h			\
	include/starpu_scheduler.h		\
	include/starpu_sched_component.h		\
	include/starpu_sched_ctx.h		\
	include/starpu_sched_ctx_hypervisor.h	\
	include/starpu_top.h			\
	include/starpu_deprecated_api.h         \
	include/starpu_hash.h			\
	include/starpu_rand.h			\
	include/starpu_disk.h			\
	include/starpu_cublas.h			\
	include/starpu_cublas_v2.h		\
	include/starpu_cusparse.h		\
	include/starpu_driver.h			\
	include/starpu_stdlib.h			\
	include/starpu_thread.h			\
	include/starpu_thread_util.h		\
	include/starpu_tree.h			\
	include/starpu_simgrid_wrap.h		\
	include/starpu_mod.f90			\
	include/fstarpu_mod.f90

nodist_versinclude_HEADERS = 			\
	include/starpu_config.h

noinst_HEADERS = \
	include/pthread_win32/pthread.h		\
	include/pthread_win32/semaphore.h

if BUILD_STARPU_TOP
all-local: starpu-top/starpu_top$(EXEEXT)
else
all-local:
endif
if STARPU_DEVEL
	@if grep -r sys/time.h $$( find $(srcdir)/examples $(srcdir)/tests $(srcdir)/src $(srcdir)/mpi/src $(srcdir)/include -name \*.[ch] -a \! -name starpu_util.h -a \! -name timer.h -a \! -name loader.c ) ; \
	then \
		echo "Please do not include sys/time, it is not available on Windows, include starpu_util.h and use starpu_timing_now() instead" ; \
		false ; \
	fi
	@if grep -re '\<ssize_t' $$( find $(srcdir)/examples $(srcdir)/tests $(srcdir)/src $(srcdir)/mpi/src $(srcdir)/include -name \*.[ch] -a \! -name starpu_config.h ) ; \
	then \
		echo "Please do not use ssize_t, it is not available on Windows, use starpu_ssize_t instead"; \
		false ; \
	fi
	@if grep -re '\<getenv\>' $$( find $(srcdir)/src $(srcdir)/mpi/src $(srcdir)/include -name \*.[ch] -a \! -name starpu_util.h -a \! -name utils.c -a \! -name simgrid.h) ; \
	then \
		echo "Please do not use getenv, use starpu_getenv instead, which catches unsafe uses"; \
		false ; \
	fi
endif

if BUILD_STARPU_TOP
starpu-top/starpu_top$(EXEEXT):
	cd starpu-top ; $(QMAKE) ; $(MAKE)
clean-local:
	cd starpu-top ; $(QMAKE) ; $(MAKE) clean ; $(RM) Makefile
	$(RM) starpu-top/starpu_top.1 starpu-top/starpu_top$(EXEEXT)
# TODO: resources
install-exec-local:
	$(MKDIR_P) $(DESTDIR)$(bindir)
	-$(INSTALL_STRIP_PROGRAM) starpu-top/starpu_top$(EXEEXT) $(DESTDIR)$(bindir)
uninstall-local:
	$(RM) $(DESTDIR)$(bindir)/starpu_top$(EXEEXT)
	$(RM) starpu-top/starpu_top$(EXEEXT)
	$(RM) starpu-top/Makefile

if STARPU_HAVE_HELP2MAN
starpu-top/starpu_top.1: starpu-top/starpu_top$(EXEEXT)
	help2man --no-discard-stderr -N --output=$@ starpu-top/starpu_top$(EXEEXT)
dist_man1_MANS =\
	starpu-top/starpu_top.1
endif
endif

if STARPU_HAVE_WINDOWS
txtdir = ${prefix}
else
txtdir = ${docdir}
endif
txt_DATA = AUTHORS COPYING.LGPL README README.dev STARPU-REVISION
EXTRA_DIST = autogen.sh AUTHORS COPYING.LGPL README README.dev STARPU-VERSION STARPU-REVISION build-aux/svn2cl.xsl mic-configure

DISTCLEANFILES = STARPU-REVISION


include starpu-top/extradist

showfailed:
	@RET=0 ; \
	for i in $(SUBDIRS) ; do \
		make -s -C $$i showfailed || RET=1 ; \
	done ; \
	exit $$RET

showcheck:
	RET=0 ; \
	for i in $(SUBDIRS) ; do \
		make -C $$i showcheck || RET=1 ; \
	done ; \
	exit $$RET

showsuite:
	RET=0 ; \
	for i in $(SUBDIRS) ; do \
		make -C $$i showsuite || RET=1 ; \
	done ; \
	exit $$RET

ctags-local:
	cd $(top_srcdir) ; $(CTAGS) -R -I LIST_TYPE
	sed -i $(top_srcdir)/tags -e '/^[^	]* [^	]*	/d' -e '/^[^	]*$$/d' 


# Cyclomatic complexity reports.

# The pmccabe tool, see <http://www.parisc-linux.org/~bame/pmccabe/>.
PMCCABE = pmccabe

VC_URL = "https://gforge.inria.fr/scm/viewvc.php/trunk/%FILENAME%?view=markup&root=starpu"

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
