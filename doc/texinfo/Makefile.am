# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2009, 2011  Université de Bordeaux 1
# Copyright (C) 2010, 2011, 2012, 2013  Centre National de la Recherche Scientifique
#
# Permission is granted to copy, distribute and/or modify this document
# under the terms of the GNU Free Documentation License, Version 1.3
# or any later version published by the Free Software Foundation;
# with no Invariant Sections, no Front-Cover Texts, and no Back-Cover Texts.
#
# See the GNU Free Documentation License in COPYING.GFDL for more details.

info_TEXINFOS = starpu.texi

chapters =	\
	chapters/advanced-examples.texi \
	chapters/api.texi \
	chapters/basic-examples.texi \
	chapters/c-extensions.texi \
	chapters/configuration.texi \
	chapters/fdl-1.3.texi \
	chapters/fft-support.texi \
	chapters/hypervisor_api.texi \
	chapters/installing.texi \
	chapters/introduction.texi \
	chapters/mpi-support.texi \
	chapters/perf-feedback.texi \
	chapters/perf-optimization.texi \
	chapters/scaling-vector-example.texi \
	chapters/sc_hypervisor.texi \
	chapters/sched_ctx.texi \
	chapters/socl.texi \
	chapters/tips-tricks.texi \
	chapters/vector_scal_cpu.texi \
	chapters/vector_scal_c.texi \
	chapters/vector_scal_cuda.texi \
	chapters/vector_scal_opencl_codelet.texi \
	chapters/vector_scal_opencl.texi

starpu_TEXINFOS = 		\
	chapters/version.texi 	\
	$(chapters)

MAINTAINERCLEANFILES = starpu.pdf starpu.html

EXTRA_DIST = starpu.css				\
	tutorial/README				\
	tutorial/Makefile			\
	tutorial/hello_world.c			\
	tutorial/hello_world_plugin.c		\
	tutorial/vector_scal.c			\
	tutorial/vector_scal_cpu.c		\
	tutorial/vector_scal_cuda.cu		\
	tutorial/vector_scal_opencl.c		\
	tutorial/vector_scal_opencl_kernel.cl	\
	tutorial/vector_scal_plugin.c		\
	tutorial/vector_scal_plugin_cuda.cu

starpu_tutorial_dir	=	$(docdir)/tutorial
starpu_tutorial__DATA	=			\
	tutorial/README				\
	tutorial/Makefile			\
	tutorial/hello_world.c			\
	tutorial/hello_world_plugin.c		\
	tutorial/vector_scal.c			\
	tutorial/vector_scal_cpu.c		\
	tutorial/vector_scal_cuda.cu		\
	tutorial/vector_scal_opencl.c		\
	tutorial/vector_scal_opencl_kernel.cl	\
	tutorial/vector_scal_plugin.c		\
	tutorial/vector_scal_plugin_cuda.cu

dist_pdf_DATA = starpu.pdf
dist_html_DATA = starpu.html

AM_MAKEINFOHTMLFLAGS = --css-include=$(top_srcdir)/doc/starpu.css --no-headers --no-split

uninstall-local:
	$(RM) $(DESTDIR)$(infodir)/dir

chapters/version.texi: $(chapters)
	@-for f in $(starpu_TEXINFOS) ; do \
                if test -f $(top_srcdir)/doc/$$f ; then stat --format=%Y $(top_srcdir)/doc/$$f 2>/dev/null ; fi \
        done | sort -r | head -1 > timestamp
	@if test -s timestamp ; then \
		LC_ALL=C date --date=@`cat timestamp` +"%d %B %Y" > timestamp_updated 2>/dev/null;\
		LC_ALL=C date --date=@`cat timestamp` +"%B %Y" > timestamp_updated_month 2>/dev/null;\
	fi
	@if test -s timestamp_updated ; then \
		echo "@set UPDATED " `cat timestamp_updated` > $(top_srcdir)/doc/chapters/version.texi;\
		echo "@set UPDATED-MONTH" `cat timestamp_updated_month` >> $(top_srcdir)/doc/chapters/version.texi;\
	else \
		echo "@set UPDATED unknown_date" > $(top_srcdir)/doc/chapters/version.texi ;\
		echo "@set UPDATED-MONTH unknown_date" >> $(top_srcdir)/doc/chapters/version.texi; \
	fi
	@echo "@set EDITION $(VERSION)" >> $(top_srcdir)/doc/chapters/version.texi
	@echo "@set VERSION $(VERSION)" >> $(top_srcdir)/doc/chapters/version.texi
	@-for f in timestamp timestamp_updated timestamp_updated_month ; do \
		if test -f $$f ; then $(RM) $$f ; fi ;\
	done

#$(top_srcdir)/doc/starpu.texi: vector_scal_c.texi vector_scal_cuda.texi vector_scal_opencl.texi vector_scal_opencl_codelet.texi
#vector_scal_c.texi: $(top_srcdir)/examples/basic_examples/vector_scal.c
#	cat $< | sed 's/{/@{/g' | sed 's/}/@}/g' | sed 's/\t/    /g' > $@
#vector_scal_cuda.texi: $(top_srcdir)/examples/basic_examples/vector_scal_cuda.cu
#	cat $< | sed 's/{/@{/g' | sed 's/}/@}/g' | sed 's/\t/    /g' > $@
#vector_scal_opencl.texi: $(top_srcdir)/examples/basic_examples/vector_scal_opencl.c
#	cat $< | sed 's/{/@{/g' | sed 's/}/@}/g' | sed 's/\t/    /g' > $@
#vector_scal_opencl_codelet.texi: $(top_srcdir)/examples/basic_examples/vector_scal_opencl_codelet.cl
#	cat $< | sed 's/{/@{/g' | sed 's/}/@}/g' | sed 's/\t/    /g' > $@
#
#CLEANFILES= \
#	vector_scal_c.texi vector_scal_cuda.texi vector_scal_opencl.texi vector_scal_opencl_codelet.texi

# Rule to update documentation on web server. Should only be used locally.
PUBLISHHOST	?= sync
update-web: starpu.html
	sed -i 's/gcc\.html#Attribute-Syntax/http:\/\/gcc.gnu.org\/onlinedocs\/gcc\/Attribute-Syntax.html#Attribute-Syntax/' starpu.html
	scp starpu.pdf starpu.html $(PUBLISHHOST):/web/runtime/html/StarPU

showcheck:
	-cat /dev/null
