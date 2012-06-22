# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2009, 2011  Université de Bordeaux 1
# Copyright (C) 2010, 2011, 2012  Centre National de la Recherche Scientifique
#
# Permission is granted to copy, distribute and/or modify this document
# under the terms of the GNU Free Documentation License, Version 1.3
# or any later version published by the Free Software Foundation;
# with no Invariant Sections, no Front-Cover Texts, and no Back-Cover Texts.
#
# See the GNU Free Documentation License in COPYING.GFDL for more details.

info_TEXINFOS = starpu.texi

starpu_TEXINFOS = chapters/advanced-api.texi \
	chapters/configuration.texi \
	chapters/perf-feedback.texi \
	chapters/vector_scal_cpu.texi \
	chapters/advanced-examples.texi \
	chapters/fdl-1.3.texi \
	chapters/perf-optimization.texi \
	chapters/vector_scal_c.texi \
	chapters/basic-api.texi \
	chapters/installing.texi \
	chapters/scaling-vector-example.texi \
	chapters/vector_scal_cuda.texi \
	chapters/basic-examples.texi \
	chapters/introduction.texi \
	chapters/tips-tricks.texi \
	chapters/vector_scal_opencl_codelet.texi \
	chapters/c-extensions.texi \
	chapters/mpi-support.texi \
	chapters/fft-support.texi \
	chapters/using.texi \
	chapters/vector_scal_opencl.texi \
	chapters/socl.texi \
	chapters/version.texi

MAINTAINERCLEANFILES = starpu.pdf

EXTRA_DIST = starpu.pdf \
	starpu.css

AM_MAKEINFOHTMLFLAGS = --css-include=$(top_srcdir)/doc/starpu.css --no-headers --no-split

uninstall-local:
	$(RM) $(DESTDIR)$(infodir)/dir

#TODO: when stat is not available on the machine, insert "unknown date"
chapters/version.texi:
	@for f in $(starpu_TEXINFOS) ; do \
                if test -f $(top_srcdir)/doc/$$f ; then stat --format=%Y $(top_srcdir)/doc/$$f ; fi \
        done | sort -r | head -1 > timestamp
	@LC_ALL=C date --date=@`cat timestamp` +"%d %B %Y" > timestamp_updated
	@LC_ALL=C date --date=@`cat timestamp` +"%B %Y" > timestamp_updated_month
	@echo "@set UPDATED " `cat timestamp_updated` > $(top_srcdir)/doc/chapters/version.texi
	@echo "@set UPDATED-MONTH" `cat timestamp_updated_month` >> $(top_srcdir)/doc/chapters/version.texi
	@echo "@set EDITION $(VERSION)" >> $(top_srcdir)/doc/chapters/version.texi && \
	@echo "@set VERSION $(VERSION)" >> $(top_srcdir)/doc/chapters/version.texi

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
