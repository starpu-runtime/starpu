#
# StarPU
# Copyright (C) Université Bordeaux 1, CNRS 2008-2009 (see AUTHORS file)
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation; either version 2.1 of the License, or (at
# your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
# See the GNU Lesser General Public License in COPYING.LGPL for more details.
#

info_TEXINFOS = starpu.texi

MAINTAINERCLEANFILES = starpu.pdf

EXTRA_DIST = starpu.pdf \
	starpu.css \
	vector_scal_c.texi \
	vector_scal_cpu.texi \
	vector_scal_cuda.texi \
	vector_scal_opencl_codelet.texi \
	vector_scal_opencl.texi

AM_MAKEINFOHTMLFLAGS = --css-include=$(top_srcdir)/doc/starpu.css --no-headers --no-split

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
