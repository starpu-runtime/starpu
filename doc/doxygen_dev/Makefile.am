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
DOXYGEN = doxygen
PDFLATEX = pdflatex
MAKEINDEX = makeindex

DOX_DIR = $(top_builddir)/doc/doxygen_dev
DOX_CONFIG = $(top_srcdir)/doc/doxygen_dev/doxygen.cfg

DOX_HTML_DIR = html_dev
DOX_LATEX_DIR = latex
DOX_PDF = $(DOX_DIR)/starpu_dev.pdf
DOX_TAG = starpu.tag

txtdir   = $(docdir)/manual

EXTRA_DIST =

if STARPU_BUILD_DOC
if STARPU_BUILD_DOC_PDF
all: $(DOX_HTML_DIR) $(DOX_PDF)
EXTRA_DIST += $(DOX_HTML_DIR) $(DOX_PDF)
txt_DATA = $(DOX_PDF)
else
all: $(DOX_HTML_DIR)
EXTRA_DIST += $(DOX_HTML_DIR)
endif
DOX_HTML_SRCDIR=$(DOX_HTML_DIR)
install-exec-hook:
	$(MKDIR_P) $(DESTDIR)$(docdir)/manual/html_dev
	(cd $(DOX_HTML_SRCDIR) && find . -type f -exec $(INSTALL) -c -m 644 {} $(DESTDIR)$(docdir)/manual/html_dev \;)
uninstall-hook:
	rm -rf $(DESTDIR)$(docdir)/manual/html_dev
else
if STARPU_AVAILABLE_DOC
EXTRA_DIST += $(top_srcdir)/doc/doxygen_dev/html_dev
DOX_HTML_SRCDIR=$(top_srcdir)/doc/doxygen_dev/html_dev
install-exec-hook:
	$(MKDIR_P) $(DESTDIR)$(docdir)/manual/html_dev
	(cd $(DOX_HTML_SRCDIR) && find . -type f -exec $(INSTALL) -c -m 644 {} $(DESTDIR)$(docdir)/manual/html_dev \;)
uninstall-hook:
	rm -rf $(DESTDIR)$(docdir)/manual/html_dev
endif
if STARPU_AVAILABLE_DOC_PDF
EXTRA_DIST += $(top_srcdir)/doc/doxygen_dev/starpu_dev.pdf
txt_DATA = $(top_srcdir)/doc/doxygen_dev/starpu_dev.pdf
endif
endif

chapters =	\
	chapters/000_introduction.doxy \
	chapters/010_core.doxy

images =

if STARPU_BUILD_DOC
config.h: $(top_srcdir)/src/common/config.h.in
	@$(SED) 's/#undef \(.*\)/#define \1 1/' $< > $@
	@$(SED) -i '1s/^/\/\*\* \@file \*\/\n/' $@

chapters/version.sty: $(chapters)
	@for f in $(chapters) ; do \
                if test -f $(top_srcdir)/doc/doxygen_dev/$$f ; then $(PROG_STAT) --format=%Y $(top_srcdir)/doc/doxygen_dev/$$f ; fi \
        done | sort -r | head -1 > timestamp_sty
	@if test -s timestamp_sty ; then \
		LC_ALL=C $(PROG_DATE) --date=@`cat timestamp_sty` +"%d %B %Y" > timestamp_sty_updated ;\
		LC_ALL=C $(PROG_DATE) --date=@`cat timestamp_sty` +"%B %Y" > timestamp_sty_updated_month ;\
	fi
	@if test -s timestamp_sty_updated ; then \
		echo ':newcommand{:STARPUUPDATED}{'`cat timestamp_sty_updated`'}' > $(top_srcdir)/doc/doxygen_dev/chapters/version.sty;\
	else \
		echo ':newcommand{:STARPUUPDATED}{unknown date}' > $(top_srcdir)/doc/doxygen_dev/chapters/version.sty;\
	fi
	@echo ':newcommand{:STARPUVERSION}{$(VERSION)}' >> $(top_srcdir)/doc/doxygen_dev/chapters/version.sty
	@$(SED) -i 's/:/\\/g' $(top_srcdir)/doc/doxygen_dev/chapters/version.sty
	@for f in timestamp_sty timestamp_sty_updated timestamp_sty_updated_month ; do \
		if test -f $$f ; then $(RM) $$f ; fi ;\
	done

chapters/version.html: $(chapters)
	@for f in $(chapters) ; do \
                if test -f $(top_srcdir)/doc/doxygen_dev/$$f ; then $(PROG_STAT) --format=%Y $(top_srcdir)/doc/doxygen_dev/$$f ; fi \
        done | sort -r | head -1 > timestamp_html
	@if test -s timestamp_html ; then \
		LC_ALL=C $(PROG_DATE) --date=@`cat timestamp_html` +"%d %B %Y" > timestamp_html_updated ;\
		LC_ALL=C $(PROG_DATE) --date=@`cat timestamp_html` +"%B %Y" > timestamp_html_updated_month ;\
	fi
	@echo "This manual documents the internal usage of StarPU version $(VERSION)." > $(top_srcdir)/doc/doxygen_dev/chapters/version.html
	@if test -s timestamp_html_updated ; then \
		echo "Its contents was last updated on "`cat timestamp_html_updated`"." >> $(top_srcdir)/doc/doxygen_dev/chapters/version.html;\
	else \
		echo "Its contents was last updated on <em>unknown_date</em>." >> $(top_srcdir)/doc/doxygen_dev/chapters/version.html;\
	fi
	@for f in timestamp_html timestamp_html_updated timestamp_html_updated_month ; do \
		if test -f $$f ; then $(RM) $$f ; fi ;\
	done


dox_inputs = $(DOX_CONFIG) 				\
	$(chapters) 					\
	config.h					\
	chapters/version.sty				\
	chapters/version.html				\
	$(top_srcdir)/src/datawizard/data_request.h	\
	$(top_srcdir)/src/datawizard/coherency.h	\
	$(top_srcdir)/src/datawizard/sort_data_handles.h	\
	$(top_srcdir)/src/datawizard/memalloc.h	\
	$(top_srcdir)/src/datawizard/copy_driver.h	\
	$(top_srcdir)/src/datawizard/filters.h	\
	$(top_srcdir)/src/datawizard/datastats.h	\
	$(top_srcdir)/src/datawizard/write_back.h	\
	$(top_srcdir)/src/datawizard/interfaces/data_interface.h	\
	$(top_srcdir)/src/datawizard/memory_manager.h	\
	$(top_srcdir)/src/datawizard/node_ops.h	\
	$(top_srcdir)/src/datawizard/memstats.h	\
	$(top_srcdir)/src/datawizard/datawizard.h	\
	$(top_srcdir)/src/datawizard/memory_nodes.h	\
	$(top_srcdir)/src/datawizard/footprint.h	\
	$(top_srcdir)/src/datawizard/malloc.h	\
	$(top_srcdir)/src/drivers/cpu/driver_cpu.h	\
	$(top_srcdir)/src/drivers/cuda/driver_cuda.h	\
	$(top_srcdir)/src/drivers/opencl/driver_opencl_utils.h	\
	$(top_srcdir)/src/drivers/opencl/driver_opencl.h	\
	$(top_srcdir)/src/drivers/disk/driver_disk.h	\
	$(top_srcdir)/src/drivers/mpi/driver_mpi_common.h	\
	$(top_srcdir)/src/drivers/mpi/driver_mpi_sink.h	\
	$(top_srcdir)/src/drivers/mpi/driver_mpi_source.h	\
	$(top_srcdir)/src/drivers/mp_common/sink_common.h	\
	$(top_srcdir)/src/drivers/mp_common/mp_common.h	\
	$(top_srcdir)/src/drivers/mp_common/source_common.h	\
	$(top_srcdir)/src/drivers/driver_common/driver_common.h	\
	$(top_srcdir)/src/drivers/mic/driver_mic_sink.h	\
	$(top_srcdir)/src/drivers/mic/driver_mic_source.h	\
	$(top_srcdir)/src/drivers/mic/driver_mic_common.h	\
	$(top_srcdir)/src/profiling/profiling.h	\
	$(top_srcdir)/src/profiling/bound.h	\
	$(top_srcdir)/src/util/starpu_data_cpy.h	\
	$(top_srcdir)/src/util/openmp_runtime_support.h	\
	$(top_srcdir)/src/util/starpu_clusters_create.h	\
	$(top_srcdir)/src/util/starpu_task_insert_utils.h	\
	$(top_srcdir)/src/common/graph.h	\
	$(top_srcdir)/src/common/fxt.h	\
	$(top_srcdir)/src/common/starpu_spinlock.h	\
	$(top_srcdir)/src/common/rbtree_i.h	\
	$(top_srcdir)/src/common/rbtree.h	\
	$(top_srcdir)/src/common/timing.h	\
	$(top_srcdir)/src/common/rwlock.h	\
	$(top_srcdir)/src/common/barrier.h	\
	$(top_srcdir)/src/common/prio_list.h	\
	$(top_srcdir)/src/common/barrier_counter.h	\
	$(top_srcdir)/src/common/uthash.h	\
	$(top_srcdir)/src/common/utils.h	\
	$(top_srcdir)/src/common/thread.h	\
	$(top_srcdir)/src/common/list.h	\
	$(top_srcdir)/src/debug/starpu_debug_helpers.h	\
	$(top_srcdir)/src/debug/traces/starpu_fxt.h	\
	$(top_srcdir)/src/starpu_parameters.h	\
	$(top_srcdir)/src/sched_policies/fifo_queues.h	\
	$(top_srcdir)/src/sched_policies/helper_mct.h	\
	$(top_srcdir)/src/sched_policies/sched_component.h	\
	$(top_srcdir)/src/sched_policies/prio_deque.h	\
	$(top_srcdir)/src/core/jobs.h	\
	$(top_srcdir)/src/core/disk_ops/unistd/disk_unistd_global.h	\
	$(top_srcdir)/src/core/dependencies/tags.h	\
	$(top_srcdir)/src/core/dependencies/data_concurrency.h	\
	$(top_srcdir)/src/core/dependencies/implicit_data_deps.h	\
	$(top_srcdir)/src/core/dependencies/cg.h	\
	$(top_srcdir)/src/core/idle_hook.h	\
	$(top_srcdir)/src/core/sched_ctx_list.h	\
	$(top_srcdir)/src/core/perfmodel/multiple_regression.h	\
	$(top_srcdir)/src/core/perfmodel/perfmodel.h	\
	$(top_srcdir)/src/core/perfmodel/regression.h	\
	$(top_srcdir)/src/core/debug.h	\
	$(top_srcdir)/src/core/sched_ctx.h	\
	$(top_srcdir)/src/core/simgrid.h	\
	$(top_srcdir)/src/core/task_bundle.h	\
	$(top_srcdir)/src/core/topology.h	\
	$(top_srcdir)/src/core/combined_workers.h	\
	$(top_srcdir)/src/core/detect_combined_workers.h	\
	$(top_srcdir)/src/core/task.h	\
	$(top_srcdir)/src/core/disk.h	\
	$(top_srcdir)/src/core/sched_policy.h	\
	$(top_srcdir)/src/core/errorcheck.h	\
	$(top_srcdir)/src/core/progress_hook.h	\
	$(top_srcdir)/src/core/drivers.h	\
	$(top_srcdir)/src/core/workers.h

$(DOX_HTML_DIR): $(DOX_TAG)
	@$(MKDIR_P) $(DOX_HTML_DIR)

$(DOX_TAG): $(dox_inputs)
	@rm -fr $(DOX_HTML_DIR) $(DOX_LATEX_DIR)
	@$(DOXYGEN) $(DOX_CONFIG)
	@$(SED) -i 's/ModuleDocumentation <\/li>/<a class="el" href="modules.html">Modules<\/a>/' html_dev/index.html
	@$(SED) -i 's/FileDocumentation <\/li>/<a class="el" href="files.html">Files<\/a>/' html_dev/index.html
        # comment for the line below: what we really want to do is to remove the line, but dy doing so, it avoids opening the interactive menu when browsing files
	@if test -f html_dev/navtree.js ; then $(SED) -i 's/\[ "Files", "Files.html", null \]/\[ "", "Files.html", null \]/' html_dev/navtree.js ; fi
	@$(SED) -i 's/.*"Files.html".*//' html_dev/pages.html
	@if test -f latex/main.tex ; then mv latex/main.tex latex/index.tex ; fi
	@$(SED) -i '/\\begin{titlepage}/,$$d' $(DOX_LATEX_DIR)/refman.tex
	@cat $(top_srcdir)/doc/doxygen_dev/refman.tex >> $(DOX_LATEX_DIR)/refman.tex

$(DOX_PDF): $(DOX_TAG) refman.tex
	@cp $(top_srcdir)/doc/doxygen_dev/chapters/version.sty $(DOX_LATEX_DIR)
	@echo $(PDFLATEX) $(DOX_LATEX_DIR)/refman.tex
	@cd $(DOX_LATEX_DIR) ;\
	rm -f *.aux *.toc *.idx *.ind *.ilg *.log *.out ;\
	$(SED) -i s'/\\item Module\\-Documentation/\\item \\hyperlink{ModuleDocumentation}{Module Documentation}/' index.tex ;\
	$(SED) -i s'/\\item File\\-Documentation/\\item \\hyperlink{FileDocumentation}{File Documentation}/' index.tex ;\
	max_print_line=1000000 $(PDFLATEX) -interaction batchmode refman.tex ;\
	! < refman.log grep -v group__ | grep -v _amgrp | grep -v deprecated__ | grep "multiply defined" || exit 1 ;\
	$(MAKEINDEX) refman.idx ;\
	max_print_line=1000000 $(PDFLATEX) -interaction batchmode refman.tex ;\
	done=0; repeat=5 ;\
	while test $$done = 0 -a $$repeat -gt 0; do \
           if $(EGREP) 'Rerun (LaTeX|to get cross-references right)' refman.log > /dev/null 2>&1; then \
	       max_print_line=1000000 $(PDFLATEX) -interaction batchmode refman.tex; \
	       repeat=`expr $$repeat - 1`; \
	   else \
	       done=1; \
	   fi; \
	done
	mv $(DOX_LATEX_DIR)/refman.pdf $(DOX_PDF)

CLEANFILES = $(DOX_TAG) config.h \
    -r \
    $(DOX_HTML_DIR) \
    $(DOX_LATEX_DIR) \
    $(DOX_PDF)

endif

EXTRA_DIST += doxygen.cfg refman.tex \
	      $(chapters) $(images)

# Rule to update documentation on web server. Should only be called from benchmarks dalton directory
PUBLISHDIR	?= /home/benchmarks/softs/starpu/starpu-scripts/mirror/files/doc
update-web: $(DOX_PDF)
	cp -pr starpu_dev.pdf html_dev $(PUBLISHDIR)

