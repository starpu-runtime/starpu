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

DOX_DIR = $(top_builddir)/doc/doxygen
DOX_CONFIG = $(top_srcdir)/doc/doxygen/doxygen.cfg

DOX_HTML_DIR = html
DOX_LATEX_DIR = latex
DOX_PDF = $(DOX_DIR)/starpu.pdf
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
	$(MKDIR_P) $(DESTDIR)$(docdir)/manual/html
	(cd $(DOX_HTML_SRCDIR) && find . -type f -exec $(INSTALL) -c -m 644 {} $(DESTDIR)$(docdir)/manual/html \;)
uninstall-hook:
	rm -rf $(DESTDIR)$(docdir)/manual/html
else
if STARPU_AVAILABLE_DOC
EXTRA_DIST += $(top_srcdir)/doc/doxygen/html
DOX_HTML_SRCDIR=$(top_srcdir)/doc/doxygen/html
install-exec-hook:
	$(MKDIR_P) $(DESTDIR)$(docdir)/manual/html
	(cd $(DOX_HTML_SRCDIR) && find . -type f -exec $(INSTALL) -c -m 644 {} $(DESTDIR)$(docdir)/manual/html \;)
uninstall-hook:
	rm -rf $(DESTDIR)$(docdir)/manual/html
endif
if STARPU_AVAILABLE_DOC_PDF
EXTRA_DIST += $(top_srcdir)/doc/doxygen/starpu.pdf
txt_DATA = $(top_srcdir)/doc/doxygen/starpu.pdf
endif
endif

chapters =	\
	chapters/000_introduction.doxy		\
	chapters/101_building.doxy		\
	chapters/110_basic_examples.doxy		\
	chapters/201_advanced_examples.doxy		\
	chapters/210_check_list_performance.doxy		\
	chapters/301_tasks.doxy		\
	chapters/310_data_management.doxy		\
	chapters/320_scheduling.doxy		\
	chapters/330_scheduling_contexts.doxy		\
	chapters/340_scheduling_context_hypervisor.doxy		\
	chapters/350_scheduling_policy_definition.doxy		\
	chapters/360_debugging_tools.doxy		\
	chapters/370_online_performance_tools.doxy		\
	chapters/380_offline_performance_tools.doxy		\
	chapters/390_faq.doxy		\
	chapters/401_out_of_core.doxy		\
	chapters/410_mpi_support.doxy		\
	chapters/420_fft_support.doxy		\
	chapters/430_mic_support.doxy		\
	chapters/450_native_fortran_support.doxy		\
	chapters/460_socl_opencl_extensions.doxy		\
	chapters/470_simgrid.doxy		\
	chapters/480_openmp_runtime_support.doxy		\
	chapters/490_clustering_a_machine.doxy		\
	chapters/495_interoperability.doxy		\
	chapters/501_environment_variables.doxy		\
	chapters/510_configure_options.doxy		\
	chapters/520_files.doxy		\
	chapters/601_scaling_vector_example.doxy		\
	chapters/610_fdl_1_3.doxy		\
	chapters/code/forkmode.c \
	chapters/code/multiformat.c \
	chapters/code/complex.c \
	chapters/code/simgrid.c \
	chapters/code/vector_scal_c.c \
	chapters/code/vector_scal_cpu.c \
	chapters/code/vector_scal_cuda.c \
	chapters/code/vector_scal_opencl.c \
	chapters/code/vector_scal_opencl_codelet.cl \
	chapters/code/disk_copy.c \
	chapters/code/disk_compute.c \
	chapters/code/nf_initexit.f90 \
	chapters/api/fft_support.doxy \
	chapters/api/versioning.doxy \
	chapters/api/threads.doxy

images = 	\
	chapters/images/data_trace.eps \
	chapters/images/data_trace.pdf \
	chapters/images/data_trace.png \
	chapters/images/distrib_data.eps \
	chapters/images/distrib_data.pdf \
	chapters/images/distrib_data.png \
	chapters/images/distrib_data_histo.eps \
	chapters/images/distrib_data_histo.pdf \
	chapters/images/distrib_data_histo.png \
	chapters/images/paje_draw_histogram.eps \
	chapters/images/paje_draw_histogram.pdf \
	chapters/images/paje_draw_histogram.png \
	chapters/images/parallel_worker2.eps \
	chapters/images/parallel_worker2.pdf \
	chapters/images/parallel_worker2.png \
	chapters/images/runtime-par.eps \
	chapters/images/runtime-par.pdf \
	chapters/images/runtime-par.png \
	chapters/images/starpu_non_linear_memset_regression_based.eps \
	chapters/images/starpu_non_linear_memset_regression_based.pdf \
	chapters/images/starpu_non_linear_memset_regression_based.png \
	chapters/images/starpu_non_linear_memset_regression_based_2.eps \
	chapters/images/starpu_non_linear_memset_regression_based_2.pdf \
	chapters/images/starpu_non_linear_memset_regression_based_2.png \
	chapters/images/starpu_starpu_slu_lu_model_11.eps \
	chapters/images/starpu_starpu_slu_lu_model_11.pdf \
	chapters/images/starpu_starpu_slu_lu_model_11.png \
	chapters/images/starpu_chol_model_11_type.eps \
	chapters/images/starpu_chol_model_11_type.pdf \
	chapters/images/starpu_chol_model_11_type.png \
	chapters/images/tasks_size_overhead.eps \
	chapters/images/tasks_size_overhead.pdf \
	chapters/images/tasks_size_overhead.png \
	chapters/images/temanejo.png

if STARPU_BUILD_DOC
starpu_config.h: $(top_srcdir)/include/starpu_config.h.in
	@$(SED) 's/#undef \(.*\)/#define \1 1/' $< > $@

chapters/version.sty: $(chapters)
	@for f in $(chapters) ; do \
                if test -f $(top_srcdir)/doc/doxygen/$$f ; then $(PROG_STAT) --format=%Y $(top_srcdir)/doc/doxygen/$$f ; fi \
        done | sort -r | head -1 > timestamp_sty
	@if test -s timestamp_sty ; then \
		LC_ALL=C $(PROG_DATE) --date=@`cat timestamp_sty` +"%F" > timestamp_sty_updated ;\
		LC_ALL=C $(PROG_DATE) --date=@`cat timestamp_sty` +"%B %Y" > timestamp_sty_updated_month ;\
	fi
	@if test -s timestamp_sty_updated ; then \
		echo ':newcommand{:STARPUUPDATED}{'`cat timestamp_sty_updated`'}' > $(top_srcdir)/doc/doxygen/chapters/version.sty;\
	else \
		echo ':newcommand{:STARPUUPDATED}{unknown date}' > $(top_srcdir)/doc/doxygen/chapters/version.sty;\
	fi
	@echo ':newcommand{:STARPUVERSION}{$(VERSION)}' >> $(top_srcdir)/doc/doxygen/chapters/version.sty
	@$(SED) -i 's/:/\\/g' $(top_srcdir)/doc/doxygen/chapters/version.sty
	@for f in timestamp_sty timestamp_sty_updated timestamp_sty_updated_month ; do \
		if test -f $$f ; then $(RM) $$f ; fi ;\
	done

chapters/version.html: $(chapters)
	@for f in $(chapters) ; do \
                if test -f $(top_srcdir)/doc/doxygen/$$f ; then $(PROG_STAT) --format=%Y $(top_srcdir)/doc/doxygen/$$f ; fi \
        done | sort -r | head -1 > timestamp_html
	@if test -s timestamp_html ; then \
		LC_ALL=C $(PROG_DATE) --date=@`cat timestamp_html` +"%F" > timestamp_html_updated ;\
		LC_ALL=C $(PROG_DATE) --date=@`cat timestamp_html` +"%B %Y" > timestamp_html_updated_month ;\
	fi
	@echo "This manual documents the usage of StarPU version $(VERSION)." > $(top_srcdir)/doc/doxygen/chapters/version.html
	@if test -s timestamp_html_updated ; then \
		echo "Its contents was last updated on "`cat timestamp_html_updated`"." >> $(top_srcdir)/doc/doxygen/chapters/version.html;\
	else \
		echo "Its contents was last updated on <em>unknown_date</em>." >> $(top_srcdir)/doc/doxygen/chapters/version.html;\
	fi
	@for f in timestamp_html timestamp_html_updated timestamp_html_updated_month ; do \
		if test -f $$f ; then $(RM) $$f ; fi ;\
	done


dox_inputs = $(DOX_CONFIG) 				\
	$(chapters) 					\
	starpu_config.h					\
	chapters/version.sty				\
	chapters/version.html				\
	$(top_srcdir)/include/starpu.h			\
	$(top_srcdir)/include/starpu_bitmap.h		\
	$(top_srcdir)/include/starpu_bound.h		\
	$(top_srcdir)/include/starpu_clusters.h		\
	$(top_srcdir)/include/starpu_cublas.h		\
	$(top_srcdir)/include/starpu_cublas_v2.h	\
	$(top_srcdir)/include/starpu_cusparse.h		\
	$(top_srcdir)/include/starpu_cuda.h		\
	$(top_srcdir)/include/starpu_data_filters.h	\
	$(top_srcdir)/include/starpu_data.h		\
	$(top_srcdir)/include/starpu_data_interfaces.h	\
	$(top_srcdir)/include/starpu_deprecated_api.h	\
	$(top_srcdir)/include/starpu_disk.h		\
	$(top_srcdir)/include/starpu_driver.h		\
	$(top_srcdir)/include/starpu_expert.h		\
	$(top_srcdir)/include/starpu_fxt.h		\
	$(top_srcdir)/include/starpu_hash.h		\
	$(top_srcdir)/include/starpu_mic.h		\
	$(top_srcdir)/include/starpu_mod.f90		\
	$(top_srcdir)/include/starpu_opencl.h		\
	$(top_srcdir)/include/starpu_openmp.h		\
	$(top_srcdir)/include/starpu_perfmodel.h	\
	$(top_srcdir)/include/starpu_profiling.h	\
	$(top_srcdir)/include/starpu_rand.h		\
	$(top_srcdir)/include/starpu_sched_component.h	\
	$(top_srcdir)/include/starpu_sched_ctx.h	\
	$(top_srcdir)/include/starpu_sched_ctx_hypervisor.h	\
	$(top_srcdir)/include/starpu_scheduler.h	\
	$(top_srcdir)/include/starpu_simgrid_wrap.h	\
	$(top_srcdir)/include/starpu_sink.h		\
	$(top_srcdir)/include/starpu_stdlib.h		\
	$(top_srcdir)/include/starpu_task_bundle.h	\
	$(top_srcdir)/include/starpu_task_dep.h		\
	$(top_srcdir)/include/starpu_task.h		\
	$(top_srcdir)/include/starpu_task_list.h	\
	$(top_srcdir)/include/starpu_task_util.h	\
	$(top_srcdir)/include/starpu_thread.h		\
	$(top_srcdir)/include/starpu_thread_util.h	\
	$(top_srcdir)/include/starpu_tree.h		\
	$(top_srcdir)/include/starpu_util.h		\
	$(top_srcdir)/include/starpu_worker.h		\
	$(top_srcdir)/include/fstarpu_mod.f90		\
	$(top_srcdir)/mpi/include/starpu_mpi.h 		\
	$(top_srcdir)/mpi/include/starpu_mpi_lb.h	\
	$(top_srcdir)/mpi/include/fstarpu_mpi_mod.f90		\
	$(top_srcdir)/sc_hypervisor/include/sc_hypervisor.h 		\
	$(top_srcdir)/sc_hypervisor/include/sc_hypervisor_config.h 	\
	$(top_srcdir)/sc_hypervisor/include/sc_hypervisor_lp.h		\
	$(top_srcdir)/sc_hypervisor/include/sc_hypervisor_monitoring.h	\
	$(top_srcdir)/sc_hypervisor/include/sc_hypervisor_policy.h	\
	$(top_srcdir)/starpurm/include/starpurm.h

doxy:
	@rm -fr $(DOX_HTML_DIR) $(DOX_LATEX_DIR)
	@$(DOXYGEN) $(DOX_CONFIG)

$(DOX_TAG): $(dox_inputs)
	@rm -fr $(DOX_HTML_DIR) $(DOX_LATEX_DIR)
	@$(DOXYGEN) $(DOX_CONFIG)
	@$(SED) -i 's/ModuleDocumentation <\/li>/<a class="el" href="modules.html">Modules<\/a>/' html/index.html
	@$(SED) -i 's/FileDocumentation <\/li>/<a class="el" href="files.html">Files<\/a>/' html/index.html
        # comment for the line below: what we really want to do is to remove the line, but dy doing so, it avoids opening the interactive menu when browsing files
	@if test -f html/navtree.js ; then $(SED) -i 's/\[ "Files", "Files.html", null \]/\[ "", "Files.html", null \]/' html/navtree.js ; fi
	@$(SED) -i 's/.*"Files.html".*//' html/pages.html
	@if test -f latex/main.tex ; then mv latex/main.tex latex/index.tex ; fi
	@$(SED) -i '/\\begin{titlepage}/,$$d' $(DOX_LATEX_DIR)/refman.tex
	@cat $(top_srcdir)/doc/doxygen/refman.tex >> $(DOX_LATEX_DIR)/refman.tex

$(DOX_HTML_DIR): $(DOX_TAG)

$(DOX_PDF): $(DOX_TAG) refman.tex
	@cp $(top_srcdir)/doc/doxygen/chapters/version.sty $(DOX_LATEX_DIR)
	@cp $(top_srcdir)/doc/doxygen/chapters/images/*pdf $(DOX_LATEX_DIR)
	@echo $(PDFLATEX) $(DOX_LATEX_DIR)/refman.tex
	@cd $(DOX_LATEX_DIR) ;\
	rm -f *.aux *.toc *.idx *.ind *.ilg *.log *.out ;\
	$(SED) -i -e 's/__env__/\\_Environment Variables!/' -e 's/\\-\\_\\-\\-\\_\\-env\\-\\_\\-\\-\\_\\-//' ExecutionConfigurationThroughEnvironmentVariables.tex ;\
	$(SED) -i -e 's/__configure__/\\_Configure Options!/' -e 's/\\-\\_\\-\\-\\_\\-configure\\-\\_\\-\\-\\_\\-//' CompilationConfiguration.tex ;\
	$(SED) -i s'/\\item Module\\.Documentation/\\item \\hyperlink{ModuleDocumentation}{Module Documentation}/' index.tex ;\
	$(SED) -i s'/\\item File\\.Documentation/\\item \\hyperlink{FileDocumentation}{File Documentation}/' index.tex ;\
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

CLEANFILES = $(DOX_TAG) starpu_config.h \
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
	cp -pr starpu.pdf html $(PUBLISHDIR)

