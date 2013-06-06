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

chapters =	\
	chapters/advanced_examples.doxy \
	chapters/basic_examples.doxy \
	chapters/building.doxy \
	chapters/c_extensions.doxy \
	chapters/fft_support.doxy \
	chapters/introduction.doxy \
	chapters/mpi_support.doxy \
	chapters/optimize_performance.doxy \
	chapters/performance_feedback.doxy \
	chapters/scheduling_context_hypervisor.doxy \
	chapters/scheduling_contexts.doxy \
	chapters/socl_opencl_extensions.doxy \
	chapters/tips_and_tricks.doxy \
	chapters/hello_pragma2.c \
	chapters/hello_pragma.c \
	chapters/api/codelet_and_tasks.doxy \
	chapters/api/cuda_extensions.doxy \
	chapters/api/data_interfaces.doxy \
	chapters/api/data_management.doxy \
	chapters/api/data_partition.doxy \
	chapters/api/expert_mode.doxy \
	chapters/api/explicit_dependencies.doxy \
	chapters/api/fft_support.doxy \
	chapters/api/fxt_support.doxy \
	chapters/api/implicit_dependencies.doxy \
	chapters/api/initialization.doxy \
	chapters/api/insert_task.doxy \
	chapters/api/lower_bound.doxy \
	chapters/api/misc_helpers.doxy \
	chapters/api/mpi.doxy \
	chapters/api/multiformat_data_interface.doxy \
	chapters/api/opencl_extensions.doxy \
	chapters/api/parallel_tasks.doxy \
	chapters/api/performance_model.doxy \
	chapters/api/profiling.doxy \
	chapters/api/running_driver.doxy \
	chapters/api/scheduling_context_hypervisor.doxy \
	chapters/api/scheduling_contexts.doxy \
	chapters/api/scheduling_policy.doxy \
	chapters/api/standard_memory_library.doxy \
	chapters/api/task_bundles.doxy \
	chapters/api/task_lists.doxy \
	chapters/api/top.doxy \
	chapters/api/versioning.doxy \
	chapters/api/workers.doxy

chapters/version.sty: $(chapters)
	@-for f in $(chapters) ; do \
                if test -f $(top_srcdir)/doc/doxygen/$$f ; then stat --format=%Y $(top_srcdir)/doc/doxygen/$$f 2>/dev/null ; fi \
        done | sort -r | head -1 > timestamp
	@if test -s timestamp ; then \
		LC_ALL=C date --date=@`cat timestamp` +"%d %B %Y" > timestamp_updated 2>/dev/null;\
		LC_ALL=C date --date=@`cat timestamp` +"%B %Y" > timestamp_updated_month 2>/dev/null;\
	fi
	@if test -s timestamp_updated ; then \
		echo "\newcommand{\STARPUUPDATED}{" `cat timestamp_updated` "}" > $(top_srcdir)/doc/doxygen/chapters/version.sty;\
		echo "\newcommand{\STARPUUPDATEDMONTH}{" `cat timestamp_updated_month` "}" >> $(top_srcdir)/doc/doxygen/chapters/version.sty;\
	else \
		echo "\newcommand{\STARPUUPDATED}{unknown_date}" > $(top_srcdir)/doc/doxygen/chapters/version.sty;\
		echo "\newcommand{\STARPUUPDATEDMONTH}{unknown_date}" >> $(top_srcdir)/doc/doxygen/chapters/version.sty; \
	fi
	@echo "\newcommand{\STARPUEDITION}{$(VERSION)}" >> $(top_srcdir)/doc/doxygen/chapters/version.sty
	@echo "\newcommand{\STARPUVERSION}{$(VERSION)}" >> $(top_srcdir)/doc/doxygen/chapters/version.sty
	@-for f in timestamp timestamp_updated timestamp_updated_month ; do \
		if test -f $$f ; then $(RM) $$f ; fi ;\
	done

#chapters/version.html: $(chapters)


showcheck:
	-cat /dev/null
