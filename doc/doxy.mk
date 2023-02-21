# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2009-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

txtdir   = $(docdir)/manual

EXTRA_DIST =

if STARPU_BUILD_DOC
if STARPU_BUILD_DOC_PDF
all: $(DOX_HTML_DIR) $(DOX_DIR)/$(DOX_PDF)
EXTRA_DIST += $(DOX_HTML_DIR) $(DOX_DIR)/$(DOX_PDF)
txt_DATA = $(DOX_DIR)/$(DOX_PDF)
else
all: $(DOX_HTML_DIR)
EXTRA_DIST += $(DOX_HTML_DIR)
endif # STARPU_BUILD_DOC_PDF
DOX_HTML_SRCDIR=$(DOX_HTML_DIR)
install-exec-hook: $(DOX_HTML_DIR)
	@$(MKDIR_P) $(DESTDIR)$(docdir)/manual/$(DOX_HTML_DIR)
	@(cd $(DOX_HTML_SRCDIR) && $(PROG_FIND) . -type f -exec $(INSTALL_DATA) {} $(DESTDIR)$(docdir)/manual/$(DOX_HTML_DIR) \;)
uninstall-hook:
	@rm -rf $(DESTDIR)$(docdir)/manual/$(DOX_HTML_DIR)
else
if STARPU_AVAILABLE_DOC
EXTRA_DIST += $(top_srcdir)/doc/$(DOX_MAIN_DIR)/$(DOX_HTML_DIR)
DOX_HTML_SRCDIR=$(top_srcdir)/doc/$(DOX_MAIN_DIR)/$(DOX_HTML_DIR)
install-exec-hook:
	@$(MKDIR_P) $(DESTDIR)$(docdir)/manual/$(DOX_HTML_DIR)
	@(cd $(DOX_HTML_SRCDIR) && $(PROG_FIND) . -type f -exec $(INSTALL_DATA) {} $(DESTDIR)$(docdir)/manual/$(DOX_HTML_DIR) \;)
uninstall-hook:
	@rm -rf $(DESTDIR)$(docdir)/manual/$(DOX_HTML_DIR)
endif # STARPU_AVAILABLE_DOC
if STARPU_AVAILABLE_DOC_PDF
EXTRA_DIST += $(top_srcdir)/doc/$(DOX_MAIN_DIR)/$(DOX_PDF)
txt_DATA = $(top_srcdir)/doc/$(DOX_MAIN_DIR)/$(DOX_PDF)
endif # STARPU_AVAILABLE_DOC_PDF
endif # STARPU_BUILD_DOC

if STARPU_BUILD_DOC
EXTRA_DIST += \
	      $(top_srcdir)/doc/$(DOX_MAIN_DIR)/chapters/version.sty \
	      $(top_srcdir)/doc/$(DOX_MAIN_DIR)/chapters/version.html

chapters/version.sty: $(chapters)
	$(MKDIR_P) $(top_srcdir)/doc/$(DOX_MAIN_DIR)/chapters
	@for f in $(chapters) ; do \
                if test -f $(top_srcdir)/doc/$(DOX_MAIN_DIR)/$$f ; then $(PROG_STAT) --format=%Y $(top_srcdir)/doc/$(DOX_MAIN_DIR)/$$f ; fi \
        done | sort -r | head -1 > timestamp_sty
	@if test -s timestamp_sty ; then \
		LC_ALL=C $(PROG_DATE) --date=@`cat timestamp_sty` +"%F" > timestamp_sty_updated ;\
		LC_ALL=C $(PROG_DATE) --date=@`cat timestamp_sty` +"%B %Y" > timestamp_sty_updated_month ;\
	fi
	@if test -s timestamp_sty_updated ; then \
		echo ':newcommand{:STARPUUPDATED}{'`cat timestamp_sty_updated`'}' > $(top_srcdir)/doc/$(DOX_MAIN_DIR)/chapters/version.sty;\
	else \
		echo ':newcommand{:STARPUUPDATED}{unknown date}' > $(top_srcdir)/doc/$(DOX_MAIN_DIR)/chapters/version.sty;\
	fi
	@echo ':newcommand{:STARPUVERSION}{$(VERSION)}' >> $(top_srcdir)/doc/$(DOX_MAIN_DIR)/chapters/version.sty
	@$(SED) -i 's/:/\\/g' $(top_srcdir)/doc/$(DOX_MAIN_DIR)/chapters/version.sty
	@for f in timestamp_sty timestamp_sty_updated timestamp_sty_updated_month ; do \
		if test -f $$f ; then $(RM) $$f ; fi ;\
	done

chapters/version.html: $(chapters) $(images)
	@for f in $(chapters) ; do \
                if test -f $(top_srcdir)/doc/$(DOX_MAIN_DIR)/$$f ; then $(PROG_STAT) --format=%Y $(top_srcdir)/doc/$(DOX_MAIN_DIR)/$$f ; fi \
        done | sort -r | head -1 > timestamp_html
	@if test -s timestamp_html ; then \
		LC_ALL=C $(PROG_DATE) --date=@`cat timestamp_html` +"%F" > timestamp_html_updated ;\
		LC_ALL=C $(PROG_DATE) --date=@`cat timestamp_html` +"%B %Y" > timestamp_html_updated_month ;\
	fi
	@echo "This manual documents the version $(VERSION) of StarPU." > $(top_srcdir)/doc/$(DOX_MAIN_DIR)/chapters/version.html
	@if test -s timestamp_html_updated ; then \
		echo "Its contents was last updated on "`cat timestamp_html_updated`"." >> $(top_srcdir)/doc/$(DOX_MAIN_DIR)/chapters/version.html;\
	else \
		echo "Its contents was last updated on <em>unknown_date</em>." >> $(top_srcdir)/doc/$(DOX_MAIN_DIR)/chapters/version.html;\
	fi
	@for f in timestamp_html timestamp_html_updated timestamp_html_updated_month ; do \
		if test -f $$f ; then $(RM) $$f ; fi ;\
	done

doxy:
	@rm -fr $(DOX_HTML_DIR) $(DOX_LATEX_DIR)
	@$(DOXYGEN) $(DOX_CONFIG)

$(DOX_HTML_DIR): $(DOX_TAG)
	@$(MKDIR_P) $(DOX_HTML_DIR)

$(DOX_TAG): $(dox_inputs)
	@rm -fr $(DOX_HTML_DIR) $(DOX_LATEX_DIR)
	@$(DOXYGEN) $(DOX_CONFIG)
	@if test -f $(DOX_HTML_DIR)/DocOrganization.html ; then $(SED) -i 's/ModuleDocumentation <\/li>/<a class="el" href="modules.html">Modules<\/a>/' $(DOX_HTML_DIR)/DocOrganization.html ; fi
	@if test -f $(DOX_HTML_DIR)/DocOrganization.html ; then $(SED) -i 's/FileDocumentation <\/li>/<a class="el" href="files.html">Files<\/a>/' $(DOX_HTML_DIR)/DocOrganization.html ; fi
        # comment for the line below: what we really want to do is to remove the line, but dy doing so, it avoids opening the interactive menu when browsing files
	@if test -f $(DOX_HTML_DIR)/navtreedata.js ; then $(SED) -i 's/\[ "Files", "Files.html", null \]/\[ "", "Files.html", null \]/' $(DOX_HTML_DIR)/navtreedata.js ; fi
	@$(SED) -i 's/.*"Files.html".*//' $(DOX_HTML_DIR)/pages.html
	@if test -f $(DOX_LATEX_DIR)/main.tex ; then mv $(DOX_LATEX_DIR)/main.tex $(DOX_LATEX_DIR)/index.tex ; fi
	@if test -f $(DOX_LATEX_DIR)/refman.tex ; then $(SED) -i '/\\begin{titlepage}/,$$d' $(DOX_LATEX_DIR)/refman.tex ; fi
	@if test -f $(DOX_LATEX_DIR)/refman.tex ; then cat $(top_srcdir)/doc/$(DOX_MAIN_DIR)/refman.tex >> $(DOX_LATEX_DIR)/refman.tex ; fi
	$(top_srcdir)/doc/sectionNumbering.py $(top_builddir)/doc/$(DOX_MAIN_DIR) $(DOX_HTML_DIR)

$(DOX_DIR)/$(DOX_PDF): $(DOX_TAG) refman.tex $(images)
	$(MKDIR_P) $(DOX_LATEX_DIR)
	@cp $(top_srcdir)/doc/$(DOX_MAIN_DIR)/chapters/version.sty $(DOX_LATEX_DIR)
	@cp $(top_srcdir)/doc/title.tex $(DOX_LATEX_DIR)
	@if test -f $(top_srcdir)/doc/$(DOX_MAIN_DIR)/modules.tex ; then cp $(top_srcdir)/doc/$(DOX_MAIN_DIR)/modules.tex $(DOX_LATEX_DIR) ; fi
	@echo $(PDFLATEX) $(DOX_LATEX_DIR)/refman.tex
	@cd $(DOX_LATEX_DIR) ;\
	rm -f *.aux *.toc *.idx *.ind *.ilg *.log *.out ;\
        for f in group__API__* ; do sed -i '1 i \\\clearpage' $$f ; done ;\
	if test -f ExecutionConfigurationThroughEnvironmentVariables.tex ; then $(SED) -i -e 's/__env__/\\_Environment Variables!/' -e 's/\\-\\_\\-\\-\\_\\-env\\-\\_\\-\\-\\_\\-//' ExecutionConfigurationThroughEnvironmentVariables.tex ; fi ;\
	if test -f CompilationConfiguration.tex ; then $(SED) -i -e 's/__configure__/\\_Configure Options!/' -e 's/\\-\\_\\-\\-\\_\\-configure\\-\\_\\-\\-\\_\\-//' CompilationConfiguration.tex ; fi ;\
	if test -f DocOrganization.tex ; then $(SED) -i s'/\\item Module\\.Documentation/\\item \\hyperlink{ModuleDocumentation}{Module Documentation}/' DocOrganization.tex ; fi ;\
	if test -f DocOrganization.tex ; then $(SED) -i s'/\\item File\\.Documentation/\\item \\hyperlink{FileDocumentation}{File Documentation}/' DocOrganization.tex ; fi ;\
	max_print_line=1000000 $(PDFLATEX) -interaction batchmode refman.tex ;\
	! < refman.log grep -v group__ | grep -v _amgrp | grep -v deprecated__ | grep "multiply defined" || exit 1 ;\
	$(MAKEINDEX) refman.idx ;\
	max_print_line=1000000 $(PDFLATEX) -interaction batchmode refman.tex ;\
	for i in $(shell seq 1 5); do \
           if $(EGREP) 'Rerun (LaTeX|to get cross-references right)' refman.log > /dev/null 2>&1; then \
	       max_print_line=1000000 $(PDFLATEX) -interaction batchmode refman.tex; \
	   else \
		break ; \
	   fi; \
	done
	mv $(DOX_LATEX_DIR)/refman.pdf $(DOX_DIR)/$(DOX_PDF)

CLEANFILES = $(DOX_TAG) $(DOX_STARPU_CONFIG) \
    -r \
    $(DOX_HTML_DIR) \
    $(DOX_LATEX_DIR) \
    $(DOX_DIR)/$(DOX_PDF)

endif

EXTRA_DIST += refman.tex $(chapters) $(images)
