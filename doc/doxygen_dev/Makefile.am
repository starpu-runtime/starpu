# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2010-2018                                CNRS
# Copyright (C) 2013-2018                                Inria
# Copyright (C) 2009,2011,2013,2014,2017                 Université de Bordeaux
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

if BUILD_DOC
all: $(DOX_HTML_DIR) $(DOX_PDF)
EXTRA_DIST += $(DOX_HTML_DIR) $(DOX_PDF)
txt_DATA = $(DOX_PDF)
DOX_HTML_SRCDIR=$(DOX_HTML_DIR)
install-exec-hook:
	$(MKDIR_P) $(DESTDIR)$(docdir)/manual/html_dev
	(cd $(DOX_HTML_SRCDIR) && find . -type f -exec $(INSTALL) -c -m 644 {} $(DESTDIR)$(docdir)/manual/html_dev \;)
uninstall-hook:
	rm -rf $(DESTDIR)$(docdir)/manual/html_dev
else
if AVAILABLE_DOC
EXTRA_DIST += $(top_srcdir)/doc/doxygen_dev/html_dev $(top_srcdir)/doc/doxygen_dev/starpu_dev.pdf
txt_DATA = $(top_srcdir)/doc/doxygen_dev/starpu_dev.pdf
DOX_HTML_SRCDIR=$(top_srcdir)/doc/doxygen_dev/html_dev
install-exec-hook:
	$(MKDIR_P) $(DESTDIR)$(docdir)/manual/html_dev
	(cd $(DOX_HTML_SRCDIR) && find . -type f -exec $(INSTALL) -c -m 644 {} $(DESTDIR)$(docdir)/manual/html_dev \;)
uninstall-hook:
	rm -rf $(DESTDIR)$(docdir)/manual/html_dev
endif
endif

chapters =	\
	chapters/000_introduction.doxy \
	chapters/010_core.doxy

images =

if BUILD_DOC
starpu_config.h: $(top_srcdir)/include/starpu_config.h.in
	@$(SED) 's/#undef \(.*\)/#define \1 1/' $< > $@

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
	starpu_config.h					\
	chapters/version.sty				\
	chapters/version.html				\
	$(top_srcdir)/src/core/workers.h

$(DOX_HTML_DIR): $(DOX_TAG) refman.tex
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
	@-cp $(top_srcdir)/doc/doxygen_dev/chapters/images/*pdf $(DOX_LATEX_DIR)
	@echo $(PDFLATEX) $(DOX_LATEX_DIR)/refman.tex
	@cd $(DOX_LATEX_DIR) ;\
	rm -f *.aux *.toc *.idx *.ind *.ilg *.log *.out ;\
	$(SED) -i -e 's/__env__/\\_Environment Variables!/' -e 's/\\-\\_\\-\\-\\_\\-env\\-\\_\\-\\-\\_\\-//' ExecutionConfigurationThroughEnvironmentVariables.tex ;\
	$(SED) -i -e 's/__configure__/\\_Configure Options!/' -e 's/\\-\\_\\-\\-\\_\\-configure\\-\\_\\-\\-\\_\\-//' CompilationConfiguration.tex ;\
	$(SED) -i s'/\\item Module\\-Documentation/\\item \\hyperlink{ModuleDocumentation}{Module Documentation}/' index.tex ;\
	$(SED) -i s'/\\item File\\-Documentation/\\item \\hyperlink{FileDocumentation}{File Documentation}/' index.tex ;\
	$(PDFLATEX) refman.tex ;\
	$(MAKEINDEX) refman.idx ;\
	$(PDFLATEX) refman.tex ;\
	done=0; repeat=5 ;\
	while test $$done = 0 -a $$repeat -gt 0; do \
           if $(EGREP) 'Rerun (LaTeX|to get cross-references right)' refman.log > /dev/null 2>&1; then \
	       $(PDFLATEX) refman.tex; \
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

EXTRA_DIST += doxygen.cfg $(chapters) $(images)

# Rule to update documentation on web server. Should only be used locally.
PUBLISHHOST	?= gforge
update-web: $(DOX_PDF)
	scp -pr starpu_dev.pdf html_dev $(PUBLISHHOST):/home/groups/starpu/htdocs/doc

showcheck:
	-cat /dev/null
