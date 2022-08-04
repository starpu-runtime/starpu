# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2021-2022       Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
include $(top_srcdir)/starpu-notests.mk

EXTRA_DIST =					\
	deploy/build.xml			\
	deploy/javaCompilerArgs			\
	starpu/handlers/SvgHandler.java		\
	starpu/handlers/TraceVizHandler.java	\
	starpu/handlers/TaskGraphHandler.java	\
	starpu/handlers/TraceGenHandler.java	\
	starpu/handlers/TraceUtils.java

ECLIPSE_DIR		=	$(shell dirname @ECLIPSE@)
ECLIPSE_WORKSPACE	=	$(abs_top_builddir)/eclipse-plugin/workspace

all: package
	cp $(abs_top_srcdir)/eclipse-plugin/build.xml $(abs_top_srcdir)/eclipse-plugin/plugin_build.xml
	rm -f build/plugin/plugins/StarPU_*jar
	$(ECLIPSE_DIR)/eclipse -noSplash -data $(ECLIPSE_WORKSPACE) -application org.eclipse.ant.core.antRunner -buildfile $(abs_top_srcdir)/eclipse-plugin/src/deploy/build.xml
	rm -f $(ECLIPSE_DIR)/dropins/StarPU_*jar
	cp build/plugin/plugins/*.jar $(ECLIPSE_DIR)/dropins/
	mv $(abs_top_srcdir)/eclipse-plugin/plugin_build.xml $(abs_top_srcdir)/eclipse-plugin/build.xml

package:
	$(ECLIPSE_DIR)/eclipse -noSplash -data $(ECLIPSE_WORKSPACE) -application org.eclipse.cdt.managedbuilder.core.headlessbuild -import $(abs_top_srcdir)/eclipse-plugin
	rm -f $(ECLIPSE_DIR)/dropins/StarPU_*jar
	$(ECLIPSE_DIR)/eclipse -noSplash -data $(ECLIPSE_WORKSPACE) -application org.eclipse.ant.core.antRunner -buildfile $(abs_top_srcdir)/eclipse-plugin/build.xml

txtdir = $(libdir)/starpu/eclipse-plugin/workspace
script = $(abs_top_srcdir)/eclipse-plugin/tools/install_workspace.sh

install-data-hook:
	(cd $(abs_top_builddir)/eclipse-plugin/workspace && $(PROG_FIND) . -type f -exec $(script) {} $(DESTDIR)$(txtdir) $(INSTALL_DATA) \;)

clean-local:
	rm -rf build

distclean-local: clean-local
	rm -f $(ECLIPSE_DIR)/dropins/StarPU_*jar
	rm -rf $(ECLIPSE_WORKSPACE)
