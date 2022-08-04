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
	./hello/hello.c				\
	./hello/.settings/language.settings.xml	\
	./hello/.project

ECLIPSE_DIR		=	$(shell dirname @ECLIPSE@)
ECLIPSE_WORKSPACE	=	$(abs_top_builddir)/eclipse-plugin/workspace

txtdir   = $(libdir)/starpu/eclipse-plugin/examples/hello
txt_DATA = hello/hello.c \
	   hello/.cproject \
	   hello/.project

script=$(abs_top_srcdir)/eclipse-plugin/tools/install_workspace.sh
install-data-hook:
	$(INSTALL_DATA) $(abs_top_srcdir)/eclipse-plugin/examples/hello/.settings/language.settings.xml $(txtdir).settings
	$(ECLIPSE_DIR)/eclipse -noSplash -data $(DESTDIR)$(txtdir)/../../workspace -application org.eclipse.cdt.managedbuilder.core.headlessbuild -import $(txtdir)

