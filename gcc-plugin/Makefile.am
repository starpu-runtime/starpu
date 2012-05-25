# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2011, 2012 Institut National de Recherche en Informatique et Automatique
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

SUBDIRS = src tests examples

EXTRA_DIST = COPYING README ChangeLog

showcheck:
	for i in $(SUBDIRS) ; do \
		make -C $$i showcheck ; \
	done

# Generate a GNU-style ChangeLog for inclusion in the tarball.
# It requires network access and may be slow.
gen-ChangeLog:
	if test "x$$CHANGELOG" = "xno"; then					\
	   echo "ChangeLog not built, per user request" >&2;			\
	elif ! xsltproc --version > /dev/null 2>&1; then			\
	   echo "xsltproc not found, ChangeLog not generated" >&2;		\
	elif ! test -d "$(srcdir)/.svn"; then					\
	   echo "Subversion meta-data not found, ChangeLog not generated" >&2;	\
	elif ! svn --version > /dev/null 2>&1; then				\
	   echo "Subversion not found, ChangeLog not generated" >&2;		\
	else									\
	   ( cd "$(srcdir)";							\
	     svn log --xml --verbose ) |					\
	   xsltproc "$(top_srcdir)/build-aux/svn2cl.xsl" - > "ChangeLog.tmp";	\
	   mv "ChangeLog.tmp" "$(distdir)/ChangeLog";				\
	fi

dist-hook: gen-ChangeLog

.PHONY: showcheck dist-hook gen-ChangeLog
