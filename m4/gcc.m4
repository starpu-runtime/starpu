dnl -*- Autoconf -*-
dnl
dnl Copyright (C) 2011 Institut National de Recherche en Informatique et Automatique
dnl
dnl StarPU is free software; you can redistribute it and/or modify
dnl it under the terms of the GNU Lesser General Public License as published by
dnl the Free Software Foundation; either version 2.1 of the License, or (at
dnl your option) any later version.
dnl
dnl StarPU is distributed in the hope that it will be useful, but
dnl WITHOUT ANY WARRANTY; without even the implied warranty of
dnl MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
dnl
dnl See the GNU Lesser General Public License in COPYING.LGPL for more details.

dnl Check whether GCC plug-in support is available (GCC 4.5+).
AC_DEFUN([STARPU_GCC_PLUGIN_SUPPORT], [
  AC_REQUIRE([AC_PROG_CC])
  AC_CACHE_CHECK([whether GCC supports plug-ins], [ac_cv_have_gcc_plugins], [
    if test "x$GCC" = xyes; then
      GCC_PLUGIN_INCLUDE_DIR="`"$CC" -print-file-name=plugin`/include"

      save_CPPFLAGS="$CPPFLAGS"
      CPPFLAGS="-I$GCC_PLUGIN_INCLUDE_DIR"

      AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[#include <gcc-plugin.h>
	    #include <tree.h>
	    #include <gimple.h>
	    tree fndecl; gimple call;]],
	  [[fndecl = lookup_name (get_identifier ("puts"));
	    call = gimple_build_call (fndecl, 0);]])],
	[ac_cv_have_gcc_plugins="yes"],
	[ac_cv_have_gcc_plugins="no"])

      CPPFLAGS="$save_CPPFLAGS"
    else
      ac_cv_have_gcc_plugins="no"
    fi
  ])
  AC_SUBST([GCC_PLUGIN_INCLUDE_DIR])
])
