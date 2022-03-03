# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2011-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
# STARPU_SEARCH_LIBS(NAME, FUNCTION, SEARCH-LIBS,
#                    [ACTION-IF-FOUND], [ACTION-IF-NOT-FOUND],
#                    [OTHER-LIBRARIES])
#
# Like AC_SEARCH_LIBS, but puts -l flags into $1_LDFLAGS instead of LIBS, and
# AC_SUBSTs it
AC_DEFUN([STARPU_SEARCH_LIBS], [dnl
	_LIBS_SAV="$LIBS"
	LIBS=""
	AC_SEARCH_LIBS([$2], [$3], [$4], [$5], [$6])
	STARPU_$1_LDFLAGS="$STARPU_$1_LDFLAGS $LIBS"
	LIBS=$_LIBS_SAV
	AC_SUBST(STARPU_$1_LDFLAGS)
])dnl

# STARPU_CHECK_LIB(NAME, LIBRARY, FUNCTION,
#                  [ACTION-IF-FOUND], [ACTION-IF-NOT-FOUND],
#                  [OTHER-LIBRARIES])
#
# Like AC_CHECK_LIB, but puts -l flags into $1_LDFLAGS instead of LIBS, and
# AC_SUBSTs it
AC_DEFUN([STARPU_CHECK_LIB], [dnl
	_LIBS_SAV="$LIBS"
	LIBS=""
	AC_CHECK_LIB([$2], [$3], [$4], [$5], [$6])
	STARPU_$1_LDFLAGS="$STARPU_$1_LDFLAGS $LIBS"
	LIBS=$_LIBS_SAV
	AC_SUBST(STARPU_$1_LDFLAGS)
])dnl

# STARPU_HAVE_LIBRARY(NAME, LIBRARY,
#                     [ACTION-IF-FOUND], [ACTION-IF-NOT-FOUND],
#                     [OTHER-LIBRARIES])
# Like AC_HAVE_LIBRARY, but puts -l flags into $1_LDFLAGS instead of LIBS, and
# AC_SUBSTs it
AC_DEFUN([STARPU_HAVE_LIBRARY], [dnl
STARPU_CHECK_LIB([$1], [$2], main, [$3], [$4], [$5])
])dnl

# STARPU_INIT_ZERO(INCLUDES, TYPE, INIT_MACRO)
# Checks whether when TYPE is initialized with INIT_MACRO, the content is just
# plain zeroes
AC_DEFUN([STARPU_INIT_ZERO], [dnl
AC_MSG_CHECKING(whether $3 just zeroes)
AC_RUN_IFELSE([AC_LANG_PROGRAM(
		$1,
		[[$2 var = $3;
		 char *p;
		 for (p = (char*) &var; p < (char*) (&var+1); p++)
		   if (*p != 0)
		     return 1;
		 return 0;
		]],
		)],
		[AC_DEFINE([STARPU_$3_ZERO], [1], [Define to 1 if `$3' is just zeroes])
		 AC_MSG_RESULT(yes)],
		[AC_MSG_RESULT(no)])
])dnl

# IS_SUPPORTED_CFLAG(flag)
# ------------------------
# Check if the CFLAGS `flag' is supported by the compiler
AC_DEFUN([IS_SUPPORTED_CFLAG],
[
	AC_REQUIRE([AC_PROG_CC])
	AC_MSG_CHECKING([whether C compiler supports $1])

	SAVED_CFLAGS="$CFLAGS"
	CFLAGS="$1"

	AC_LINK_IFELSE(
		AC_LANG_PROGRAM(
			[[]],
			[[AC_LANG_SOURCE([const char *hello = "Hello World";])]]
		),
		[
			m4_default_nblank([$2], [GLOBAL_AM_CFLAGS="$GLOBAL_AM_CFLAGS $1"])
			AC_MSG_RESULT(yes)
		],
		[
			AC_MSG_RESULT(no)
		]
	)
	CFLAGS="$SAVED_CFLAGS"
])

# IS_SUPPORTED_CXXFLAG(flag)
# ------------------------
# Check if the CXXFLAGS `flag' is supported by the compiler
AC_DEFUN([IS_SUPPORTED_CXXFLAG],
[
	AC_REQUIRE([AC_PROG_CXX])
	AC_LANG_PUSH([C++])
	AC_MSG_CHECKING([whether CXX compiler supports $1])

	SAVED_CXXFLAGS="$CXXFLAGS"
	CXXFLAGS="$1"

	AC_LINK_IFELSE(
		AC_LANG_PROGRAM(
			[[]],
			[[AC_LANG_SOURCE([const char *hello = "Hello World";])]]
		),
		[
			m4_default_nblank([$2], [GLOBAL_AM_CXXFLAGS="$GLOBAL_AM_CXXFLAGS $1"])
			AC_MSG_RESULT(yes)
		],
		[
			AC_MSG_RESULT(no)
		]
	)
	CXXFLAGS="$SAVED_CXXFLAGS"
	AC_LANG_POP([C++])
])

# IS_SUPPORTED_FFLAG(flag)
# ------------------------
# Check if the FFLAGS `flag' is supported by the compiler
AC_DEFUN([IS_SUPPORTED_FFLAG],
[
	AC_LANG_PUSH([Fortran 77])
	AC_MSG_CHECKING([whether Fortran 77 compiler supports $1])

	SAVED_FFLAGS="$FFLAGS"
	FFLAGS="$1"

	AC_LINK_IFELSE(
		AC_LANG_PROGRAM(
			[],
			[[AC_LANG_SOURCE([])]]
		),
		[
			m4_default_nblank([$2], [GLOBAL_AM_FFLAGS="$GLOBAL_AM_FFLAGS $1"])
			AC_MSG_RESULT(yes)
		],
		[
			AC_MSG_RESULT(no)
		]
	)
	FFLAGS="$SAVED_FFLAGS"
	AC_LANG_POP([Fortran 77])
])

# IS_SUPPORTED_FCFLAG(flag)
# ------------------------
# Check if the FCLAGS `flag' is supported by the compiler
AC_DEFUN([IS_SUPPORTED_FCFLAG],
[
	AC_LANG_PUSH([Fortran])
	AC_MSG_CHECKING([whether Fortran compiler supports $1])

	SAVED_FCFLAGS="$FCFLAGS"
	FCFLAGS="$1"

	AC_LINK_IFELSE(
		AC_LANG_PROGRAM(
			[],
			[[AC_LANG_SOURCE([])]]
		),
		[
			m4_default_nblank([$2], [GLOBAL_AM_FCFLAGS="$GLOBAL_AM_FCFLAGS $1"])
			AC_MSG_RESULT(yes)
		],
		[
			AC_MSG_RESULT(no)
		]
	)
	FCFLAGS="$SAVED_FCFLAGS"
	AC_LANG_POP([Fortran])
])

# IS_SUPPORTED_FLAG(flag)
# ------------------------
# Check with C, C++, F77 and F90 that the `flag' is supported by the compiler
AC_DEFUN([IS_SUPPORTED_FLAG],
[
	IS_SUPPORTED_CFLAG($1)
	IS_SUPPORTED_CXXFLAG($1)
	IS_SUPPORTED_FFLAG($1)
	IS_SUPPORTED_FCFLAG($1)
])

AC_DEFUN([IS_SUPPORTED_FLAG_VAR],
[
	IS_SUPPORTED_CFLAG($1,[$2_CFLAGS="$$2_CFLAGS $1"])
	IS_SUPPORTED_CXXFLAG($1,[$2_CXXFLAGS="$$2_CXXFLAGS $1"])
	IS_SUPPORTED_FFLAG($1,[$2_FFLAGS="$$2_FFLAGS $1"])
	IS_SUPPORTED_FCFLAG($1,[$2_FCFLAGS="$$2_FCFLAGS $1"])
])

# AC_PYTHON_MODULE(modulename, [action-if-found], [action-if-not-found])
# Check if the given python module is available
AC_DEFUN([AC_PYTHON_MODULE],
[
	echo "import $1" | $PYTHON - 2>/dev/null
	if test $? -ne 0 ; then
	   	$3
	else
		$2
	fi
])
