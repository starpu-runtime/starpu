#
# StarPU
# Copyright (C) INRIA 2008-2009 (see AUTHORS file)
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation; either version 2.1 of the License, or (at
# your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
# See the GNU Lesser General Public License in COPYING.LGPL for more details.
#

AC_INIT([StarPU],0.1, [http://runtime.bordeaux.inria.fr/StarPU/], starpu)
AC_CONFIG_AUX_DIR([build-aux])
AM_INIT_AUTOMAKE([-Wall -Werror foreign])
AC_MSG_NOTICE([Configuring StarPU])

AC_CONFIG_SRCDIR(include/starpu.h)

AC_PREREQ(2.60)


AC_PROG_CC
AC_PROG_CPP
AC_PROG_SED
AC_PROG_LN_S
AC_PROG_F77

AC_PROG_LIBTOOL

AC_PROG_INSTALL
AC_PROG_MKDIR_P
AC_PROG_LN_S

AC_HEADER_STDC

# This will be useful for program which use CUDA (and .cubin files) which need
# some path to the CUDA code at runtime.
AC_DEFINE_UNQUOTED(STARPUDIR, "$PWD", [location of StarPU sources])
AC_SUBST(STARPUDIR, $PWD)

AC_SEARCH_LIBS([pthread_create],[pthread],,AC_MSG_ERROR([pthread library unavailable]))
AC_SEARCH_LIBS([sqrt],[m],,AC_MSG_ERROR([math library unavailable]))

###############################################################################
#                                                                             #
#                                 CPUs settings                               #
#                                                                             #
###############################################################################

AC_MSG_CHECKING(whether CPUs should be used)
AC_ARG_ENABLE(cpu, [AS_HELP_STRING([--disable-cpu],
			[do not use the CPU(s)])],
			enable_cpu=$enableval, enable_cpu=yes)
AC_MSG_RESULT($enable_cpu)
AC_SUBST(USE_CPU, $enable_cpu)
AM_CONDITIONAL(USE_CPU, test x$enable_cpu = xyes)

if test x$enable_cpu = xyes; then
	AC_DEFINE(USE_CPUS, [1], [CPU driver is activated])

	# This value is set quite randomly, but StarPU should not take more
	# core than there are in the system
	AC_DEFINE(NMAXCORES, [16], [Maximum number of CPUs supported])
fi

###############################################################################
#                                                                             #
#                                 CUDA settings                               #
#                                                                             #
###############################################################################

AC_ARG_ENABLE(cuda, [AS_HELP_STRING([--disable-cuda],
		[do not use CUDA device(s)])],, [enable_cuda=maybe])

if test x$enable_cuda = xyes -o x$enable_cuda = xmaybe; then
	#AC_MSG_CHECKING(whether CUDA is available)
	AC_ARG_WITH(cuda-dir, 
		[AS_HELP_STRING([--with-cuda-dir=<path>],
		[specify CUDA installation directory (default is /usr/local/cuda/)])],
		[
			cuda_dir=$withval
			# in case this was not explicit yet
			enable_cuda=yes
		], cuda_dir=/usr/local/cuda/)
	
	if test -d "$cuda_dir/include/"; then
		CPPFLAGS="${CPPFLAGS} -I$cuda_dir/include/ "
	fi
	if test -d "$cuda_dir/lib/"; then
		LDFLAGS="${LDFLAGS} -L$cuda_dir/lib/ "
	fi

	# do we have a valid CUDA setup ?
	have_valid_cuda=yes
	AC_CHECK_HEADER([cuda.h],,[have_valid_cuda=no])
	AC_SEARCH_LIBS([cuInit],[cuda],,[have_valid_cuda=no])

	# we also check that CUBLAS is available
	AC_SEARCH_LIBS([cublasInit],[cublas],,[have_valid_cuda=no])

	# in case CUDA was explicitely required, but is not available, this is an error
	if test x$enable_cuda = xyes -a x$have_valid_cuda = no; then
		AC_MSG_ERROR([cannot find CUDA])
	fi

	# now we enable CUDA if and only if a proper setup is available
	enable_cuda=$have_valid_cuda
fi

AC_MSG_CHECKING(whether CUDA should be used)
AC_MSG_RESULT($enable_cuda)
AC_SUBST(USE_CUDA, $enable_cuda)
AM_CONDITIONAL(USE_CUDA, test x$enable_cuda = xyes)
if test x$enable_cuda = xyes; then
	AC_DEFINE(USE_CUDA, [1], [CUDA support is activated])

	#in case this is a 64bit setup, we tell nvcc to use a -m64 flag
	AC_CHECK_SIZEOF([void *])
	if test x$SIZEOF_VOID_P = x8; then
		NVCCFLAGS="${NVCCFLAGS} -m64"
		AC_SUBST(NVCCFLAGS)
	fi
fi

###############################################################################
#                                                                             #
#                                 Cell settings                               #
#                                                                             #
###############################################################################

AC_MSG_CHECKING(whether GORDON should be used)
AC_ARG_ENABLE(gordon, [AS_HELP_STRING([--enable-gordon],
			[use Cell's SPUs])],
			enable_gordon=$enableval, enable_gordon=no)
AC_MSG_RESULT($enable_gordon)
AC_SUBST(USE_GORDON, $enable_gordon)
AM_CONDITIONAL(USE_GORDON, test x$enable_gordon = xyes)

if test x$enable_gordon = xyes; then
	AC_CHECK_LIB(spe2, spe_context_create,,AC_MSG_ERROR([cannot find libspe2]))
	AC_DEFINE(USE_GORDON, [1], [Cell support is enabled])
fi

###############################################################################
#                                                                             #
#                   Debug and Performance analysis tools                      #
#                                                                             #
###############################################################################

AC_MSG_CHECKING(whether debug messages should be displayed)
AC_ARG_ENABLE(verbose, [AS_HELP_STRING([--enable-verbose],
			[display verbose debug messages])],
			enable_verbose=$enableval, enable_verbose=no)
AC_MSG_RESULT($enable_verbose)
if test x$enable_verbose = xyes; then
	AC_DEFINE(VERBOSE, [1], [display verbose debug messages])
fi


AC_MSG_CHECKING(whether coverage testing should be enabled)
AC_ARG_ENABLE(coverage, [AS_HELP_STRING([--enable-coverage],
			[enable coverage checking])],
			enable_coverage=$enableval, enable_coverage=no)
AC_MSG_RESULT($enable_coverage)
AC_SUBST(COVERAGE, $enable_coverage)
if test x$enable_coverage = xyes; then
	CPPFLAGS="${CPPFLAGS} --coverage"
	LDFLAGS="${LDFLAGS} --coverage"
fi


# shall we use FxT to generate trace of the execution ?
AC_MSG_CHECKING(whether fxt traces should be generated)
AC_ARG_WITH(fxt, [AS_HELP_STRING([--with-fxt[=<dir>]], [generate fxt traces])],
	[
		if test x$withval != xno; then
			use_fxt=yes
			if test x$withval = xyes; then
				AC_MSG_RESULT(yes)
				use_fxt_from_system=yes
			else
				# use specified path
				# TODO check if the dir is actually containing FxT
				use_fxt_from_system=no
				fxtdir=$withval
				AC_MSG_RESULT(yes using $fxtdir)
				AC_SUBST(FXTDIR, $fxtdir)
			fi
		else
			use_fxt=no
			AC_MSG_RESULT(no)
		fi
	],
	[	
		use_fxt=no
		AC_MSG_RESULT(no)
	])
AC_SUBST(USE_FXT, $use_fxt)
AM_CONDITIONAL(USE_FXT, test x$use_fxt = xyes)

if test x$use_fxt = xyes; then
	AC_DEFINE(USE_FXT, [1], [enable FxT traces])
	AC_DEFINE(CONFIG_FUT, [1], [enable FUT traces])

	AC_SUBST(USE_FXTDIR_FROM_USER,$use_fxt_from_system)
	if test x$use_fxt_from_system = xno; then
		CPPFLAGS="${CPPFLAGS} -I$fxtdir/include/ "
		LDFLAGS="${LDFLAGS} -L$fxtdir/lib/ "
	fi

	AC_CHECK_LIB(fxt, fut_setup,,AC_MSG_ERROR([cannot find fxt lib]))
	AC_CHECK_HEADER([fxt/fxt.h],,AC_MSG_ERROR([cannot find headers for fxt]))
	AC_CHECK_HEADER([fxt/fut.h],,AC_MSG_ERROR([cannot find headers for fxt]))

	# In case FxT traces are generated, we may use our (poor) hand-made gtk
	# tool to visualize traces

	PKG_PROG_PKG_CONFIG
	PKG_CHECK_MODULES([GTK], [gtk+-2.0], enable_gtk=yes, enable_gtk=no) 
	AC_SUBST(USE_GTK, $enable_gtk)
	if test x$enable_gtk = xyes; then
		AC_SUBST(GTK_CFLAGS)
		AC_SUBST(GTK_LIBS)
		AC_DEFINE(USE_GTK, [1], [enable GTK])
	fi
fi

AC_MSG_CHECKING(whether performance debugging should be enabled)
AC_ARG_ENABLE(perf-debug, [AS_HELP_STRING([--enable-perf-debug],
			[enable performance debugging])],
			enable_perf_debug=$enableval, enable_perf_debug=no)
AC_MSG_RESULT($enable_perf_debug)
AC_SUBST(PERF_DEBUG, $enable_perf_debug)
if test x$enable_perf_debuf = xyes; then
	AC_DEFINE(PERF_DEBUG, [1], [enable performance debug])
	CPPFLAGS="${CPPFLAGS} -pg "
	LDFLAGS="${LDFLAGS} -pg "
fi

AC_MSG_CHECKING(whether performance model debugging should be enabled)
AC_ARG_ENABLE(model-debug, [AS_HELP_STRING([--enable-model-debug],
			[enable performance model debugging])],
			enable_model_debug=$enableval, enable_model_debug=no)
AC_MSG_RESULT($enable_model_debug)
if  test x$enable_model_debug = xyes; then
	AC_DEFINE(MODEL_DEBUG, [1], [enable performance model debug])
fi

AC_MSG_CHECKING(whether statistics should be generated)
AC_ARG_ENABLE(stats, [AS_HELP_STRING([--enable-stats],
			[enable statistics])],
			enable_stats=$enableval, enable_stats=no)
AC_MSG_RESULT($enable_stats)
AC_SUBST(STATS, $enable_stats)
AC_SUBST(DATA_STATS, $enable_stats)

if test x$enable_stats = xyes; then
        AC_DEFINE(DATA_STATS, [1], [enable statistics])
fi


###############################################################################
#                                                                             #
#                  Miscellaneous options for StarPU                           #
#                                                                             #
###############################################################################

AC_MSG_CHECKING(whether a dynamic library should be generated)
AC_ARG_ENABLE(dynamic, [AS_HELP_STRING([--enable-dynamic],
			[generate a dynamic library])],
			enable_dynamic=$enableval, enable_dynamic=no)
AC_MSG_RESULT($enable_dynamic)
AC_SUBST(DYNAMIC, $enable_dynamic)

AC_MSG_CHECKING(whether priorities should be enabled)
AC_ARG_ENABLE(priority, [AS_HELP_STRING([--disable-priority],
			[do not use priorities])],
			enable_priority=$enableval, enable_priority=yes)
AC_MSG_RESULT($enable_priority)
if test x$enable_priority = xno; then
	AC_DEFINE(NO_PRIO, [1], [Disable priorities])
fi

AC_MSG_CHECKING(whether data RW-lock should be used)
AC_ARG_ENABLE(data-rw-lock, [AS_HELP_STRING([--disable-data-rw-lock],
			[do not use data RW-locks])],
			enable_data_rw_lock=$enableval, enable_data_rw_lock=yes)
AC_MSG_RESULT($enable_data_rw_lock)
if test x$enable_data_rw_lock = xno; then
	AC_DEFINE(NO_DATA_RW_LOCK, [1], [data RW-lock are disabled])
fi


AC_MSG_CHECKING(whether allocation cache should be used)
AC_ARG_ENABLE(allocation-cache, [AS_HELP_STRING([--enable-allocation-cache],
			[enable data allocation cache])],
			enable_allocation_cache=$enableval, enable_allocation_cache=no)
AC_MSG_RESULT($enable_allocation_cache)
if test x$enable_allocation_cache = xyes; then
	AC_DEFINE(USE_ALLOCATION_CACHE, [1], [enable data allocation cache])
fi

# by default, we put the performance models in $PWD/.sampling/
perf_model_dir=($PWD/.sampling/)
AC_ARG_WITH(perf-model-dir, [AS_HELP_STRING([--with-perf-model-dir=<dir>], [specify where performance models shoulds be stored])],
	[
		if x$withval != x$no; then
			perf_model_dir=$withval
		fi
	]
	)
AC_MSG_CHECKING(performance models location)
AC_MSG_RESULT($perf_model_dir)
AC_DEFINE_UNQUOTED(PERF_MODEL_DIR, "$perf_model_dir", [performance models location])

###############################################################################
#                                                                             #
#                                  Examples                                   #
#                                                                             #
###############################################################################

AC_ARG_ENABLE(opengl-render, [AS_HELP_STRING([--enable-opengl-render],
			[enable OpenGL rendering of some examples])],
			enable_opengl_render=$enableval, enable_opengl_render=no)

if test x$enable_opengl_render = xyes; then
	AC_CHECK_LIB(glut, glutInit,,AC_MSG_ERROR([cannot find glut]))
	AC_CHECK_LIB(GL, glXCreateContext,,AC_MSG_ERROR([cannot find GL]))
	AC_CHECK_LIB(GLU, gluLookAt,,AC_MSG_ERROR([cannot find GLU]))
	
	AC_DEFINE(OPENGL_RENDER, [1], [enable OpenGL rendering of some examples])
fi

AC_MSG_CHECKING(whether OpenGL rendering is enabled)
AC_SUBST(OPENGL_RENDER, $enable_opengl_render)
AC_MSG_RESULT($enable_opengl_render)

# In case there are BLAS kernels that are used by the example applications
# we may specify which library to use. Note that this is not used for StarPU
# itself.

blas_lib=maybe
AC_ARG_ENABLE(blas-lib,
 [  --enable-blas-lib[=blaslibname]:
                      none [default]: no BLAS lib is used
                      atlas: use ATLAS library
                      goto: use GOTO library],
 [ 
     if   test "x$enableval" = "xatlas" ; then
        blas_lib=atlas
     elif test "x$enableval" = "xgoto" ; then
        blas_lib=goto
     elif test "x$enableval" = "xnone" ; then
        blas_lib=none
     elif test x$enableval = xno; then
	blas_lib=none
     else
        echo
        echo "Error!"
        echo "Unknown BLAS library"
        exit -1
     fi
 ])

if test x$blas_lib = xmaybe -o x$blas_lib = xgoto; then
AC_ARG_WITH(goto-dir, [AS_HELP_STRING([--with-goto-dir=<dir>], [specify GOTO lib location])],
	[
		blas_lib=goto
		gotodir=$withval
		AC_SUBST(GOTODIR, $gotodir)

		CPPFLAGS="${CPPFLAGS} -I$gotodir/ "
		LDFLAGS="${LDFLAGS} -L$gotodir/ "
	]
	)

if test x$blas_lib = xgoto; then
# test whether ATLAS is actually available
AC_CHECK_LIB(goto, sgemm_,,AC_MSG_ERROR([cannot find goto lib]))
AC_DEFINE(GOTO, [1], [use GOTO library])
fi

fi
     
if test x$blas_lib = xmaybe -o x$blas_lib = xatlas; then
AC_ARG_WITH(atlas-dir, [AS_HELP_STRING([--with-atlas-dir=<dir>], [specify ATLAS lib location])],
	[
		AC_MSG_CHECKING(ATLAS location)
		blas_lib=atlas
		atlasdir=$withval
		AC_MSG_RESULT($atlasdir)
		AC_SUBST(ATLASDIR, $atlasdir)

		CPPFLAGS="${CPPFLAGS} -I$atlasdir/include/ "
		LDFLAGS="${LDFLAGS} -L$atlasdir/lib/ "
	]
	)

if test x$blas_lib = xatlas; then
# test whether ATLAS is actually available
AC_CHECK_HEADER([cblas.h],,AC_MSG_ERROR([cannot find atlas headers]))
AC_CHECK_LIB(atlas, ATL_sgemm,,AC_MSG_ERROR([cannot find atlas lib]),)
AC_CHECK_LIB(cblas, cblas_sgemm,,AC_MSG_ERROR([cannot find atlas lib]),[-latlas])
AC_DEFINE(ATLAS, [1], [use ATLAS library])
fi

fi
 
if test x$blas_lib = xmaybe; then
     #perhaps it is possible to use some BLAS lib from the system
     use_system_blas=no
     AC_SEARCH_LIBS([sgemm_],[blas],use_system_blas=yes,,)
     if test x$use_system_blas = xyes; then
        AC_DEFINE(SYSTEM_BLAS, [1], [use refblas library])
	blas_lib=system
     else
	blas_lib=none
     fi
fi

AM_CONDITIONAL(ATLAS_BLAS_LIB, test x$blas_lib = xatlas)
AM_CONDITIONAL(GOTO_BLAS_LIB, test x$blas_lib = xgoto)
AM_CONDITIONAL(SYSTEM_BLAS_LIB, test x$blas_lib = xsystem)
AM_CONDITIONAL(NO_BLAS_LIB, test x$blas_lib = xnone)

AC_MSG_CHECKING(which BLAS lib should be used)
AC_MSG_RESULT($blas_lib)
AC_SUBST(BLAS_LIB,$blas_lib)

AC_CONFIG_HEADER(src/common/config.h include/starpu_config.h)

AC_OUTPUT([
	Makefile
	src/Makefile
	tools/Makefile
	libstarpu.pc
	examples/Makefile
	tests/Makefile
])
