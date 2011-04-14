# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2009, 2010, 2011  Université de Bordeaux 1
# Copyright (C) 2010, 2011  Centre National de la Recherche Scientifique
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

AC_INIT([StarPU],0.4, [starpu-bugs@lists.gforge.inria.fr], starpu)
AC_CONFIG_SRCDIR(include/starpu.h)
AC_CONFIG_AUX_DIR([build-aux])
AC_CANONICAL_SYSTEM
AM_INIT_AUTOMAKE([1.10 -Wall -Werror foreign])

m4_ifdef([AM_SILENT_RULES], [AM_SILENT_RULES([yes])])

AC_PREREQ(2.60)

AC_PROG_CC
AC_PROG_CPP
AC_PROG_SED
AC_PROG_LN_S
AC_PROG_F77

AC_LIBTOOL_WIN32_DLL
AC_PROG_LIBTOOL

AC_PROG_INSTALL
AC_PROG_MKDIR_P
AC_PROG_LN_S

AC_HEADER_STDC

AC_C_RESTRICT

AC_PATH_PROGS([STARPU_MS_LIB], [lib])
AC_ARG_VAR([STARPU_MS_LIB], [Path to Microsoft's Visual Studio `lib' tool])
AM_CONDITIONAL([STARPU_HAVE_MS_LIB], [test "x$STARPU_MS_LIB" != "x"])
case "$target" in
*-*-mingw*|*-*-cygwin*)
  starpu_windows=yes
  libext=a
  AC_DEFINE(STARPU_HAVE_WINDOWS, [], [Define this on windows.])
  ;;
esac
AM_CONDITIONAL([STARPU_HAVE_WINDOWS], [test "x$starpu_windows" = "xyes"])

# on Darwin, GCC targets i386 by default, so we don't have atomic ops
AC_CHECK_SIZEOF([void *])
if test x$SIZEOF_VOID_P = x4; then
	case "$target" in
	i386-*darwin*) CFLAGS+=" -march=i686 " ;;
	esac
fi



# This will be useful for program which use CUDA (and .cubin files) which need
# some path to the CUDA code at runtime.
AC_DEFINE_UNQUOTED(STARPU_BUILD_DIR, "$PWD", [location of StarPU build directory])
AC_SUBST(STARPU_BUILD_DIR, $PWD)
case "${srcdir}" in
/*)  AC_DEFINE_UNQUOTED(STARPU_SRC_DIR, "$(eval echo ${srcdir})", [location of StarPU sources])
     AC_SUBST(STARPU_SRC_DIR, "$(eval echo ${srcdir})") ;;
*)   AC_DEFINE_UNQUOTED(STARPU_SRC_DIR, "$(eval echo $PWD/${srcdir})", [location of StarPU sources])
     AC_SUBST(STARPU_SRC_DIR, "$(eval echo $PWD/${srcdir})") ;;
esac

AC_CHECK_LIB([pthread], [pthread_create])
AC_COMPILE_IFELSE(
  AC_LANG_PROGRAM([[
    #include <pthread.h>
  ]], [[ pthread_t t; pthread_create(&t, NULL, NULL, NULL); ]]),,
  AC_MSG_ERROR([pthread_create unavailable]))
AC_SEARCH_LIBS([sqrt],[m],,AC_MSG_ERROR([math library unavailable]))
AC_HAVE_LIBRARY([wsock32])
AC_CHECK_FUNCS([sysconf])

AC_CHECK_FUNC([pthread_spin_lock], have_pthread_spin_lock=yes, have_pthread_spin_lock=no)
if test x$have_pthread_spin_lock = xyes; then
	AC_DEFINE(HAVE_PTHREAD_SPIN_LOCK,[],[pthread_spin_lock is available])
fi

# yes, that's non portable, but it's still better than sched_setaffinity
AC_CHECK_FUNCS(pthread_setaffinity_np)

# There is no posix_memalign on Mac OS X, only memalign
AC_CHECK_FUNCS([posix_memalign], [AC_DEFINE([STARPU_HAVE_POSIX_MEMALIGN], [1], [Define to 1 if you have the `posix_memalign' function.])])
AC_CHECK_FUNCS([memalign], [AC_DEFINE([STARPU_HAVE_MEMALIGN], [1], [Define to 1 if you have the `memalign' function.])])

# Some systems don't have drand48
AC_CHECK_FUNC([drand48], have_drand48=yes, have_drand48=no)
# Maybe the user still does not want to use the provided drand48
AC_ARG_ENABLE(default-drand48, [AS_HELP_STRING([--disable-default-drand48],
				   [Do not use the default version of drand48])],
				   enable_default_drand48=$enableval, enable_default_drand48=yes)
if test x$have_drand48 = xyes -a x$enable_default_drand48 = xyes ; then
  AC_DEFINE([starpu_srand48(seed)],[srand48(seed)],[srand48 equivalent function])
  AC_DEFINE([starpu_drand48()],[drand48()],[drand48 equivalent function])
  AC_DEFINE([starpu_erand48(xsubi)],[erand48(xsubi)],[erand48 equivalent function])
else
  AC_DEFINE([starpu_srand48(seed)],[srand(seed)],[srand48 equivalent function])
  AC_DEFINE([starpu_drand48()],[((double)(rand()) / RAND_MAX)],[drand48 equivalent function])
  AC_DEFINE([starpu_erand48(xsubi)],[starpu_drand48()],[erand48 equivalent function])
fi

AC_CHECK_HEADERS([malloc.h], [AC_DEFINE([STARPU_HAVE_MALLOC_H], [1], [Define to 1 if you have the <malloc.h> header file.])])

# This defines HAVE_SYNC_VAL_COMPARE_AND_SWAP
STARPU_CHECK_SYNC_VAL_COMPARE_AND_SWAP

# This defines HAVE_SYNC_BOOL_COMPARE_AND_SWAP
STARPU_CHECK_SYNC_BOOL_COMPARE_AND_SWAP

# This defines HAVE_SYNC_FETCH_AND_ADD
STARPU_CHECK_SYNC_FETCH_AND_ADD

# This defines HAVE_SYNC_FETCH_AND_OR
STARPU_CHECK_SYNC_FETCH_AND_OR

# This defines HAVE_SYNC_LOCK_TEST_AND_SET
STARPU_CHECK_SYNC_LOCK_TEST_AND_SET

# This defines HAVE_SYNC_SYNCHRONIZE
STARPU_CHECK_SYNC_SYNCHRONIZE

CPPFLAGS="${CPPFLAGS} -D_GNU_SOURCE "

AC_SEARCH_LIBS([set_mempolicy],[numa],[enable_libnuma=yes],[enable_libnuma=no])
AC_MSG_CHECKING(whether libnuma is available)
AC_MSG_RESULT($enable_libnuma)
if test x$enable_libnuma = xyes; then
	AC_DEFINE(STARPU_HAVE_LIBNUMA,[],[libnuma is available])
fi

###############################################################################
#                                                                             #
#                                 CPUs settings                               #
#                                                                             #
###############################################################################

AC_MSG_CHECKING(maximum number of CPUs)
AC_ARG_ENABLE(nmaxcpus, [AS_HELP_STRING([--enable-nmaxcpus=<number>],
			[maximum number of CPUs])],
			nmaxcpus=$enableval, nmaxcpus=16)
AC_MSG_RESULT($nmaxcpus)
# TODO: rename to STARPU_MAXCPUS for coherency with CUDA/OpenCL?
AC_DEFINE_UNQUOTED(STARPU_NMAXCPUS, [$nmaxcpus], [Maximum number of CPUs supported])

AC_MSG_CHECKING(whether CPUs should be used)
AC_ARG_ENABLE(cpu, [AS_HELP_STRING([--disable-cpu],
			[do not use the CPU(s)])],
			enable_cpu=$enableval, enable_cpu=yes)
AC_MSG_RESULT($enable_cpu)
AC_SUBST(STARPU_USE_CPU, $enable_cpu)

AM_CONDITIONAL(STARPU_USE_CPU, test x$enable_cpu = xyes)

if test x$enable_cpu = xyes; then
	AC_DEFINE(STARPU_USE_CPU, [1], [CPU driver is activated])
fi

# How many parallel worker can we support ?
nmaxcombinedworkers=`expr 2 \* $nmaxcpus`
AC_DEFINE_UNQUOTED(STARPU_NMAX_COMBINEDWORKERS,
	[$nmaxcombinedworkers], [Maximum number of worker combinations])

###############################################################################
#                                                                             #
#                                 CUDA settings                               #
#                                                                             #
###############################################################################

AC_MSG_CHECKING(maximum number of CUDA devices)
AC_ARG_ENABLE(maxcudadev, [AS_HELP_STRING([--enable-maxcudadev=<number>],
			[maximum number of CUDA devices])],
			nmaxcudadev=$enableval, nmaxcudadev=4)
AC_MSG_RESULT($nmaxcudadev)
AC_DEFINE_UNQUOTED(STARPU_MAXCUDADEVS, [$nmaxcudadev],
		[maximum number of CUDA devices])

AC_ARG_ENABLE(cuda, [AS_HELP_STRING([--disable-cuda],
		[do not use CUDA device(s)])],, [enable_cuda=maybe])

#AC_MSG_CHECKING(whether CUDA is available)
AC_ARG_WITH(cuda-dir,
	[AS_HELP_STRING([--with-cuda-dir=<path>],
	[specify CUDA installation directory])],
	[
		cuda_dir="$withval"
		# in case this was not explicit yet
		enable_cuda=yes
	], cuda_dir=no)

AC_ARG_WITH(cuda-include-dir,
	[AS_HELP_STRING([--with-cuda-include-dir=<path>],
	[specify where CUDA headers are installed])],
	[
		cuda_include_dir="$withval"
		# in case this was not explicit yet
		enable_cuda=yes
	], [cuda_include_dir=no])

AC_ARG_WITH(cuda-lib-dir,
	[AS_HELP_STRING([--with-cuda-lib-dir=<path>],
	[specify where CUDA libraries are installed])],
	[
		cuda_lib_dir="$withval"
		# in case this was not explicit yet
		enable_cuda=yes
	], [cuda_lib_dir=no])

AC_DEFUN([STARPU_CHECK_CUDA],
[
    __cuda_dir=$1

    if test "$__cuda_dir" != "no" ; then
	AC_MSG_CHECKING(whether CUDA is available in $__cuda_dir)
    else
	AC_MSG_CHECKING(whether CUDA is available)
    fi
    AC_MSG_RESULT()

    SAVED_LDFLAGS="${LDFLAGS}"

    if test "$__cuda_dir" != "no" ; then
	STARPU_CUDA_LDFLAGS="-L${__cuda_dir}/lib"
	LDFLAGS="${SAVED_LDFLAGS} -L${__cuda_dir}/lib"
    fi

    AC_HAVE_LIBRARY([cuda],[have_valid_cuda=yes],[have_valid_cuda=no])
    unset ac_cv_lib_cuda_main

    if test "$have_valid_cuda" = "no" ; then
        if test "$__cuda_dir" != "no" ; then
	    STARPU_CUDA_LDFLAGS="-L${__cuda_dir}/lib64"
	    LDFLAGS="${SAVED_LDFLAGS} -L${__cuda_dir}/lib64"
	    AC_HAVE_LIBRARY([cuda],[have_valid_cuda=yes],[have_valid_cuda=no])
            unset ac_cv_lib_cuda_main
        fi
    fi

    if test "$have_valid_cuda" = "no" ; then
        LDFLAGS="${SAVED_LDFLAGS}"
    fi
])

AC_DEFUN([STARPU_CHECK_CUDA_RUNTIME],
[
    __cuda_dir=$1
    __cuda_include_dir=$2
    __cuda_lib_dir=$3

    if test "$__cuda_dir" != "no" ; then
	AC_MSG_CHECKING(whether CUDA RT is available in $__cuda_dir)
    else
	AC_MSG_CHECKING(whether CUDA RT is available)
    fi
    AC_MSG_RESULT()

    if test "$__cuda_include_dir" = "no" -a "$__cuda_dir" != "no" ; then
        __cuda_include_dir="$__cuda_dir/include"
    fi
    if test "$__cuda_lib_dir" = "no" -a "$__cuda_dir" != "no" ; then
        __cuda_lib_dir="$__cuda_dir/lib"
    fi

    SAVED_CPPFLAGS="$CPPFLAGS"
    SAVED_LDFLAGS="${LDFLAGS}"
    SAVED_STARPU_CUDA_LDFLAGS=${STARPU_CUDA_LDFLAGS}

    if test "$__cuda_lib_dir" != "no" ; then
	STARPU_CUDA_LDFLAGS="${SAVED_STARPU_CUDA_LDFLAGS} -L$__cuda_lib_dir"
    fi
    if test "$__cuda_include_dir" != "no" ; then
        CPPFLAGS="${CPPFLAGS} -I$__cuda_include_dir"
    fi

    AC_CHECK_HEADER([cuda.h],[have_valid_cuda=yes],[have_valid_cuda=no])
    unset ac_cv_header_cuda_h

    if test "$have_valid_cuda" = "yes" ; then
	if test "$__cuda_lib_dir" != "no"; then
	    LDFLAGS="${SAVED_LDFLAGS} -L$__cuda_lib_dir"
        fi
	AC_HAVE_LIBRARY([cudart],[have_valid_cuda=yes],[have_valid_cuda=no])
        unset ac_cv_lib_cudart_main
        if test "$have_valid_cuda" = "no" ; then
            if test "$3" = "no" -a "$__cuda_dir" != "no" ; then
                __cuda_lib_dir="$__cuda_dir/lib64"
	        STARPU_CUDA_LDFLAGS="${SAVED_STARPU_CUDA_LDFLAGS} -L$__cuda_lib_dir"
	        LDFLAGS="${SAVED_LDFLAGS} -L$__cuda_lib_dir"
	        AC_HAVE_LIBRARY([cudart],[have_valid_cuda=yes],[have_valid_cuda=no])
                unset ac_cv_lib_cudart_main
            fi
        fi
    fi

    if test "$have_valid_cuda" = "yes" ; then
        STARPU_CUDA_LDFLAGS="$STARPU_CUDA_LDFLAGS -lcudart"
	# we also check that CUBLAS is available
	AC_HAVE_LIBRARY([cublas],[have_valid_cuda=yes],[have_valid_cuda=no])
        unset ac_cv_lib_cublas_main
        if test "$have_valid_cuda" = "yes" ; then
            STARPU_CUDA_LDFLAGS="$STARPU_CUDA_LDFLAGS -lcublas"
        fi
    fi

    CPPFLAGS="${SAVED_CPPFLAGS}"
    LDFLAGS="${SAVED_LDFLAGS}"

    if test "$have_valid_cuda" = "yes" -a "$__cuda_include_dir" != "no"; then
        STARPU_CUDA_CPPFLAGS="-I$__cuda_include_dir"
	NVCCFLAGS="${NVCCFLAGS} -I$__cuda_include_dir"
    fi
])

if test x$enable_cuda = xyes -o x$enable_cuda = xmaybe; then
    STARPU_CHECK_CUDA($cuda_dir)
    if test "$have_valid_cuda" = "no" ; then
        for f in "/usr/local/cuda" "/c/cuda" "/cygdrive/c/cuda" "/opt/cuda" ; do
            STARPU_CHECK_CUDA($f)
            if test "$have_valid_cuda" = "yes" ; then
                break
            fi
        done
    fi

    if test "$have_valid_cuda" = "yes" ; then
        STARPU_CHECK_CUDA_RUNTIME($cuda_dir, $cuda_include_dir, $cuda_lib_dir)
        if test "$have_valid_cuda" = "no" ; then
            for f in "/usr/local/cuda" "/c/cuda" "/cygdrive/c/cuda" "/opt/cuda" ; do
                STARPU_CHECK_CUDA_RUNTIME($f, "no", "no")
                if test "$have_valid_cuda" = "yes" ; then
                    break
                fi
            done
        fi
    fi

    # in case CUDA was explicitely required, but is not available, this is an error
    if test x$enable_cuda = xyes -a x$have_valid_cuda = no; then
	AC_MSG_ERROR([cannot find CUDA])
    fi
    # now we enable CUDA if and only if a proper setup is available
    enable_cuda=$have_valid_cuda
fi

AC_MSG_CHECKING(whether CUDA should be used)
AC_MSG_RESULT($enable_cuda)
AC_SUBST(STARPU_USE_CUDA, $enable_cuda)
AM_CONDITIONAL(STARPU_USE_CUDA, test x$enable_cuda = xyes)
if test x$enable_cuda = xyes; then
	AC_DEFINE(STARPU_USE_CUDA, [1], [CUDA support is activated])

	# On Darwin, the libstdc++ dependency is not automatically added by nvcc
#	case "$target" in
#		*-*darwin*) AC_HAVE_LIBRARY([stdc++], []) ;;
#		#*-*darwin*) AC_HAVE_LIBRARY([stdc++], [STARPU_CUDA_LDFLAGS="$STARPU_CUDA_LDFLAGS -lstdc++"]) ;;
#	esac
	STARPU_CUDA_LDFLAGS="$STARPU_CUDA_LDFLAGS -lcuda -lcufft"

        if test "$F77" = "gfortran" ; then
            STARPU_CUDA_LDFLAGS="$STARPU_CUDA_LDFLAGS -lgfortran"
        fi

	#in case this is a 64bit setup, we tell nvcc to use a -m64 flag
	AC_CHECK_SIZEOF([void *])
	if test x$SIZEOF_VOID_P = x8; then
		NVCCFLAGS="${NVCCFLAGS} -m64"
		AC_SUBST(NVCCFLAGS)
	fi

	AC_PATH_PROG([NVCC], [nvcc], [nvcc], [$cuda_dir/bin:$PATH:/usr/local/bin:/usr/bin:/bin], [AC_MSG_WARN(Cannot find nvcc.)])
fi

enable_magma=no
if test x$enable_cuda = xyes; then
	# Should we use the MAGMA library (instead or in addition to CUBLAS)
	magma_dir=/usr/local # default
	enable_magma=maybe

	AC_ARG_WITH(magma, [AS_HELP_STRING([--with-magma=<path>],
			[specify that MAGMA should be used and its installation directory])],
			[
				if test x$withval = xyes; then
					# No path was specified but MAGMA is explicitely enabled
					enable_magma=yes
				else
				if test x$withval = xno; then
					# MAGMA is explicitely disabled
					enable_magma=no
				else
					# MAGMA is enabled and the PATH is given in $withval
					enable_magma=yes
					magma_dir="$withval"
				fi
				fi
			], [])

	# Do we have a valid MAGMA setup ?
	if test x$enable_magma = xyes -o x$enable_magma = xmaybe; then
		SAVED_LDFLAGS="${LDFLAGS}"
		SAVED_CPPFLAGS="${CPPFLAGS}"
		if test -d "$magma_dir/lib/"; then
			LDFLAGS="${SAVED_LDFLAGS} -L$magma_dir/lib/ "
			have_valid_magma=yes
			AC_CHECK_LIB(magmablas, main,,[have_valid_magma=no])
			AC_CHECK_LIB(magma, magmablas_sgemm,,[have_valid_magma=no])
		fi

		if test -d "$magma_dir/include/"; then
			CPPFLAGS="${SAVED_CPPFLAGS} -I$magma_dir/include/ "
			AC_CHECK_HEADER([magmablas.h],,[have_valid_magma=no])
			#AC_CHECK_HEADER([magma.h],,[have_valid_magma=no])
		fi

		if test x$have_valid_magma = xno; then
			# If MAGMA was explicitely required, this is an error
			if test x$enable_magma = xyes; then
				AC_MSG_ERROR([cannot find MAGMA])
			fi

			# Restore old flags and don't use MAGMA
			LDFLAGS="${SAVED_LDFLAGS}"
			CPPFLAGS="${SAVED_CPPFLAGS}"
			enable_magma=no
		else
			enable_magma=yes
		fi

	else
		have_valid_magma=no
		enable_magma=no
	fi
fi

AC_MSG_CHECKING(whether MAGMA should be used)
AC_MSG_RESULT($enable_magma)
if test x$enable_magma = xyes; then
	AC_DEFINE(STARPU_HAVE_MAGMA, [1], [use MAGMA library])
fi

# cufftDoubleComplex may not be available on an old CUDA setup
AC_CHECK_TYPE(cufftDoubleComplex,
	[have_cufftdoublecomplex=yes],
	[have_cufftdoublecomplex=no], [#include <cufft.h>])
AM_CONDITIONAL(STARPU_HAVE_CUFFTDOUBLECOMPLEX, test x$have_cufftdoublecomplex = xyes)
if test x$have_cufftdoublecomplex = xyes; then
	AC_DEFINE(STARPU_HAVE_CUFFTDOUBLECOMPLEX, [1], [cufftDoubleComplex is available])
fi

# The CURAND library is only available since CUDA 3.2
have_curand=$enable_cuda
if test x$enable_cuda = xyes; then
        SAVED_LDFLAGS="${LDFLAGS}"
        LDFLAGS="${LDFLAGS} ${STARPU_CUDA_LDFLAGS}"
	AC_HAVE_LIBRARY([curand],[have_curand=yes],[have_curand=no])
        LDFLAGS="${SAVED_LDFLAGS}"
fi
AC_MSG_CHECKING(whether CURAND is available)
AC_MSG_RESULT($have_curand)
if test x$have_curand = xyes; then
    AC_DEFINE(STARPU_HAVE_CURAND,[1], [CURAND is available])
    STARPU_CUDA_LDFLAGS="$STARPU_CUDA_LDFLAGS -lcurand"
fi

# Peer transfers are only supported since CUDA 4.0
have_cuda_memcpy_peer=no
if test x$enable_cuda = xyes; then
    SAVED_LDFLAGS="${LDFLAGS}"
    LDFLAGS="${LDFLAGS} ${STARPU_CUDA_LDFLAGS}"
    AC_CHECK_FUNC([cudaMemcpyPeer], have_cuda_memcpy_peer=yes, have_cuda_memcpy_peer=no)
    LDFLAGS="${SAVED_LDFLAGS}"
fi
if test x$have_cuda_memcpy_peer = xyes; then
    AC_DEFINE(HAVE_CUDA_MEMCPY_PEER,[],[Peer transfers are supported in CUDA])
fi

if test x$enable_cuda = xyes; then
    STARPU_CUDA_LDFLAGS="$STARPU_CUDA_LDFLAGS -lstdc++"
    AC_SUBST(STARPU_CUDA_LDFLAGS)
    AC_SUBST(STARPU_CUDA_CPPFLAGS)
fi

###############################################################################
#                                                                             #
#                                 OpenCL settings                             #
#                                                                             #
###############################################################################

AC_MSG_CHECKING(maximum number of OpenCL devices)
AC_ARG_ENABLE(maxopencldev, [AS_HELP_STRING([--enable-maxopencldev=<number>],
			[maximum number of OPENCL devices])],
			nmaxopencldev=$enableval, nmaxopencldev=8)
AC_MSG_RESULT($nmaxopencldev)
AC_DEFINE_UNQUOTED(STARPU_MAXOPENCLDEVS, [$nmaxopencldev],
		[maximum number of OPENCL devices])
AC_ARG_ENABLE(opencl, [AS_HELP_STRING([--disable-opencl],
		[do not use OpenCL device(s)])],, [enable_opencl=maybe])

AC_DEFUN([STARPU_CHECK_OPENCL],
[
    __opencl_dir=$1
    __opencl_include_dir=$2
    __opencl_lib_dir=$3

    if test "$__opencl_dir" != "no" ; then
	AC_MSG_CHECKING(whether OpenCL is available in $__opencl_dir)
    else
	AC_MSG_CHECKING(whether OpenCL is available)
    fi
    AC_MSG_RESULT()

    if test "$__opencl_include_dir" = "no" -a "$__opencl_dir" != "no" ; then
        __opencl_include_dir="$__opencl_dir/include"
    fi
    if test "$__opencl_lib_dir" = "no" -a "$__opencl_dir" != "no" ; then
        __opencl_lib_dir="$__opencl_dir/lib"
    fi

    SAVED_CPPFLAGS="$CPPFLAGS"
    SAVED_LDFLAGS="${LDFLAGS}"

    if test "$__opencl_include_dir" != "no" ; then
        CPPFLAGS="${CPPFLAGS} -I$__opencl_include_dir"
    fi
    AC_CHECK_HEADER([CL/cl.h],[have_valid_opencl=yes],[have_valid_opencl=no])
    unset ac_cv_header_CL_cl_h

    if test "$have_valid_opencl" = "yes" ; then
	if test "$__opencl_lib_dir" != "no"; then
	    LDFLAGS="${SAVED_LDFLAGS} -L$__opencl_lib_dir"
        fi
	AC_HAVE_LIBRARY([OpenCL],[have_valid_opencl=yes],[have_valid_opencl=no])
        unset ac_cv_lib_OpenCL_main
        if test "$have_valid_opencl" = "no" ; then
            if test "$3" = "no" -a "$__opencl_dir" != "no" ; then
                __opencl_lib_dir="$__opencl_dir/lib64"
	        LDFLAGS="${SAVED_LDFLAGS} -L$__opencl_lib_dir"
	        AC_HAVE_LIBRARY([OpenCL],[have_valid_opencl=yes],[have_valid_opencl=no])
                unset ac_cv_lib_OpenCL_main
            fi
        fi
    fi

    CPPFLAGS="${SAVED_CPPFLAGS}"
    LDFLAGS="${SAVED_LDFLAGS}"

    if test "$have_valid_opencl" = "yes" -a "$__opencl_include_dir" != "no"; then
        STARPU_OPENCL_CPPFLAGS="-I$__opencl_include_dir"
    fi
    if test "$have_valid_opencl" = "yes" ; then
        if test "$__opencl_lib_dir" != "no"; then
            STARPU_OPENCL_LDFLAGS="-L$__opencl_lib_dir"
        fi
        STARPU_OPENCL_LDFLAGS="${STARPU_OPENCL_LDFLAGS} -lOpenCL"
    fi

])

#AC_MSG_CHECKING(whether OpenCL is available)
AC_ARG_WITH(opencl-dir,
	[AS_HELP_STRING([--with-opencl-dir=<path>],
	[specify OpenCL installation directory])],
	[
		opencl_dir="$withval"
		# in case this was not explicit yet
		enable_opencl=yes
	], opencl_dir=no)

AC_ARG_WITH(opencl-include-dir,
	[AS_HELP_STRING([--with-opencl-include-dir=<path>],
	[specify where OpenCL headers are installed])],
	[
		opencl_include_dir="$withval"
		# in case this was not explicit yet
		enable_opencl=yes
	], [opencl_include_dir=no])

AC_ARG_WITH(opencl-lib-dir,
	[AS_HELP_STRING([--with-opencl-lib-dir=<path>],
	[specify where OpenCL libraries are installed])],
	[
		opencl_lib_dir="$withval"
		# in case this was not explicit yet
		enable_opencl=yes
	], [opencl_lib_dir=no])

if test x$enable_opencl = xyes -o x$enable_opencl = xmaybe; then
    	STARPU_CHECK_OPENCL($opencl_dir, $opencl_include_dir, $opencl_lib_dir)
        if test "$have_valid_opencl" = "no" ; then
            for f in "/usr/local/cuda" "/c/cuda" "/cygdrive/c/cuda" "/opt/cuda" ; do
    	        STARPU_CHECK_OPENCL($f, "no", "no")
                if test "$have_valid_opencl" = "yes" ; then
                    break
                fi
            done
        fi

	# in case OpenCL was explicitely required, but is not available, this is an error
	if test x$enable_opencl = xyes -a x$have_valid_opencl = no; then
	    AC_MSG_ERROR([cannot find OpenCL])
	fi

	# now we enable OpenCL if and only if a proper setup is available
	enable_opencl=$have_valid_opencl
fi

AC_MSG_CHECKING(whether OpenCL should be used)
AC_MSG_RESULT($enable_opencl)
AC_SUBST(STARPU_USE_OPENCL, $enable_opencl)
AM_CONDITIONAL(STARPU_USE_OPENCL, test x$enable_opencl = xyes)
if test x$enable_opencl = xyes ; then
	AC_DEFINE(STARPU_USE_OPENCL, [1], [OpenCL support is activated])
	STARPU_OPENCL_CPPFLAGS="${STARPU_OPENCL_CPPFLAGS} -DSTARPU_OPENCL_DATADIR=${datarootdir}/starpu/opencl"
        AC_SUBST(STARPU_OPENCL_DATAdir, "$(eval echo ${datarootdir}/starpu/opencl/examples)")
        AC_SUBST(STARPU_OPENCL_CPPFLAGS)
        AC_SUBST(STARPU_OPENCL_LDFLAGS)
fi

###############################################################################
#                                                                             #
#                                 Cell settings                               #
#                                                                             #
###############################################################################

#TODO fix the default dir
AC_ARG_ENABLE(gordon, [AS_HELP_STRING([--enable-gordon],
			[use Cell SPUs])],, enable_gordon=maybe)

if test x$enable_gordon = xyes -o x$enable_gordon = xmaybe; then

	AC_ARG_WITH(gordon-dir, [AS_HELP_STRING([--with-gordon-dir=<path>],
			[specify Gordon installation directory (default is /usr/local/)])],
			[
				gordon_dir="$withval"
				enable_gordon=yes
			], gordon_dir=/usr/local/)

	# do we have a valid Gordon setup ?
	have_valid_gordon=yes

	# can we use dynamic code loading facilities ?
	AC_CHECK_LIB(elf, elf_memory,, [have_valid_gordon=no])

	AC_CHECK_LIB(spe2, spe_context_create,,[have_valid_gordon=no])
	AC_CHECK_FUNC(spe_in_mbox_write, [], [have_valid_gordon=no])

	PKG_PROG_PKG_CONFIG
	if test -d "$gordon_dir"; then
		PKG_CONFIG_PATH="${PKG_CONFIG_PATH}:$gordon_dir"
	fi
	AC_SUBST(PKG_CONFIG_PATH)
	PKG_CHECK_MODULES([GORDON], [libgordon], [], have_valid_gordon=no)

	CPPFLAGS="${CPPFLAGS} ${GORDON_CFLAGS}"
	LIBS="${LIBS} ${GORDON_LIBS}"

	# AC_CHECK_FUNC(gordon_init, [gordon], [have_valid_gordon=no])

	# in case Gordon was explicitely required, but is not available, this is an error
	if test x$enable_gordon = xyes -a x$have_valid_gordon = no; then
		AC_MSG_ERROR([cannot find Gordon])
	fi


	# now we enable Gordon if and only if a proper setup is available
	enable_gordon=$have_valid_gordon
fi

AC_MSG_CHECKING(whether GORDON should be used)
AC_MSG_RESULT($enable_gordon)
AC_SUBST(STARPU_USE_GORDON, $enable_gordon)
AM_CONDITIONAL(STARPU_USE_GORDON, test x$enable_gordon = xyes)

if test x$enable_gordon = xyes; then
	AC_DEFINE(STARPU_USE_GORDON, [1], [Cell support is enabled])
	GORDON_REQUIRES=gordon
fi
AC_SUBST(GORDON_REQUIRES)

AC_MSG_CHECKING(whether blocking drivers should be disabled)
AC_ARG_ENABLE(blocking-drivers, [AS_HELP_STRING([--disable-blocking-drivers], [disable blocking drivers])],
				enable_blocking=$enableval, enable_blocking=$enable_gordon)
AC_MSG_RESULT($enable_blocking)

if test x$enable_blocking = xno; then
	AC_DEFINE(STARPU_NON_BLOCKING_DRIVERS, [1], [drivers must progress])
fi


###############################################################################
#                                                                             #
#                   Debug and Performance analysis tools                      #
#                                                                             #
###############################################################################

AC_MSG_CHECKING(whether debug mode should be enabled)
AC_ARG_ENABLE(debug, [AS_HELP_STRING([--enable-debug], [enable debug mode])],
			enable_debug=$enableval, enable_debug=no)
AC_MSG_RESULT($enable_debug)

if test x$enable_debug = xyes; then
	CFLAGS=" -O0 "
else
	CFLAGS=" -O3 "
fi
CFLAGS+=" -gdwarf-2 -g3 "

AC_MSG_CHECKING(whether extra checks should be performed)
AC_ARG_ENABLE(fast, [AS_HELP_STRING([--enable-fast],
			[do not enforce assertions])],
			enable_fast=$enableval, enable_fast=no)
AC_MSG_RESULT($enable_fast)
if test x$enable_fast = xyes; then
	AC_DEFINE(STARPU_NO_ASSERT, [1], [disable assertions])
fi



AC_MSG_CHECKING(whether debug messages should be displayed)
AC_ARG_ENABLE(verbose, [AS_HELP_STRING([--enable-verbose],
			[display verbose debug messages])],
			enable_verbose=$enableval, enable_verbose=no)
AC_MSG_RESULT($enable_verbose)
if test x$enable_verbose = xyes; then
	AC_DEFINE(STARPU_VERBOSE, [1], [display verbose debug messages])
fi


AC_MSG_CHECKING(whether coverage testing should be enabled)
AC_ARG_ENABLE(coverage, [AS_HELP_STRING([--enable-coverage],
			[enable coverage checking])],
			enable_coverage=$enableval, enable_coverage=no)
AC_MSG_RESULT($enable_coverage)
AC_SUBST(COVERAGE, $enable_coverage)
AM_CONDITIONAL(STARPU_COVERAGE_ENABLED, [test "x$enable_coverage" = "xyes"])
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
AC_SUBST(STARPU_USE_FXT, $use_fxt)
AM_CONDITIONAL(STARPU_USE_FXT, test x$use_fxt = xyes)

if test x$use_fxt = xyes; then
	AC_DEFINE(STARPU_USE_FXT, [1], [enable FxT traces])
	AC_DEFINE(CONFIG_FUT, [1], [enable FUT traces])

	AC_SUBST(USE_FXTDIR_FROM_USER,$use_fxt_from_system)
	if test x$use_fxt_from_system = xno; then
		CPPFLAGS="${CPPFLAGS} -I$fxtdir/include/ "
		LDFLAGS="${LDFLAGS} -L$fxtdir/lib/ "
	fi

	# if we use monotonic clocks, FxT uses -lrt
	AC_CHECK_LIB(rt, clock_gettime,,AC_MSG_ERROR([cannot find clock_gettime]))

	AC_CHECK_LIB(fxt, fut_setup,,AC_MSG_ERROR([cannot find fxt lib]))
	AC_CHECK_HEADER([fxt/fxt.h],,AC_MSG_ERROR([cannot find headers for fxt]))
	AC_CHECK_HEADER([fxt/fut.h],,AC_MSG_ERROR([cannot find headers for fxt]))
fi

AC_MSG_CHECKING(whether performance debugging should be enabled)
AC_ARG_ENABLE(perf-debug, [AS_HELP_STRING([--enable-perf-debug],
			[enable performance debugging])],
			enable_perf_debug=$enableval, enable_perf_debug=no)
AC_MSG_RESULT($enable_perf_debug)
AC_SUBST(STARPU_PERF_DEBUG, $enable_perf_debug)
if test x$enable_perf_debuf = xyes; then
	AC_DEFINE(STARPU_PERF_DEBUG, [1], [enable performance debug])
	CPPFLAGS="${CPPFLAGS} -pg "
	LDFLAGS="${LDFLAGS} -pg "
fi

AC_MSG_CHECKING(whether performance model debugging should be enabled)
AC_ARG_ENABLE(model-debug, [AS_HELP_STRING([--enable-model-debug],
			[enable performance model debugging])],
			enable_model_debug=$enableval, enable_model_debug=no)
AC_MSG_RESULT($enable_model_debug)
if  test x$enable_model_debug = xyes; then
	AC_DEFINE(STARPU_MODEL_DEBUG, [1], [enable performance model debug])
fi

AC_MSG_CHECKING(whether statistics should be generated)
AC_ARG_ENABLE(stats, [AS_HELP_STRING([--enable-stats],
			[enable statistics])],
			enable_stats=$enableval, enable_stats=no)
AC_MSG_RESULT($enable_stats)
AC_SUBST(STATS, $enable_stats)
AC_SUBST(STARPU_DATA_STATS, $enable_stats)

if test x$enable_stats = xyes; then
        AC_DEFINE(STARPU_DATA_STATS, [1], [enable statistics])
fi

AC_CHECK_HEADERS([glpk.h])
AC_HAVE_LIBRARY([glpk])

###############################################################################
#                                                                             #
#                  Miscellaneous options for StarPU                           #
#                                                                             #
###############################################################################

AC_MSG_CHECKING(how many buffers can be manipulated per task)
AC_ARG_ENABLE(maxbuffers, [AS_HELP_STRING([--enable-maxbuffers=<nbuffers>],
			[maximum number of buffers per task])],
			nmaxbuffers=$enableval, nmaxbuffers=8)
AC_MSG_RESULT($nmaxbuffers)
AC_DEFINE_UNQUOTED(STARPU_NMAXBUFS, [$nmaxbuffers],
		[how many buffers can be manipulated per task])

# We have one memory node shared by all CPU workers, one node per GPU, and
# currently the Cell driver is using the same memory node as the CPU.
maxnodes=1
if test x$enable_cuda = xyes ; then
	# we could have used nmaxcudadev + 1, but this would certainly give an
	# odd number.
	maxnodes=`expr $maxnodes + $nmaxcudadev`
fi
if test x$enable_opencl = xyes ; then
	# we could have used nmaxcudadev + 1, but this would certainly give an
	# odd number.
	maxnodes=`expr $maxnodes + $nmaxopencldev`
fi
# todo: set maxnodes to the next power of 2 greater than maxnodes

AC_MSG_CHECKING(maximum number of memory nodes)
AC_MSG_RESULT($maxnodes)
AC_DEFINE_UNQUOTED(STARPU_MAXNODES, [$maxnodes],
		[maximum number of memory nodes])


AC_MSG_CHECKING(whether allocation cache should be used)
AC_ARG_ENABLE(allocation-cache, [AS_HELP_STRING([--enable-allocation-cache],
			[enable data allocation cache])],
			enable_allocation_cache=$enableval, enable_allocation_cache=no)
AC_MSG_RESULT($enable_allocation_cache)
if test x$enable_allocation_cache = xyes; then
	AC_DEFINE(STARPU_USE_ALLOCATION_CACHE, [1], [enable data allocation cache])
fi

AC_ARG_WITH(perf-model-dir, [AS_HELP_STRING([--with-perf-model-dir=<dir>], [specify where performance models shoulds be stored])],
	[
		if test x$withval = xno; then
			AC_MSG_ERROR(--without-perf-model-dir is not a valid option)
		fi

		perf_model_dir="$withval"
		have_explicit_perf_model_dir=yes
		AC_DEFINE_UNQUOTED(STARPU_PERF_MODEL_DIR, ["$perf_model_dir"], [performance models location])
	], [
		# by default, we put the performance models in
		# $HOME/.starpu/sampling/
		have_explicit_perf_model_dir=no
		perf_model_dir="\$HOME/.starpu/sampling/"
	]
	)
AC_MSG_CHECKING(using explicit performance model location)
AC_MSG_RESULT($have_explicit_perf_model_dir)

AC_MSG_CHECKING(performance models location)
AC_MSG_RESULT($perf_model_dir)

# On many multicore CPUs, clock cycles are not synchronized
AC_CHECK_LIB([rt], [clock_gettime])
AC_CHECK_FUNCS([clock_gettime])

# Compute the maximum number of workers (we round it to 16 for alignment
# purposes).
nmaxworkers=`expr 16 \* \( \( $nmaxcpus + $nmaxcudadev + $nmaxopencldev + 15 \) / 16 \) `
AC_MSG_CHECKING(Maximum number of workers)
AC_MSG_RESULT($nmaxworkers)
AC_DEFINE_UNQUOTED(STARPU_NMAXWORKERS, [$nmaxworkers], [Maximum number of workers])

###############################################################################
#                                                                             #
#                                    MPI                                      #
#                                                                             #
###############################################################################

AC_ARG_WITH(mpicc, [AS_HELP_STRING([--with-mpicc[=<path to mpicc>]],
			[Path of the mpicc compiler])],
	[
		if test x$withval = xyes; then
			AC_MSG_ERROR(--with-mpicc must be given a pathname)
		else
			mpicc_path=$withval
		fi
	],
	[
		# nothing was specified: default value is used
		AC_PATH_PROG(mpicc_path, mpicc, [no], [])
	])

# We test if the MPICC compiler exists
if test ! -x $mpicc_path; then
	#MPICC does not exists or is not executable
	AC_MSG_RESULT(The mpicc compiler is not valid)
	use_mpi=no
else
	use_mpi=yes
fi

AC_MSG_CHECKING(mpicc path)
AC_MSG_RESULT($mpicc_path)
AC_SUBST(MPICC, $mpicc_path)
if test x$use_mpi = xyes; then
	cc_or_mpicc=$mpicc_path
else
	cc_or_mpicc=$CC
fi
AC_SUBST(CC_OR_MPICC, $cc_or_mpicc)

# If the user specifically asks for it, or if we are in a developer checkout, we enable mpi check
AC_ARG_ENABLE(mpi-check, AC_HELP_STRING([--enable-mpi-check], [Enable execution of MPI testcases]))
running_mpi_check=0
if test -d "$srcdir/.svn" ; then
    running_mpi_check=yes
fi
if test x$enable_mpi_check = xyes ; then
    running_mpi_check=yes
fi
if test x$enable_mpi_check = xno ; then
    running_mpi_check=no
fi

# Check if mpiexec is available
AC_ARG_WITH(mpiexec, [AS_HELP_STRING([--with-mpiexec[=<path to mpiexec>]],
			[Path of mpiexec])],
	[
		if test x$withval = xyes; then
			AC_MSG_ERROR(--with-mpiexec must be given a pathname)
		else
			mpiexec_path=$withval
		fi
	],
	[
		# nothing was specified: look in the path
		AC_PATH_PROG(mpiexec_path, mpiexec, [no], [$(dirname $mpicc_path):$PATH])
	])

AC_MSG_CHECKING(whether mpiexec is available)
AC_MSG_RESULT($mpiexec_path)

# We test if MPIEXEC exists
if test ! -x $mpiexec_path; then
	#MPIEXEC does not exists or is not executable
	AC_MSG_RESULT(The mpiexec script is not valid)
        running_mpi_check=no
        mpiexec_path=""
fi

AM_CONDITIONAL(STARPU_MPI_CHECK, test x$running_mpi_check = xyes)
if test x$use_mpi = xyes; then
        AC_MSG_CHECKING(whether MPI tests should be run)
        AC_MSG_RESULT($running_mpi_check)
	AC_SUBST(MPIEXEC,$mpiexec_path)
fi

AC_MSG_CHECKING(whether the StarPU MPI library should be generated)
AC_MSG_RESULT($use_mpi)
AC_SUBST(USE_MPI, $use_mpi)
AM_CONDITIONAL(USE_MPI, test x$use_mpi = xyes)
if test x$use_mpi = xyes; then
	AC_DEFINE(STARPU_USE_MPI,[],[whether the StarPU MPI library is available])
fi

###############################################################################
#                                                                             #
#                           Flags for C Compiler                              #
#                                                                             #
###############################################################################

# IS_SUPPORTED_CFLAG(flag)
# ------------------------
# Check if the CFLAGS `flag' is supported by the compiler
AC_DEFUN([IS_SUPPORTED_CFLAG],
[
	AC_REQUIRE([AC_PROG_CC])

	AC_MSG_CHECKING([whether compiler support $1])

	SAVED_CFLAGS="$CFLAGS"
	CFLAGS="$1 -we10006"

	AC_COMPILE_IFELSE(
		AC_LANG_PROGRAM(
			[[]],
			[[
				AC_LANG_SOURCE([
					const char *hello = "Hello World";
				])
			]]
		),
		[
			GLOBAL_AM_CFLAGS="$GLOBAL_AM_CFLAGS $1"
			AC_MSG_RESULT(yes)
		],
			AC_MSG_RESULT(no)
	)
	CFLAGS="$SAVED_CFLAGS"
])

IS_SUPPORTED_CFLAG(-W)
IS_SUPPORTED_CFLAG(-Wall)
IS_SUPPORTED_CFLAG(-Wextra)
AC_SUBST(GLOBAL_AM_CFLAGS)

###############################################################################
#                                                                             #
#                               GCC extensions                                #
#                                                                             #
###############################################################################

AC_ARG_ENABLE([gcc-extensions],
  [AS_HELP_STRING([--enable-gcc-extensions],
    [build the GCC plug-in that provides C language extensions (experimental)])],
  [enable_gcc_plugin="$enableval"],
  [enable_gcc_plugin="no"])

if test "x$enable_gcc_plugin" = "xyes"; then
   STARPU_GCC_PLUGIN_SUPPORT

   if test "x$ac_cv_have_gcc_plugins" != "xyes"; then
     AC_MSG_ERROR([This compiler lacks GCC plug-in support.])
   fi

   build_gcc_plugin="yes"
else
   build_gcc_plugin="no"
fi

AM_CONDITIONAL([BUILD_GCC_PLUGIN], [test "x$build_gcc_plugin" = "xyes"])

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

	AC_DEFINE(STARPU_OPENGL_RENDER, [1], [enable OpenGL rendering of some examples])
fi

AC_MSG_CHECKING(whether OpenGL rendering is enabled)
AC_SUBST(STARPU_OPENGL_RENDER, $enable_opengl_render)
AC_MSG_RESULT($enable_opengl_render)

AC_PATH_XTRA
if test "x$x_includes" != "xNONE"; then
	AC_DEFINE(STARPU_HAVE_X11, [1], [enable X11])
fi
AM_CONDITIONAL([HAVE_X11], [test "x$x_includes" != "xNONE"])

# In case there are BLAS kernels that are used by the example applications
# we may specify which library to use. Note that this is not used for StarPU
# itself.

blas_lib=maybe
AC_ARG_ENABLE(blas-lib,
 [  --enable-blas-lib[=blaslibname]:
                      none [default]: no BLAS lib is used
                      atlas: use ATLAS library
                      goto: use GotoBLAS library],
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
AC_ARG_WITH(goto-dir, [AS_HELP_STRING([--with-goto-dir=<dir>], [specify GotoBLAS lib location])],
	[
		blas_lib=goto
		gotodir=$withval
		AC_SUBST(GOTODIR, $gotodir)

		CPPFLAGS="${CPPFLAGS} -I$gotodir/ "
		LDFLAGS="${LDFLAGS} -L$gotodir/ "
	]
	)

if test x$blas_lib = xgoto; then
AC_CHECK_LIB(gfortran, main,,)
# Perhaps that GotoBLAS2 is available instead (so that we have libgotoblas2.{so,a})
AC_CHECK_LIB(goto2, sgemm_,, [havegoto2=no])
if test x$havegoto2 = xno; then
AC_CHECK_LIB(goto, sgemm_,,AC_MSG_ERROR([cannot find goto lib]))
fi
AC_DEFINE(STARPU_GOTO, [1], [use STARPU_GOTO library])
fi

fi

if test x$blas_lib = xmaybe -o x$blas_lib = xatlas; then
AC_ARG_WITH(atlas-dir, [AS_HELP_STRING([--with-atlas-dir=<dir>], [specify ATLAS lib location])],
	[
		AC_MSG_CHECKING(STARPU_ATLAS location)
		blas_lib=atlas
		atlasdir=$withval
		AC_MSG_RESULT($atlasdir)
		AC_SUBST(ATLASDIR, $atlasdir)

		CPPFLAGS="${CPPFLAGS} -I$atlasdir/include/ "
		LDFLAGS="${LDFLAGS} -L$atlasdir/lib/ "
	]
	)

if test x$blas_lib = xatlas; then
# test whether STARPU_ATLAS is actually available
AC_CHECK_HEADER([cblas.h],,AC_MSG_ERROR([cannot find atlas headers]))
AC_CHECK_LIB(atlas, ATL_sgemm,,AC_MSG_ERROR([cannot find atlas lib]),)
AC_CHECK_LIB(cblas, cblas_sgemm,,AC_MSG_ERROR([cannot find atlas lib]),[-latlas])
AC_DEFINE(STARPU_ATLAS, [1], [use STARPU_ATLAS library])
fi

fi

if test x$blas_lib = xmaybe; then
	# Should we use MKL ?
	AC_ARG_WITH(mkl-cflags, [AS_HELP_STRING([--with-mkl-cflags], [specify MKL compilation flags])],
		[
			CPPFLAGS="${CPPFLAGS} $withval"
			blas_lib=mkl
		])

	AC_ARG_WITH(mkl-ldflags, [AS_HELP_STRING([--with-mkl-ldflags], [specify MKL linking flags])],
		[
			LDFLAGS="${LDFLAGS} $withval"
			blas_lib=mkl
		])
	if test x$blas_lib=xmkl; then
	        AC_DEFINE(STARPU_MKL, [1], [use MKL library])
	fi
fi

if test x$blas_lib = xmaybe; then
     #perhaps it is possible to use some BLAS lib from the system
     use_system_blas=no
     AC_SEARCH_LIBS([sgemm_],[blas],use_system_blas=yes,,)
     if test x$use_system_blas = xyes; then
        AC_DEFINE(STARPU_SYSTEM_BLAS, [1], [use refblas library])
	blas_lib=system
     else
	blas_lib=none
     fi
fi

AM_CONDITIONAL(ATLAS_BLAS_LIB, test x$blas_lib = xatlas)
AM_CONDITIONAL(GOTO_BLAS_LIB, test x$blas_lib = xgoto)
AM_CONDITIONAL(MKL_BLAS_LIB, test x$blas_lib = xmkl)
AM_CONDITIONAL(SYSTEM_BLAS_LIB, test x$blas_lib = xsystem)
AM_CONDITIONAL(NO_BLAS_LIB, test x$blas_lib = xnone)

AC_MSG_CHECKING(which BLAS lib should be used)
AC_MSG_RESULT($blas_lib)
AC_SUBST(BLAS_LIB,$blas_lib)

have_fftw=no
have_fftwf=no
have_fftwl=no

PKG_CHECK_MODULES([FFTW],  [fftw3],  [
  AC_DEFINE([STARPU_HAVE_FFTW], [1], [Define to 1 if you have the libfftw3 library.])
  AC_SUBST([STARPU_HAVE_FFTW], [1])
  have_fftw=yes
], [:])
AM_CONDITIONAL(STARPU_HAVE_FFTW, [test x$have_fftw = xyes])

PKG_CHECK_MODULES([FFTWF], [fftw3f], [
  AC_DEFINE([STARPU_HAVE_FFTWF], [1], [Define to 1 if you have the libfftw3f library.])
  AC_SUBST([STARPU_HAVE_FFTWF], [1])
  have_fftwf=yes
], [:])
AM_CONDITIONAL(STARPU_HAVE_FFTWF, [test x$have_fftwf = xyes])

PKG_CHECK_MODULES([FFTWL], [fftw3l], [
  AC_DEFINE([STARPU_HAVE_FFTWL], [1], [Define to 1 if you have the libfftw3l library.])
  AC_SUBST([HAVE_FFTWFL], [1])
  have_fftwl=yes
], [:])
AM_CONDITIONAL(STARPU_HAVE_FFTWL, [test x$have_fftwl = xyes])

AC_ARG_WITH([hwloc], [AS_HELP_STRING([--without-hwloc], [Disable hwloc (enabled by default)])])
SAVED_LDFLAGS="${LDFLAGS}"
SAVED_CPPFLAGS="${CPPFLAGS}"
AS_IF([test "x$with_hwloc" != "xno"], [
	PKG_CHECK_MODULES([HWLOC],[hwloc], [
		have_valid_hwloc=yes
		have_pkgconfig_hwloc=yes], [
		have_valid_hwloc=no
		have_pkgconfig_hwloc=no])
	AS_IF([test "$have_valid_hwloc" = "no"], [
		hwloc_dir="$withval"
		CPPFLAGS="${SAVED_CPPFLAGS} -I$hwloc_dir/include"
		AC_CHECK_HEADER([hwloc.h],[have_valid_hwloc=yes],[have_valid_hwloc=no])
		LDFLAGS="${SAVED_LDFLAGS} -L$hwloc_dir/lib"
		AC_HAVE_LIBRARY([hwloc],[have_valid_hwloc=yes],[have_valid_hwloc=no])
		])
    ])
AS_IF([test "$have_valid_hwloc" = "yes"], [
	AC_DEFINE([STARPU_HAVE_HWLOC], [1], [Define to 1 if you have the hwloc library.])
	HWLOC_REQUIRES=hwloc
	AC_SUBST([STARPU_HAVE_HWLOC], [1])
	AS_IF([test "$have_pkgconfig_hwloc" = "no"], [
		HWLOC_CFLAGS="-I$hwloc_dir/include"
		HWLOC_LIBS="-L$hwloc_dir/lib -lhwloc"])
	])
LDFLAGS="${SAVED_LDFLAGS}"
CPPFLAGS="${SAVED_CPPFLAGS}"
AC_MSG_CHECKING(whether hwloc should be used)
AC_MSG_RESULT($have_valid_hwloc)
AC_SUBST(HWLOC_REQUIRES)

# is the header file f77.h available ?
have_f77_h=yes
AC_CHECK_HEADER([f77.h],,[have_f77_h=no])
AC_MSG_CHECKING(whether header file f77.h is available)
AC_MSG_RESULT($have_f77_h)
AC_SUBST(STARPU_HAVE_F77_H, $have_f77_h)
AM_CONDITIONAL(STARPU_HAVE_F77_H, test x$have_f77_h = xyes)
if test x$have_f77_h = xyes; then
        AC_DEFINE([STARPU_HAVE_F77_H], [1], [Define to 1 if you have the <f77.h> header file.])
fi

# Do we want to run optional tests
AC_MSG_CHECKING(whether optional tests should be run)
AC_ARG_ENABLE(optional_tests, [AS_HELP_STRING([--optional-tests],
			[run optional tests])],
			want_optional_tests=$enableval, want_optional_tests=no)
AC_MSG_RESULT($want_optional_tests)
AM_CONDITIONAL([COND_OPT], [test "$want_optional_tests" = yes])

# File configuration
AC_CONFIG_COMMANDS([executable-scripts], [chmod +x ]tests/regression/regression.sh)
AC_CONFIG_FILES(tests/regression/regression.sh tests/regression/profiles tests/regression/profiles.build.only)
AC_CONFIG_HEADER(src/common/config.h include/starpu_config.h)

AC_CONFIG_HEADERS([gcc-plugin/src/starpu-gcc-config.h])

AC_OUTPUT([
	Makefile
	src/Makefile
	tools/Makefile
	gcc-plugin/Makefile
	gcc-plugin/src/Makefile
	gcc-plugin/tests/Makefile
	libstarpu.pc
	examples/Makefile
        examples/opt/Makefile
	examples/starpufft/Makefile
	examples/stencil/Makefile
	tests/Makefile
        tests/opt/Makefile
	doc/Makefile
	mpi/Makefile
])

