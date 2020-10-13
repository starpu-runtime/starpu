# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2009-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
# Copyright (C) 2011       Télécom-SudParis
# Copyright (C) 2013       Thibaut Lambert
# Copyright (C) 2017       Guillaume Beauchamp
# Copyright (C) 2018,2020  Federal University of Rio Grande do Sul (UFRGS)
# Copyright (C) 2018       Umeà University
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
AC_INIT([StarPU], [1.3.7], [starpu-devel@lists.gforge.inria.fr], [starpu], [http://gitlab.inria.fr/starpu/starpu])
AC_CONFIG_SRCDIR(include/starpu.h)
AC_CONFIG_AUX_DIR([build-aux])

# libtool doesn't actually properly manage a space in the workdir
case `pwd` in
  *[[\\\"\#\$\&\'\`$am_lf\ \	]]*)
    AC_MSG_ERROR([unsafe absolute working directory name]);;
esac

dnl Versioning.

STARPU_MAJOR_VERSION="`echo $PACKAGE_VERSION | cut -d . -f 1`"
STARPU_MINOR_VERSION="`echo $PACKAGE_VERSION | cut -d . -f 2`"
STARPU_RELEASE_VERSION="`echo $PACKAGE_VERSION | cut -d . -f 3`"
STARPU_RELEASE_VERSION="`echo $PACKAGE_VERSION | cut -d . -f 3| sed 's/rc.*//'`"
dnl we do not want the rcXX in the release version. we would like to use sed -r 's/[a-z]+.*//' to remove any string but the -r option is not portable
AC_SUBST([STARPU_MAJOR_VERSION])
AC_SUBST([STARPU_MINOR_VERSION])
AC_SUBST([STARPU_RELEASE_VERSION])
AC_SUBST([STARPU_EFFECTIVE_VERSION])
AC_DEFINE_UNQUOTED([STARPU_MAJOR_VERSION], [$STARPU_MAJOR_VERSION], [Major version number of StarPU.])
AC_DEFINE_UNQUOTED([STARPU_MINOR_VERSION], [$STARPU_MINOR_VERSION], [Minor version number of StarPU.])
AC_DEFINE_UNQUOTED([STARPU_RELEASE_VERSION], [$STARPU_RELEASE_VERSION], [Release version number of StarPU.])

. "$srcdir/STARPU-VERSION"
AC_SUBST([LIBSTARPU_INTERFACE_CURRENT])
AC_SUBST([LIBSTARPU_INTERFACE_REVISION])
AC_SUBST([LIBSTARPU_INTERFACE_AGE])
AC_SUBST([LIBSTARPUMPI_INTERFACE_CURRENT])
AC_SUBST([LIBSTARPUMPI_INTERFACE_REVISION])
AC_SUBST([LIBSTARPUMPI_INTERFACE_AGE])
AC_SUBST([LIBSTARPUFFT_INTERFACE_CURRENT])
AC_SUBST([LIBSTARPUFFT_INTERFACE_REVISION])
AC_SUBST([LIBSTARPUFFT_INTERFACE_AGE])
AC_SUBST([LIBSTARPURM_INTERFACE_CURRENT])
AC_SUBST([LIBSTARPURM_INTERFACE_REVISION])
AC_SUBST([LIBSTARPURM_INTERFACE_AGE])
AC_SUBST([LIBSOCL_INTERFACE_CURRENT])
AC_SUBST([LIBSOCL_INTERFACE_REVISION])
AC_SUBST([LIBSOCL_INTERFACE_AGE])

AC_CANONICAL_SYSTEM

AM_INIT_AUTOMAKE([1.11 -Wall -Wno-portability foreign silent-rules color-tests parallel-tests])

m4_ifdef([AM_SILENT_RULES],
  [AM_SILENT_RULES(yes)])

AC_PREREQ(2.64)

m4_ifdef([AM_PROG_AR], [AM_PROG_AR])
AC_PROG_CC
AM_PROG_CC_C_O
AC_PROG_CXX
AC_PROG_CPP
AC_PROG_SED
AC_PROG_LN_S
AC_PROG_F77
AC_PROG_FC
AC_PROG_GREP
AC_PROG_EGREP
AC_PROG_INSTALL
AC_PROG_MKDIR_P
AC_CHECK_PROGS(PROG_STAT,gstat stat)
AC_CHECK_PROGS(PROG_DATE,gdate date)

dnl locate pkg-config
PKG_PROG_PKG_CONFIG

if test x$enable_perf_debug = xyes; then
    enable_shared=no
fi
default_enable_mpi_check=maybe
default_enable_mpi=maybe

###############################################################################
#                                                                             #
#                                 Drivers                                     #
#                                                                             #
###############################################################################

AC_ARG_ENABLE(opencl-simulator, [AS_HELP_STRING([--enable-opencl-simulator],
				[Enable the use of an OpenCL simulator])],
				enable_opencl_simulator=$enableval, enable_opencl_simulator=no)
if test x$enable_opencl_simulator = xyes; then
	enable_simgrid=yes
	AC_DEFINE(STARPU_OPENCL_SIMULATOR, [1], [Define this to enable using an OpenCL simulator])
fi

AC_ARG_WITH(simgrid-dir,
	[AS_HELP_STRING([--with-simgrid-dir=<path>],
	[specify SimGrid installation directory])],
	[
		simgrid_dir="$withval"
		# in case this was not explicit yet
		enable_simgrid=yes
	], simgrid_dir=no)

AC_ARG_WITH(simgrid-include-dir,
	[AS_HELP_STRING([--with-simgrid-include-dir=<path>],
	[specify where SimGrid headers are installed])],
	[
		simgrid_include_dir="$withval"
		# in case this was not explicit yet
		enable_simgrid=yes
	], [simgrid_include_dir=no])

AC_ARG_WITH(simgrid-lib-dir,
	[AS_HELP_STRING([--with-simgrid-lib-dir=<path>],
	[specify where SimGrid libraries are installed])],
	[
		simgrid_lib_dir="$withval"
		# in case this was not explicit yet
		enable_simgrid=yes
	], [simgrid_lib_dir=no])

AC_ARG_ENABLE(simgrid, [AS_HELP_STRING([--enable-simgrid],
			[Enable simulating execution in simgrid])],
			enable_simgrid=$enableval, enable_simgrid=no)
if test x$enable_simgrid = xyes ; then
	PKG_CHECK_MODULES([SIMGRID], [simgrid], [], [:])

	if test "$simgrid_include_dir" != "no" ; then
	   	SIMGRID_CFLAGS="$SIMGRID_CFLAGS -I$simgrid_include_dir"
	fi
	if test "$simgrid_lib_dir" != "no" ; then
	   	SIMGRID_LIBS="$SIMGRID_LIBS -L$simgrid_lib_dir"
	fi
	if test "$simgrid_dir" != "no" ; then
	   	SIMGRID_CFLAGS="$SIMGRID_CFLAGS -I$simgrid_dir/include"
	   	SIMGRID_LIBS="$SIMGRID_LIBS -L$simgrid_dir/lib"
	fi
   	if test -n "$SIMGRID_CFLAGS" ; then
	   	CFLAGS="$SIMGRID_CFLAGS $CFLAGS"
	   	CXXFLAGS="$SIMGRID_CFLAGS $CXXFLAGS"
	   	NVCCFLAGS="$SIMGRID_CFLAGS $NVCCFLAGS"
	fi
	if test -n "$SIMGRID_LIBS" ; then
		LDFLAGS="$SIMGRID_LIBS $LDFLAGS"
	fi
	AC_HAVE_LIBRARY([simgrid], [],
		[
			AC_MSG_ERROR(Simgrid support needs simgrid installed)
		]
	)
	AC_CHECK_HEADERS([simgrid/msg.h], [AC_DEFINE([STARPU_HAVE_SIMGRID_MSG_H], [1], [Define to 1 if you have msg.h in simgrid/.])])
	AC_CHECK_HEADERS([msg/msg.h], [AC_DEFINE([STARPU_HAVE_MSG_MSG_H], [1], [Define to 1 if you have msg.h in msg/.])])
	AC_CHECK_HEADERS([simgrid/host.h], [AC_DEFINE([STARPU_HAVE_SIMGRID_HOST_H], [1], [Define to 1 if you have host.h in simgrid/.])])
	AC_CHECK_HEADERS([xbt/base.h], [AC_DEFINE([STARPU_HAVE_XBT_BASE_H], [1], [Define to 1 if you have base.h in xbt/.])])
	AC_CHECK_HEADERS([simgrid/version.h], [AC_DEFINE([STARPU_HAVE_SIMGRID_VERSION_H], [1], [Define to 1 if you have version.h in simgrid/.])], [], [[
			  #ifdef STARPU_HAVE_XBT_BASE_H
			  #include <xbt/base.h>
			  #endif
			  ]])
	AC_CHECK_HEADERS([simgrid/simdag.h], [AC_DEFINE([STARPU_HAVE_SIMGRID_SIMDAG_H], [1], [Define to 1 if you have simdag.h in simgrid/.])])
	AC_CHECK_HEADERS([xbt/synchro.h], [AC_DEFINE([STARPU_HAVE_XBT_SYNCHRO_H], [1], [Define to 1 if you have synchro.h in xbt/.])])
	AC_CHECK_HEADERS([xbt/config.h], [AC_DEFINE([STARPU_HAVE_XBT_CONFIG_H], [1], [Define to 1 if you have config.h in xbt/.])])
	AC_CHECK_HEADERS([simgrid/actor.h], [AC_DEFINE([STARPU_HAVE_SIMGRID_ACTOR_H], [1], [Define to 1 if you have actor.h in simgrid/.])])
	AC_CHECK_HEADERS([simgrid/engine.h], [AC_DEFINE([STARPU_HAVE_SIMGRID_ENGINE_H], [1], [Define to 1 if you have engine.h in simgrid/.])])
	AC_CHECK_HEADERS([simgrid/semaphore.h], [AC_DEFINE([STARPU_HAVE_SIMGRID_SEMAPHORE_H], [1], [Define to 1 if you have semaphore.h in simgrid/.])])
	AC_CHECK_HEADERS([simgrid/mutex.h], [AC_DEFINE([STARPU_HAVE_SIMGRID_MUTEX_H], [1], [Define to 1 if you have mutex.h in simgrid/.])])
	AC_CHECK_HEADERS([simgrid/cond.h], [AC_DEFINE([STARPU_HAVE_SIMGRID_COND_H], [1], [Define to 1 if you have cond.h in simgrid/.])])
	AC_CHECK_HEADERS([simgrid/barrier.h], [AC_DEFINE([STARPU_HAVE_SIMGRID_BARRIER_H], [1], [Define to 1 if you have barrier.h in simgrid/.])])
	AC_CHECK_HEADERS([simgrid/engine.h])
	AC_CHECK_HEADERS([simgrid/zone.h], [AC_DEFINE([STARPU_HAVE_SIMGRID_ZONE_H], [1], [Define to 1 if you have zone.h in simgrid/.])])
	AC_CHECK_TYPES([smx_actor_t], [AC_DEFINE([STARPU_HAVE_SMX_ACTOR_T], [1], [Define to 1 if you have the smx_actor_t type.])], [], [[#include <simgrid/simix.h>]])

	# Latest functions
	AC_CHECK_FUNCS([MSG_process_attach sg_actor_attach sg_actor_init sg_actor_on_exit MSG_zone_get_hosts sg_zone_get_hosts MSG_process_self_name MSG_process_userdata_init sg_actor_data])
	AC_CHECK_FUNCS([xbt_mutex_try_acquire smpi_process_set_user_data SMPI_thread_create sg_zone_get_by_name sg_link_name sg_link_bandwidth_set sg_host_route sg_host_self sg_host_list sg_host_speed simcall_process_create sg_config_continue_after_help])
	AC_CHECK_FUNCS([simgrid_init], [AC_DEFINE([STARPU_SIMGRID_HAVE_SIMGRID_INIT], [1], [Define to 1 if you have the `simgrid_init' function.])])
	AC_CHECK_FUNCS([xbt_barrier_init], [AC_DEFINE([STARPU_SIMGRID_HAVE_XBT_BARRIER_INIT], [1], [Define to 1 if you have the `xbt_barrier_init' function.])])
	AC_CHECK_FUNCS([sg_actor_sleep_for sg_actor_self sg_actor_ref sg_host_get_properties sg_host_send_to sg_host_sendto sg_cfg_set_int sg_actor_self_execute sg_actor_execute simgrid_get_clock])
	AC_CHECK_DECLS([smpi_process_set_user_data], [], [], [[#include <smpi/smpi.h>]])

	# Oldies for compatibility with older simgrid
	AC_CHECK_FUNCS([MSG_get_as_by_name MSG_zone_get_by_name MSG_environment_get_routing_root MSG_host_get_speed])

	AC_DEFINE(STARPU_SIMGRID, [1], [Define this to enable simgrid execution])
	# We won't bind or detect anything
	with_hwloc=no

        # disable mpi checks by default, they require static linking, we don't
        # want that by default
	default_enable_mpi_check=no

	# disable MPI support by default
	default_enable_mpi=no

	# Simgrid 3.12 & 3.13 need -std=c++11 to be able to build anything in C++...
	AC_LANG_PUSH([C++])
	if test x$enable_shared = xno ; then
		# When linking statically, libtool does not realize we need libstdc++ for simgrid_cpp.cpp
		SIMGRID_LIBS="$SIMGRID_LIBS -lstdc++"
		LIBS="$LIBS -lstdc++"
	fi

	case \ $CXXFLAGS\  in
	*\ -std=*\ *) ;;
	*)
		AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
				  #ifdef STARPU_HAVE_SIMGRID_MSG_H
				  #include <simgrid/msg.h>
				  #include <simgrid/host.h>
				  #else
				  #include <msg/msg.h>
				  #endif
				  ]])],,
				  CXXFLAGS="-std=c++11 $CXXFLAGS")
		;;
	esac
	AC_LANG_POP([C++])
	AC_ARG_ENABLE(simgrid-mc, [AS_HELP_STRING([--enable-simgrid-mc],
				[Enable using Model Checker of simgrid])],
				enable_simgrid_mc=$enableval, enable_simgrid_mc=no)
	if test x$enable_simgrid_mc = xyes ; then
		AC_DEFINE(STARPU_SIMGRID_MC, [1], [Define this to enable Model Checker in simgrid execution])
		AC_PATH_PROG([SIMGRID_MC], [simgrid-mc], [no], [$simgrid_dir/bin:$PATH])
		LDFLAGS="$LDFLAGS -Wl,-znorelro -Wl,-znoseparate-code"
	fi
fi
AM_CONDITIONAL(STARPU_SIMGRID_MC, test x$enable_simgrid_mc = xyes)
AM_CONDITIONAL(STARPU_SIMGRID, test x$enable_simgrid = xyes)
AC_SUBST(SIMGRID_CFLAGS)
AC_SUBST(SIMGRID_LIBS)
AC_MSG_CHECKING(whether SimGrid is enabled)
AC_MSG_RESULT($enable_simgrid)

AC_MSG_CHECKING(whether blocking drivers should be enabled)
AC_ARG_ENABLE(blocking-drivers, [AS_HELP_STRING([--enable-blocking-drivers], [enable blocking drivers])],
				enable_blocking=$enableval, enable_blocking=$enable_simgrid)
AC_MSG_RESULT($enable_blocking)

if test x$enable_blocking = xno ; then
	AC_DEFINE(STARPU_NON_BLOCKING_DRIVERS, [1], [drivers must progress])
fi

if test x$enable_blocking = xyes ; then
	AC_MSG_CHECKING(whether worker callbacks should be enabled)
	AC_ARG_ENABLE(worker-callbacks, [AS_HELP_STRING([--enable-worker-callbacks], [enable worker callbacks])],
				enable_worker_cb=$enableval, enable_worker_cb=no)
	AC_MSG_RESULT($enable_worker_cb)
else
	# worker sleep/wake-up callbacks only make sense if blocking drivers are enabled
	enable_worker_cb=no
fi

if test x$enable_worker_cb = xyes ; then
	AC_DEFINE(STARPU_WORKER_CALLBACKS, [1], [workers must call callbacks on sleep/wake-up])
fi

###############################################################################
#                                                                             #
#                                LIBTOOLS                                     #
#                                                                             #
###############################################################################

#c++11 detection
AX_CXX_COMPILE_STDCXX(11,noext,optional)

AC_SUBST([STARPU_HAVE_CXX11], $HAVE_CXX11)
AM_CONDITIONAL([STARPU_HAVE_CXX11], [test "$HAVE_CXX11" -eq 1])
if test $HAVE_CXX11 -eq 1; then
  AC_DEFINE(STARPU_HAVE_CXX11, [1], [compiler supports cxx11])
fi

LT_PREREQ([2.2])
LT_INIT([win32-dll])

AC_HEADER_STDC

AC_C_RESTRICT

# Check if bash is available
AC_PATH_PROG([REALBASH], [bash], , [/bin:$PATH])

# Record git version
AC_PATH_PROG(gitcommand, git)
if test "$gitcommand" = "" ; then
   if test -f $srcdir/STARPU-REVISION ; then
      cp $srcdir/STARPU-REVISION .
   else
      echo "unknown" > ./STARPU-REVISION
   fi
else
   git log -n 1 --pretty="%H" $srcdir > ./STARPU-REVISION
fi

AM_CONDITIONAL([STARPU_CROSS_COMPILING], [test "x$cross_compiling" = "xyes"])

###############################################################################
#                                                                             #
#                           MPI compilers                                     #
#                                                                             #
###############################################################################

#Check MPICC
if test x$enable_simgrid = xyes ; then
    DEFAULT_MPICC=smpicc
else
    DEFAULT_MPICC=mpicc
fi
AC_ARG_WITH(mpicc, [AS_HELP_STRING([--with-mpicc=<mpicc name or path to mpicc>], [Name or path of the mpicc compiler])], [DEFAULT_MPICC=$withval])
case $DEFAULT_MPICC in
	/*) mpicc_path="$DEFAULT_MPICC" ;;
	*)  AC_PATH_PROG(mpicc_path, $DEFAULT_MPICC, [no], [$simgrid_dir/bin:$PATH]) ;;
esac
# We test if the MPICC compiler exists
if test ! -x $mpicc_path; then
    AC_MSG_RESULT(The mpicc compiler '$mpicc_path' does not have the execute permission)
    mpicc_path=no
fi

AC_MSG_CHECKING(whether mpicc is available)
AC_MSG_RESULT($mpicc_path)
AC_SUBST(MPICC, $mpicc_path)

if test x$mpicc_path != xno ; then
    MPIPATH=$(dirname $mpicc_path):$PATH
else
    MPIPATH=$PATH
fi

#Check MPICXX/MPIC++
if test x$enable_simgrid = xyes ; then
    DEFAULT_MPICXX=smpicxx
else
    DEFAULT_MPICXX=mpicxx
fi
AC_ARG_WITH(mpicxx, [AS_HELP_STRING([--with-mpicxx=<mpicxx name or path to mpicxx>], [Name or path of the mpicxx/mpic++ compiler])], [DEFAULT_MPICXX=$withval])
case $DEFAULT_MPICXX in
	/*) mpicxx_path="$DEFAULT_MPICXX" ;;
	*)  AC_PATH_PROG(mpicxx_path, $DEFAULT_MPICXX, [no], [$MPIPATH]) ;;
esac

# try with mpic++ if mpicxx was not found
if test x$mpicxx_path = xno ; then
    DEFAULT_MPICXX=mpic++
    AC_PATH_PROG(mpicxx_path, $DEFAULT_MPICXX, [no], [$MPIPATH])
fi

# We test if the MPICXX/MPIC++ compiler exists
if test ! -x $mpicxx_path; then
    AC_MSG_RESULT(The mpicxx compiler '$mpicxx_path' does not have the execute permission)
    mpicxx_path=no
fi

AC_MSG_CHECKING(whether mpicxx is available)
AC_MSG_RESULT($mpicxx_path)
AC_SUBST(MPICXX, $mpicxx_path)

# Check if mpiexec is available
if test x$enable_simgrid = xyes ; then
    DEFAULT_MPIEXEC=smpirun
    AC_ARG_WITH(smpirun, [AS_HELP_STRING([--with-smpirun[=<name of smpirun or path to smpirun>]], [Name or path of the smpirun helper])], [DEFAULT_MPIEXEC=$withval])
else
    DEFAULT_MPIEXEC=mpiexec
    AC_ARG_WITH(mpiexec, [AS_HELP_STRING([--with-mpiexec=<name of mpiexec or path to mpiexec>], [Name or path of mpiexec])], [DEFAULT_MPIEXEC=$withval])
fi

case $DEFAULT_MPIEXEC in
    /*) mpiexec_path="$DEFAULT_MPIEXEC" ;;
    *)  AC_PATH_PROG(mpiexec_path, $DEFAULT_MPIEXEC, [no], [$MPIPATH])
esac
AC_MSG_CHECKING(whether mpiexec is available)
AC_MSG_RESULT($mpiexec_path)

# We test if MPIEXEC exists
if test ! -x $mpiexec_path; then
    AC_MSG_RESULT(The mpiexec script '$mpiexec_path' is not valid)
    default_enable_mpi_check=no
    mpiexec_path=""
fi
AC_SUBST(MPIEXEC,$mpiexec_path)

###############################################################################
#                                                                             #
#                                    MPI                                      #
#                                                                             #
###############################################################################

AC_ARG_ENABLE(mpi, [AS_HELP_STRING([--disable-mpi],
                              [Disable StarPU MPI library generation])],
            [enable_mpi=$enableval],
            [enable_mpi=$default_enable_mpi])

if test x$enable_mpi = xmaybe ; then
    if test -x "$mpicc_path"; then
	enable_mpi=yes
    else
	enable_mpi=no
    fi
fi

# in case MPI was explicitely required, but mpicc is not available, this is an error
if test x$enable_mpi = xyes -a ! -x "$mpicc_path"; then
   AC_MSG_ERROR([Compiler MPI '$mpicc_path' not valid])
fi

build_mpi_lib=$enable_mpi

###############################################################################
#                                                                             #
#                                NEW MADELEINE                                #
#                                                                             #
###############################################################################

AC_ARG_ENABLE(nmad, [AS_HELP_STRING([--enable-nmad],
		                    [Enable StarPU MPI library generation using the new madeleine backend])],
            [enable_nmad=$enableval],
            [enable_nmad=no])

build_nmad_lib=no
AC_SUBST(CC_OR_MPICC, $cc_or_mpicc)
#We can only build StarPU MPI Library if User wants it and MPI is available
if test x$enable_mpi = xyes -a x$enable_nmad = xyes ; then
    build_nmad_lib=yes
    build_mpi_lib=no
    PKG_CHECK_MODULES([NMAD],[nmad])
    AC_CHECK_DECL([piom_ltask_set_bound_thread_os_indexes], have_piom_ltask_set_bound_thread_os_indexes=yes, have_piom_ltask_set_bound_thread_os_indexes=no, [[#include <pioman.h>]])
    if test x$have_piom_ltask_set_bound_thread_os_indexes = xyes; then
      AC_DEFINE(HAVE_PIOM_LTASK_SET_BOUND_THREAD_OS_INDEXES, [1], [piom_ltask_set_bound_thread_os_indexes is availabe])
    fi
else
    build_nmad_lib=no
fi

###############################################################################
#                                                                             #
#                             MPI Master Slave                                #
#                                                                             #
###############################################################################

AC_ARG_ENABLE(mpi-master-slave, [AS_HELP_STRING([--enable-mpi-master-slave],
                              [Enable StarPU to run with the master-slave mode])],
              use_mpi_master_slave=$enableval,
              use_mpi_master_slave=no)
#We can only build MPI Master Slave if User wants it and MPI compiler are available
if test x$use_mpi_master_slave = xyes -a x$mpicc_path != xno -a x${mpicxx_path} != xno ; then
    build_mpi_master_slave=yes
else
    build_mpi_master_slave=no
fi

#users cannot use both at the same time
if test x$build_mpi_master_slave = xyes -a x$enable_mpi = xyes; then
    AC_MSG_WARN(StarPU-MPI and MPI Master-Slave cannot be used at the same time ! Disabling StarPU-MPI...)
    build_mpi_lib=no
    build_nmad_lib=no
    enable_mpi=no
fi

if test x$build_mpi_master_slave = xyes; then
    AC_DEFINE(STARPU_USE_MPI_MASTER_SLAVE, [1], [MPI Master Slave support is enabled])
    CC=$mpicc_path
    CCLD=$mpicc_path
    CXX=$mpicxx_path
    CXXLD=mpicxx_path
fi

AC_ARG_WITH(mpi-master-slave-multiple-thread, [AS_HELP_STRING([--with-mpi-master-slave-multiple-thread])],
	[AC_DEFINE([STARPU_MPI_MASTER_SLAVE_MULTIPLE_THREAD], [1], [Use multiple threads to communicate with slaves])])

AC_MSG_CHECKING(whether the master-slave mode should be enabled)
AC_MSG_RESULT($build_mpi_master_slave)
AM_CONDITIONAL([STARPU_USE_MPI_MASTER_SLAVE], [test x$build_mpi_master_slave = xyes])

AC_MSG_CHECKING(maximum number of MPI master-slave devices)
AC_ARG_ENABLE(maxmpidev, [AS_HELP_STRING([--enable-maxmpidev=<number>],
			[maximum number of MPI master-slave devices])],
			nmaxmpidev=$enableval,
            [
             if test x$build_mpi_master_slave = xyes; then
                 nmaxmpidev=4
             else
                 nmaxmpidev=0
             fi
            ])
AC_MSG_RESULT($nmaxmpidev)
AC_DEFINE_UNQUOTED(STARPU_MAXMPIDEVS, [$nmaxmpidev], [maximum number of MPI devices])

###############################################################################
#                                                                             #
#                       Miscellaneous things for MPI                          #
#                                                                             #
###############################################################################

AC_ARG_ENABLE(mpi-pedantic-isend, [AS_HELP_STRING([--enable-mpi-pedantic-isend],
				   [Prevent StarPU MPI from reading buffers while being sent over MPI])],
				   enable_mpi_pedantic_isend=$enableval, enable_mpi_pedantic_isend=no)
if test x$enable_mpi_pedantic_isend = xyes; then
	AC_DEFINE(STARPU_MPI_PEDANTIC_ISEND, [1], [enable StarPU MPI pedantic isend])
fi

# If the user specifically asks for it, or if we are in a developer checkout, we enable mpi check
AC_ARG_ENABLE(mpi-check, AC_HELP_STRING([--enable-mpi-check], [Enable execution of MPI testcases]),
	      [enable_mpi_check=$enableval], [enable_mpi_check=$default_enable_mpi_check])
running_mpi_check=no
if test -d "$srcdir/.git" -o -f "$srcdir/.git"; then
    running_mpi_check=yes
fi
if test x$enable_mpi_check = xyes ; then
    running_mpi_check=yes
fi
if test x$enable_mpi_check = xno ; then
    running_mpi_check=no
fi
if test x$enable_mpi = xno ; then
    running_mpi_check=no
fi

AM_CONDITIONAL(STARPU_MPI_CHECK, test x$running_mpi_check = xyes)
AC_MSG_CHECKING(whether MPI tests should be run)
AC_MSG_RESULT($running_mpi_check)

AC_MSG_CHECKING(whether the StarPU MPI library should be generated)
AC_MSG_RESULT($build_mpi_lib)
AC_MSG_CHECKING(whether the StarPU MPI nmad library should be generated)
AC_MSG_RESULT($build_nmad_lib)

if test x$build_mpi_lib = xyes -o x$build_nmad_lib = xyes ; then
	AC_DEFINE(STARPU_USE_MPI,[1],[whether the StarPU MPI library is available])
	if test x$build_mpi_lib = xyes ; then
		AC_DEFINE(STARPU_USE_MPI_MPI,[1],[whether the StarPU MPI library (with a native MPI implementation) is available])
	else
		AC_DEFINE(STARPU_USE_MPI_NMAD,[1],[whether the StarPU MPI library (with a NewMadeleine implementation) is available])
	fi
fi

if test x$enable_mpi = xyes ; then
    if test x$enable_simgrid = xyes ; then
        if test x$enable_shared = xyes ; then
	    AC_MSG_ERROR([MPI with simgrid can not work with shared libraries, use --disable-shared to fix this, or disable MPI with --disable-mpi])
        else
	    CFLAGS="$CFLAGS -fPIC"
	    CXXFLAGS="$CXXFLAGS -fPIC"
	    NVCCFLAGS="$NVCCFLAGS --compiler-options -fPIC"
	    FFLAGS="$FFLAGS -fPIC"
        fi
    fi
fi

AM_CONDITIONAL(STARPU_USE_MPI_MPI, test x$build_mpi_lib = xyes)
AM_CONDITIONAL(STARPU_USE_MPI_NMAD, test x$build_nmad_lib = xyes)
AM_CONDITIONAL(STARPU_USE_MPI, test x$build_nmad_lib = xyes -o x$build_mpi_lib = xyes)

AC_ARG_WITH(mpiexec-args, [AS_HELP_STRING([--with-mpiexec-args[=<arguments to give when running mpiexec>]],
					  [Arguments for mpiexec])],
	    [
		mpiexec_args=$withval
	    ])
AC_SUBST(MPIEXEC_ARGS,$mpiexec_args)

AC_MSG_CHECKING(whether MPI debug messages should be displayed)
AC_ARG_ENABLE(mpi-verbose, [AS_HELP_STRING([--enable-mpi-verbose],
					   [display MPI verbose debug messages (--enable-mpi-verbose=extra increase the verbosity)])],
	      enable_mpi_verbose=$enableval, enable_mpi_verbose=no)
AC_MSG_RESULT($enable_mpi_verbose)
if test x$enable_mpi_verbose = xyes; then
	AC_DEFINE(STARPU_MPI_VERBOSE, [1], [display MPI verbose debug messages])
fi
if test x$enable_mpi_verbose = xextra; then
	AC_DEFINE(STARPU_MPI_VERBOSE, [1], [display MPI verbose debug messages])
	AC_DEFINE(STARPU_MPI_EXTRA_VERBOSE, [1], [display MPI verbose debug messages])
fi

if test x$enable_mpi = xyes -o x$build_mpi_master_slave = xyes ; then
    cc_or_mpicc=$mpicc_path
    # For some reason, libtool uses gcc instead of mpicc when linking
    # libstarpumpi.
    # On Darwin (and maybe other systems ?) the linker will fail (undefined
    # references to MPI_*). We manually add the required flags to fix this
    # issue.
    AC_SUBST(MPICC_LDFLAGS, `$mpicc_path --showme:link`)
else
    cc_or_mpicc=$CC
fi
AC_SUBST(CC_OR_MPICC, $cc_or_mpicc)

###############################################################################
#                                                                             #
#                           MIC device compilation                            #
#   (Must be done in beginning to change prefix in the whole configuration)   #
#                                                                             #
###############################################################################

AC_ARG_ENABLE(mic, [AS_HELP_STRING([--enable-mic],
	      [use MIC device(s)])], [enable_mic=$enableval], [enable_mic=no])
AC_ARG_ENABLE(mic-rma, [AS_HELP_STRING([--disable-mic-rma],
	      [disable MIC RMA transfer])], [enable_mic_rma=$enableval], [enable_mic_rma=yes])

if test x$enable_mic = xyes ; then
	AC_DEFINE(STARPU_USE_MIC, [1], [MIC workers support is enabled])
fi
if test x$enable_mic_rma = xyes ; then
	AC_DEFINE([STARPU_MIC_USE_RMA], [1], [MIC RMA transfer is enable])
fi

AM_CONDITIONAL([STARPU_USE_MIC], [test "x$enable_mic" = "xyes"])

###############################################################################

###############################################################################
#                                                                             #
#                           NUMA memory nodes                                 #
#                                                                             #
###############################################################################

AC_MSG_CHECKING(maximum number of NUMA nodes)
AC_ARG_ENABLE(maxnumanodes, [AS_HELP_STRING([--enable-maxnumanodes=<number>],
			[maximum number of NUMA nodes])],
			nmaxnumanodes=$enableval, nmaxnumanodes=2)
AC_MSG_RESULT($nmaxnumanodes)
AC_DEFINE_UNQUOTED(STARPU_MAXNUMANODES, [$nmaxnumanodes],
		[maximum number of NUMA nodes])


###############################################################################

AC_PATH_PROGS([STARPU_MS_LIB], [lib])
AC_ARG_VAR([STARPU_MS_LIB], [Path to Microsoft's Visual Studio `lib' tool])
AM_CONDITIONAL([STARPU_HAVE_MS_LIB], [test "x$STARPU_MS_LIB" != "x"])
case "$target" in
*-*-mingw*|*-*-cygwin*)
  starpu_windows=yes
  libext=a
  AC_DEFINE(STARPU_HAVE_WINDOWS, [1], [Define this on windows.])
  ;;
*-*-linux*)
  starpu_linux=yes
  AC_DEFINE(STARPU_LINUX_SYS, [1], [Define to 1 on Linux])
  ;;
*-*-openbsd*)
  starpu_openbsd=yes
  AC_DEFINE(STARPU_OPENBSD_SYS, [1], [Define to 1 on OpenBSD systems])
  ;;
*-*darwin*)
  starpu_darwin=yes
  AC_DEFINE(STARPU_HAVE_DARWIN, [1], [Define this on darwin.])
  ;;
esac
AM_CONDITIONAL([STARPU_HAVE_WINDOWS], [test "x$starpu_windows" = "xyes"])
AM_CONDITIONAL([STARPU_LINUX_SYS], [test "x$starpu_linux" = "xyes"])
AM_CONDITIONAL([STARPU_HAVE_DARWIN], [test "x$starpu_darwin" = "xyes"])
AM_CONDITIONAL([STARPU_OPENBSD_SYS], [test "x$starpu_openbsd" = "xyes"])

# on Darwin, GCC targets i386 by default, so we don't have atomic ops
AC_CHECK_SIZEOF([void *])
SIZEOF_VOID_P=$ac_cv_sizeof_void_p
case $SIZEOF_VOID_P in
	4)
		case "$target" in
		i386-*darwin*) CFLAGS+=" -march=i686 " ;;
		esac
		STARPU_MS_LIB_ARCH=X86
		;;
	8)
		STARPU_MS_LIB_ARCH=X64
		;;
esac
AC_SUBST(STARPU_MS_LIB_ARCH)

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

case "$target" in
*-*-mingw*|*-*-cygwin*)
    AC_ARG_ENABLE(native-winthreads, [AS_HELP_STRING([--enable-native-winthreads],
    				   [Use native windows threads instead of pthread])],
    				   enable_native_winthreads=$enableval, enable_native_winthreads=no)
    ;;
esac
if test x"$enable_native_winthreads" != xyes ; then
    INCLUDE_PTHREAD_H='#include <pthread.h>'
fi

AC_CHECK_HEADERS([unistd.h], [AC_DEFINE([STARPU_HAVE_UNISTD_H], [1], [Define to 1 if you have the <unistd.h> header file.])])

AC_CHECK_TYPE([struct timespec],
	       AC_DEFINE(STARPU_HAVE_STRUCT_TIMESPEC,[1],[struct timespec is defined]),
	       [], [
#include <sys/types.h>
#include <sys/stat.h>
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif
#include <time.h>
$INCLUDE_PTHREAD_H
])

if test x"$enable_native_winthreads" = xyes ; then
    CPPFLAGS+=" -I$STARPU_SRC_DIR/include/pthread_win32 "
    AC_COMPILE_IFELSE(
          [AC_LANG_PROGRAM([[
	        #define STARPU_CONFIGURE
		#include <pthread.h>
		]],
		[[ pthread_t t; pthread_create(&t, NULL, NULL, NULL); ]])],
		AC_DEFINE(STARPU_NATIVE_WINTHREADS,[1],[Using native windows threads]),
		AC_MSG_ERROR([pthread_create unavailable]))
else
    AC_CHECK_LIB([pthread], [pthread_create], [
        LIBS="$LIBS -lpthread"
        STARPU_EXPORTED_LIBS="$STARPU_EXPORTED_LIBS -lpthread"
    ])
fi

AC_SEARCH_LIBS([sqrt],[m],,AC_MSG_ERROR([math library unavailable]))
AC_HAVE_LIBRARY([ws2_32])
AC_CHECK_FUNCS([sysconf])
AC_CHECK_FUNCS([getrlimit])
AC_CHECK_FUNCS([scandir])

AC_CHECK_FUNC([pthread_spin_lock], have_pthread_spin_lock=yes, have_pthread_spin_lock=no)
if test x$have_pthread_spin_lock = xyes; then
	AC_DEFINE(HAVE_PTHREAD_SPIN_LOCK,[1],[pthread_spin_lock is available])
	AC_DEFINE(STARPU_HAVE_PTHREAD_SPIN_LOCK,[1],[pthread_spin_lock is available])
fi

AC_CHECK_FUNC([pthread_barrier_init], have_pthread_barrier=yes, have_pthread_barrier=no)
if test x$have_pthread_barrier = xyes; then
	AC_DEFINE(STARPU_HAVE_PTHREAD_BARRIER,[1],[pthread_barrier is available])
fi

# yes, that's non portable, but it's still better than sched_setaffinity
AC_CHECK_FUNCS(pthread_setaffinity_np)

AC_CHECK_FUNC([pthread_setname_np], have_pthread_setname_np=yes, have_pthread_setname_np=no)
if test x$have_pthread_setname_np = xyes; then
	AC_DEFINE(STARPU_HAVE_PTHREAD_SETNAME_NP,[1],[pthread_setname_np is available])
fi

# There is no posix_memalign on Mac OS X, only memalign
AC_CHECK_FUNCS([posix_memalign], [AC_DEFINE([STARPU_HAVE_POSIX_MEMALIGN], [1], [Define to 1 if you have the `posix_memalign' function.])])
AC_CHECK_FUNCS([memalign], [AC_DEFINE([STARPU_HAVE_MEMALIGN], [1], [Define to 1 if you have the `memalign' function.])])

# Some systems don't have drand48
AC_CHECK_FUNC([drand48], have_drand48=yes, have_drand48=no)
AC_CHECK_FUNC([erand48_r], have_erand48_r=yes, have_erand48_r=no)
# Maybe the user still does not want to use the provided drand48
AC_ARG_ENABLE(default-drand48, [AS_HELP_STRING([--disable-default-drand48],
				   [Do not use the default version of drand48])],
				   enable_default_drand48=$enableval, enable_default_drand48=yes)
if test x$have_drand48 = xyes -a x$enable_default_drand48 = xyes ; then
   AC_DEFINE([STARPU_USE_DRAND48], [1], [Define to 1 if drandr48 is available and should be used])
fi
if test x$have_erand48_r = xyes ; then
   AC_DEFINE([STARPU_USE_ERAND48_R], [1], [Define to 1 if erandr48_r is available])
fi

# Some systems do not define strerror_r
AC_CHECK_FUNC([strerror_r], [AC_DEFINE([STARPU_HAVE_STRERROR_R], [1], [Define to 1 if the function strerro_r is available.])])

# Some systems may not define setenv
AC_CHECK_FUNC([setenv], [AC_DEFINE([STARPU_HAVE_SETENV], [1], [Define to 1 if the function setenv is available.])])

# Some systems do not define unsetenv
AC_CHECK_FUNC([unsetenv], [AC_DEFINE([STARPU_HAVE_UNSETENV], [1], [Define to 1 if the function unsetenv is available.])])

# Some systems do not define nearbyintf...
AC_CHECK_FUNC([nearbyintf], [AC_DEFINE([STARPU_HAVE_NEARBYINTF], [1], [Define to 1 if the function nearbyintf is available.])])

# ... but they may define rintf.
AC_CHECK_FUNC([rintf], [AC_DEFINE([STARPU_HAVE_RINTF], [1], [Define to 1 if the function rintf is available.])])

# Define quick check
AC_ARG_ENABLE(quick-check, [AS_HELP_STRING([--enable-quick-check],
				   [Lower default values for the testcases run by make check to allow a faster execution])],
				   enable_quick_check=$enableval, enable_quick_check=no)
if  test x$enable_quick_check = xyes; then
	AC_DEFINE(STARPU_QUICK_CHECK, [1], [enable quick check])
fi
AM_CONDITIONAL([STARPU_QUICK_CHECK], [test "x$enable_quick_check" = "xyes"])

# Define long check
AC_ARG_ENABLE(long-check, [AS_HELP_STRING([--enable-long-check],
				   [Enable some exhaustive checks which take a really long time])],
				   enable_long_check=$enableval, enable_long_check=no)
if  test x$enable_long_check = xyes; then
	AC_DEFINE(STARPU_LONG_CHECK, [1], [enable long check])
fi
AM_CONDITIONAL([STARPU_LONG_CHECK], [test "x$enable_long_check" = "xyes"])

# Define new check
AC_ARG_ENABLE(new-check, [AS_HELP_STRING([--enable-new-check],
				   [Enable new and known-to-fail testcases])],
				   enable_new_check=$enableval, enable_new_check=no)
if  test x$enable_new_check = xyes; then
	AC_DEFINE(STARPU_NEW_CHECK, [1], [enable new check])
fi
AM_CONDITIONAL([STARPU_NEW_CHECK], [test "x$enable_new_check" = "xyes"])

AC_CHECK_HEADERS([malloc.h], [AC_DEFINE([STARPU_HAVE_MALLOC_H], [1], [Define to 1 if you have the <malloc.h> header file.])])

AC_ARG_ENABLE(valgrind, [AS_HELP_STRING([--disable-valgrind],
				   [Do not check the availability of valgrind.h and helgrind.h])],
				   enable_valgrind=$enableval, enable_valgrind=yes)
if test "$enable_valgrind" != "no" ; then
   AC_CHECK_HEADERS([valgrind/valgrind.h], [AC_DEFINE([STARPU_HAVE_VALGRIND_H], [1], [Define to 1 if you have the <valgrind/valgrind.h> header file.])])
   AC_CHECK_HEADERS([valgrind/memcheck.h], [AC_DEFINE([STARPU_HAVE_MEMCHECK_H], [1], [Define to 1 if you have the <valgrind/memcheck.h> header file.])])
   AC_CHECK_HEADERS([valgrind/helgrind.h], [AC_DEFINE([STARPU_HAVE_HELGRIND_H], [1], [Define to 1 if you have the <valgrind/helgrind.h> header file.])])
fi
if test "$enable_valgrind" = "full" ; then
   AC_DEFINE([STARPU_VALGRIND_FULL], [1], [Define to 1 to disable STARPU_SKIP_IF_VALGRIND when running tests.])
fi

AC_CHECK_FUNC([sched_yield], [AC_DEFINE([STARPU_HAVE_SCHED_YIELD], [1], [Define to 1 if the function sched_yield is available.])])

AC_CHECK_HEADERS([aio.h])
AC_CHECK_LIB([rt], [aio_read])
#AC_CHECK_HEADERS([libaio.h])
#AC_CHECK_LIB([aio], [io_setup])
AC_CHECK_FUNCS([copy_file_range])

AC_CHECK_FUNCS([mkostemp])
AC_CHECK_FUNCS([mkdtemp])

AC_CHECK_FUNCS([pread pwrite])

AC_ARG_ENABLE(hdf5, [AS_HELP_STRING([--enable-hdf5], [disable HDF5 support])],
                    enable_hdf5=$enableval, enable_hdf5=no)

if test "x$enable_hdf5" != xno ; then
	AC_ARG_WITH(hdf5-include-dir,
		[AS_HELP_STRING([--with-hdf5-include-dir=<path>],
		[specify where HDF5 headers are installed])],
		[
			hdf5_include_dir="$withval"
		], [hdf5_include_dir=""])

	hdf5_inc_dir="/usr/include/hdf5 /usr/include/hdf5/serial ${hdf5_include_dir}"

	enable_include_hdf5=no
	for f in $hdf5_inc_dir; do
		if test -n "$f" ; then
			SAVED_CPPFLAGS="${CPPFLAGS}"
			CPPFLAGS="$CPPFLAGS -I$f"
			AC_CHECK_HEADERS([hdf5.h])
			if test "$ac_cv_header_hdf5_h" = "yes" ; then
				CPPFLAGS="-I${f} ${SAVED_CFLAGS}"
				enable_include_hdf5=yes
				break
			else
				CPPFLAGS=${SAVED_CPPFLAGS}
			fi
			unset ac_cv_header_hdf5_h
		fi
	done


	AC_ARG_WITH(hdf5-lib-dir,
		[AS_HELP_STRING([--with-hdf5-lib-dir=<path>],
		[specify where HDF5 libraries are installed])],
		[
			hdf5_libraries_dir="$withval"
		], [hdf5_libraries_dir=""])

	hdf5_lib_dir="/usr/lib/x86_64-linux-gnu/hdf5 /usr/lib/x86_64-linux-gnu/hdf5/serial ${hdf5_libraries_dir}"

	enable_libraries_hdf5=no
	for f in $hdf5_lib_dir; do
		if test -n "$f" ; then
			SAVED_LDFLAGS="${LDFLAGS}"
			LDFLAGS=-L${f}
			STARPU_HAVE_LIBRARY(HDF5, [hdf5])
			if test "$ac_cv_lib_hdf5_main" = "yes" ; then
				LDFLAGS="-L${f} ${SAVED_LDFLAGS} ${STARPU_HDF5_LDFLAGS}"
				enable_libraries_hdf5=yes
				break
			else
				LDFLAGS=${SAVED_LDFLAGS}
			fi
			unset ac_cv_lib_hdf5_main
		fi
	done
fi

if test  "x$enable_libraries_hdf5" = "xyes" -a "x$enable_include_hdf5" = "xyes" -a "x$enable_hdf5" != "xno"; then
        AC_DEFINE([STARPU_HAVE_HDF5], [1], [Define to 1 if you have the <hdf5.h> header file.])
	enable_hdf5=yes
else
	enable_hdf5=no
fi
AM_CONDITIONAL(STARPU_HAVE_HDF5, test "x$enable_hdf5" = "xyes")


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

# This defines HAVE_ATOMIC_COMPARE_EXCHANGE_N
STARPU_CHECK_ATOMIC_COMPARE_EXCHANGE_N

# This defines HAVE_ATOMIC_EXCHANGE_N
STARPU_CHECK_ATOMIC_EXCHANGE_N

# This defines HAVE_ATOMIC_FETCH_ADD
STARPU_CHECK_ATOMIC_FETCH_ADD

# This defines HAVE_ATOMIC_FETCH_OR
STARPU_CHECK_ATOMIC_FETCH_OR

# This defines HAVE_ATOMIC_TEST_AND_SET
STARPU_CHECK_ATOMIC_TEST_AND_SET

# This defines HAVE_SYNC_SYNCHRONIZE
STARPU_CHECK_SYNC_SYNCHRONIZE

CPPFLAGS="${CPPFLAGS} -D_GNU_SOURCE "

STARPU_SEARCH_LIBS([LIBNUMA],[set_mempolicy],[numa],[enable_libnuma=yes],[enable_libnuma=no])
AC_MSG_CHECKING(whether libnuma is available)
AC_MSG_RESULT($enable_libnuma)
if test x$enable_libnuma = xyes; then
	AC_DEFINE(STARPU_HAVE_LIBNUMA,[1],[libnuma is available])
fi

AC_MSG_CHECKING(whether statement expressions are available)
AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
#define maxint(a,b) ({int _a = (a), _b = (b); _a > _b ? _a : _b; })
]],
		[[ int x=maxint(12,42); ]])],
		[statement_expressions="yes"],
		[statement_expressions="no"])
AC_MSG_RESULT($statement_expressions)
if test x$statement_expressions = xyes; then
	AC_DEFINE(STARPU_HAVE_STATEMENT_EXPRESSIONS,[1],[statement expressions are available])
fi

###############################################################################
#									      #
#				SCHED_CTX settings			      #
#									      #
###############################################################################
AC_MSG_CHECKING(maximum number of sched_ctxs)
AC_ARG_ENABLE(max_sched_ctxs, [AS_HELP_STRING([--enable-max-sched-ctxs=<number>],
			[maximum number of sched_ctxs])],
			max_sched_ctxs=$enableval, max_sched_ctxs=10)
AC_MSG_RESULT($max_sched_ctxs)
AC_DEFINE_UNQUOTED(STARPU_NMAX_SCHED_CTXS, [$max_sched_ctxs], [Maximum number of sched_ctxs supported])

AC_ARG_ENABLE([sc_hypervisor],
  [AS_HELP_STRING([--enable-sc-hypervisor],
    [enable resizing contexts (experimental)])],
  [enable_sc_hypervisor="yes"],
  [enable_sc_hypervisor="no"])

#for pkgconfig
AC_SUBST(STARPU_SC_HYPERVISOR)
if test "x$enable_sc_hypervisor" = "xyes"; then
  AC_DEFINE(STARPU_USE_SC_HYPERVISOR, [1], [enable sc_hypervisor lib])
#   PKG_CHECK_MODULES([SC_HYPERVISOR], [libsc_hypervisor], [], build_sc_hypervisor="yes")
   STARPU_SC_HYPERVISOR="-lsc_hypervisor"
   build_sc_hypervisor="yes"
else
   build_sc_hypervisor="no"
fi

AM_CONDITIONAL([STARPU_BUILD_SC_HYPERVISOR], [test "x$build_sc_hypervisor" = "xyes"])
AM_CONDITIONAL([STARPU_USE_SC_HYPERVISOR], [test "x$build_sc_hypervisor" = "xyes"])

AC_ARG_ENABLE([sc_hypervisor_debug],
  [AS_HELP_STRING([--enable-sc-hypervisor-debug],
    [enable debug for resizing contexts (experimental)])],
  [enable_sc_hypervisor_debug="yes"],
  [enable_sc_hypervisor_debug="no"])


AC_SUBST(STARPU_SC_HYPERVISOR_DEBUG, $enable_sc_hypervisor_debug)
AM_CONDITIONAL([STARPU_SC_HYPERVISOR_DEBUG], [test "x$enable_sc_hypervisor_debug" = "xyes"])

if test "x$enable_sc_hypervisor_debug" = "xyes"; then
  AC_DEFINE(STARPU_SC_HYPERVISOR_DEBUG, [1], [enable debug sc_hypervisor])
fi

###############################################################################
#                                                                             #
#                                 CPUs settings                               #
#                                                                             #
###############################################################################

AC_MSG_CHECKING(maximum number of CPUs)
AC_ARG_ENABLE(maxcpus, [AS_HELP_STRING([--enable-maxcpus=<number>],
			[maximum number of CPUs])],
			maxcpus=$enableval, maxcpus=64)
AC_MSG_RESULT($maxcpus)
AC_DEFINE_UNQUOTED(STARPU_MAXCPUS, [$maxcpus], [Maximum number of CPUs supported])

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

AC_DEFUN([STARPU_CHECK_CUDA_L],
[
    __cuda_L=$1
    SAVED_LDFLAGS="${LDFLAGS}"
    STARPU_CUDA_LDFLAGS="${__cuda_L}"
    AC_MSG_CHECKING(whether CUDA library is available in $__cuda_L)
    AC_MSG_RESULT()
    LDFLAGS="${SAVED_LDFLAGS} ${__cuda_L}"
    AC_HAVE_LIBRARY([cudart],[have_valid_cuda=yes],[have_valid_cuda=no])
    unset ac_cv_lib_cudart_main
    if test "$have_valid_cuda" = yes ; then
	STARPU_CUDA_LDFLAGS="$STARPU_CUDA_LDFLAGS -lcudart"
	LDFLAGS="${SAVED_LDFLAGS} ${STARPU_CUDA_LDFLAGS}"
	# we also check that CUBLAS is available
	AC_HAVE_LIBRARY([cublas],[have_valid_cuda=yes],[have_valid_cuda=no])
	unset ac_cv_lib_cublas_main
	if test "$have_valid_cuda" = "yes" ; then
	   STARPU_CUDA_LDFLAGS="$STARPU_CUDA_LDFLAGS -lcublas"
	fi
    fi
    LDFLAGS="${SAVED_LDFLAGS}"
])
AC_DEFUN([STARPU_CHECK_CUDA],
[
    __cuda_dir=$1
    __cuda_include_dir=$2
    __cuda_lib_dir=$3

    if test -z "$__cuda_lib_dir" ; then
	__cuda_lib_dir=no
    fi
    if test -z "$__cuda_include_dir" ; then
	__cuda_include_dir=no
    fi
    if test -z "$__cuda_dir" ; then
	__cuda_dir=no
    fi

    if test "$__cuda_dir" != "no" ; then
	AC_MSG_CHECKING(whether CUDA is available in $__cuda_dir, $__cuda_include_dir and $__cuda_lib_dir)
    else
	AC_MSG_CHECKING(whether CUDA is available)
    fi
    AC_MSG_RESULT()

    if test "$__cuda_include_dir" = "no" -a "$__cuda_dir" != "no" ; then
        __cuda_include_dir="$__cuda_dir/include"
    fi

    SAVED_CPPFLAGS="$CPPFLAGS"
    have_valid_cuda=no

    if test "$__cuda_include_dir" != "no" ; then
        CPPFLAGS="${CPPFLAGS} -I$__cuda_include_dir"
    fi

    AC_CHECK_HEADER([cuda.h],[have_valid_cuda=yes],[have_valid_cuda=no])
    unset ac_cv_header_cuda_h

    if test "$have_valid_cuda" = "yes" ; then
	if test "$__cuda_lib_dir" != "no" ; then
	    STARPU_CHECK_CUDA_L("-L${__cuda_lib_dir}")
	else
	    if test "$__cuda_dir" != "no" ; then
		for __cuda_libdir in lib64 lib lib/x64 lib/Win32 ; do
		    STARPU_CHECK_CUDA_L("-L${__cuda_dir}/${__cuda_libdir}")
		    if test "$have_valid_cuda" = yes ; then
			break
		    fi
		done
	    else
		STARPU_CHECK_CUDA_L("")
	    fi
	fi
    fi

    if test "$have_valid_cuda" = "no" ; then
	CPPFLAGS="${SAVED_CPPFLAGS}"
	unset STARPU_CUDA_LDFLAGS
    else
   	if test "$NVCC" = "" ; then
            AC_PATH_PROG([NVCC], [nvcc], [not-found],
	   	[$cuda_dir/bin:$PATH:/usr/local/cuda/bin:/usr/bin:/bin])
	fi
	if test "x$NVCC" = "xnot-found"; then
	    AC_MSG_WARN(['nvcc' not found, disabling CUDA])
	    have_valid_cuda=no
	else
	    # This is for very old cuda, to enable the use of double etc.
	    AC_MSG_CHECKING(whether nvcc supports sm_13 architecture)
	    OLD_NVCCFLAGS="$NVCCFLAGS"
	    NVCCFLAGS="$NVCCFLAGS -arch sm_13"
	    echo "int main(int argc, char **argv) { return 0;}" > cuda_test.cu
	    $NVCC $NVCCFLAGS -c cuda_test.cu >/dev/null 2>&1
	    if test $? -eq 0
	    then
		AC_MSG_RESULT(yes)
	    else
		AC_MSG_RESULT(no)
		NVCCFLAGS="$OLD_NVCCFLAGS"
	    fi

	    # This is for recent cuda, which complains if we don't actually set an arch!?
	    AC_MSG_CHECKING(whether nvcc supports -Wno-deprecated-gpu-targets)
	    OLD_NVCCFLAGS="$NVCCFLAGS"
	    NVCCFLAGS="$NVCCFLAGS -Wno-deprecated-gpu-targets"
	    echo "int main(int argc, char **argv) { return 0;}" > cuda_test.cu
	    $NVCC $NVCCFLAGS -c cuda_test.cu >/dev/null 2>&1
	    if test $? -eq 0
	    then
		AC_MSG_RESULT(yes)
	    else
		AC_MSG_RESULT(no)
		NVCCFLAGS="$OLD_NVCCFLAGS"
	    fi

	    rm -f cuda_test*
	fi

	if test -n "$NVCC_CC"; then
	    NVCCFLAGS="${NVCCFLAGS} -ccbin \${NVCC_CC}"
	fi
	if test "$__cuda_include_dir" != "no"; then
	    STARPU_CUDA_CPPFLAGS="-I$__cuda_include_dir"
	    NVCCFLAGS="${NVCCFLAGS} -I$__cuda_include_dir"
	fi
    fi
])

if test x$enable_cuda = xyes -o x$enable_cuda = xmaybe; then
    STARPU_CHECK_CUDA("$cuda_dir", "$cuda_include_dir", "$cuda_lib_dir")
    if test "$have_valid_cuda" = "no" ; then
	STARPU_CHECK_CUDA("$CUDA_ROOT", "$CUDA_PATH", "$CUDA_INC_PATH", "$CUDA_LIB_PATH")
    fi
    if test "$have_valid_cuda" = "no" ; then
	for f in "/usr/local/cuda" "/c/cuda" "/cygdrive/c/cuda" "/opt/cuda" "$CUDA_ROOT" "$CUDA_PATH" "$CUDA_INC_PATH/.." "$CUDA_INC/.." "$CUDA_BIN/.." "$CUDA_SDK/.." "$CUDA_INSTALL_PATH" "$CUDA_TOOLKIT"; do
	    if test -n "$f" ; then
		STARPU_CHECK_CUDA("$f", "no", "no")
		if test "$have_valid_cuda" = "yes" ; then
		    break
		fi
	    fi
	done
    fi

    # Check cuda is compatible with the C compiler
    AC_MSG_CHECKING(whether CUDA is working)
    if test "$have_valid_cuda" = "yes" ; then
        SAVED_CPPFLAGS="${CPPFLAGS}"
        CPPFLAGS="${CPPFLAGS} ${STARPU_CUDA_CPPFLAGS}"
	SAVED_LDFLAGS="${LDFLAGS}"
	LDFLAGS="${LDFLAGS} ${STARPU_CUDA_LDFLAGS} -lcudart"
	AC_COMPILE_IFELSE([AC_LANG_PROGRAM(
		[[#include <cuda.h>]],
		[[]]
		)],
	    [
	      AC_RUN_IFELSE([AC_LANG_PROGRAM(
	        [[#include <cuda.h>]],
		[[]]
		)],
		[have_valid_cuda="yes"],
		[
	          AC_MSG_RESULT([CUDA found and can be compiled, but compiled application can not be run, is the CUDA path missing in LD_LIBRARY_PATH?])
	          have_valid_cuda="no"
		])
	    ],
	    [
	    AC_MSG_ERROR([CUDA found, but cuda.h could not be compiled])
	    have_valid_cuda="no"
	    ]
	)
        CPPFLAGS="${SAVED_CPPFLAGS}"
	LDFLAGS="${SAVED_LDFLAGS}"
    fi
    AC_MSG_RESULT($have_valid_cuda)

    # in case CUDA was explicitely required, but is not available, this is an error
    if test x$enable_cuda = xyes -a x$have_valid_cuda = xno; then
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
	STARPU_CUDA_LDFLAGS="$STARPU_CUDA_LDFLAGS -lcudart"
	STARPU_CUFFT_LDFLAGS="-lcufft"

	AC_LANG_PUSH([C++])
	case \ $NVCCFLAGS\  in
	*\ -std=*\ *) ;;
	*)
		SAVED_CXX="$CXX"
		CXX="$NVCC"
		AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
				  #ifdef STARPU_HAVE_SIMGRID_MSG_H
				  #include <simgrid/msg.h>
				  #include <simgrid/host.h>
				  #else
				  #include <msg/msg.h>
				  #endif
				  ]])],,
				  NVCCFLAGS="-std=c++11 $NVCCFLAGS")
		CXX="$SAVED_CXX"
	esac
	AC_LANG_POP([C++])

        if test "$F77" = "gfortran" -o "$FC" = "gfortran" ; then
            STARPU_CUDA_FORTRAN_LDFLAGS="-lgfortran"
            AC_SUBST(STARPU_CUDA_FORTRAN_LDFLAGS)
        fi

	#in case this is a 64bit setup, we tell nvcc to use a -m64 flag, if missing from existing flags
	if test x$SIZEOF_VOID_P = x8; then
		case \ $NVCCFLAGS\  in
			*\ -m64\ *) ;;
			*) NVCCFLAGS="${NVCCFLAGS} -m64" ;;
		esac
	fi

	SAVED_CPPFLAGS="${CPPFLAGS}"
	CPPFLAGS="${CPPFLAGS} ${STARPU_CUDA_CPPFLAGS}"
	SAVED_LDFLAGS="${LDFLAGS}"
	LDFLAGS="${LDFLAGS} ${STARPU_CUDA_LDFLAGS}"
	SAVED_LIBS="${LIBS}"
	AC_CHECK_HEADERS([cuda_gl_interop.h])

	AC_CHECK_LIB([cusparse], [cusparseCreate],
		     [AC_DEFINE([HAVE_LIBCUSPARSE], [1], [Define to 1 if you have the cusparse library])
		      STARPU_CUDA_LDFLAGS="$STARPU_CUDA_LDFLAGS -lcusparse"])
	AC_CHECK_DECLS([cusparseSetStream], [], [], [[#include <cusparse.h>]])

	LDFLAGS="${LDFLAGS} -lnvidia-ml"
	AC_COMPILE_IFELSE([AC_LANG_PROGRAM(
		[[#include <nvml.h>]],
		[[nvmlInit();]]
		)],
	    [
	      AC_RUN_IFELSE([AC_LANG_PROGRAM(
	        [[#include <nvml.h>]],
		[[nvmlInit();]]
		)],
		[have_valid_nvml="yes"],
		[
	          AC_MSG_RESULT([NVML found and can be compiled, but compiled application can not be run, you are probably on a machine without the CUDA driver])
	          have_valid_nvml="no"
		])
	    ],
	    [
	    AC_MSG_WARN([NVML found, but nvml.h could not be compiled])
	    have_valid_nvml="no"
	    ]
	)
	if test x$have_valid_nvml = xyes ; then
		AC_CHECK_DECLS([nvmlDeviceGetTotalEnergyConsumption], [
			AC_CHECK_FUNCS([nvmlDeviceGetTotalEnergyConsumption])
			], [], [[#include <nvml.h>]])
		AC_DEFINE([HAVE_LIBNVIDIA_ML], [1], [Define to 1 if you have the nvidia-ml library])
		STARPU_CUDA_LDFLAGS="$STARPU_CUDA_LDFLAGS -lnvidia-ml"
	fi
	AC_MSG_CHECKING(whether nvidia-ml should be used)
	AC_MSG_RESULT($have_valid_nvml)

	CPPFLAGS="${SAVED_CPPFLAGS}"
	LDFLAGS="${SAVED_LDFLAGS}"
	LIBS="${SAVED_LIBS}"
fi

have_magma=no
if test x$enable_cuda = xyes; then
	PKG_CHECK_MODULES([MAGMA],  [magma], [
	AC_DEFINE([STARPU_HAVE_MAGMA], [1], [Define to 1 if you have the MAGMA library.])
	AC_SUBST([STARPU_HAVE_MAGMA], [1])
	have_magma=yes
], [:])
fi
AM_CONDITIONAL(STARPU_HAVE_MAGMA, [test x$have_magma = xyes])
AC_MSG_CHECKING(whether MAGMA should be used)
AC_MSG_RESULT($have_magma)

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
    STARPU_CURAND_LDFLAGS="$STARPU_CURAND_LDFLAGS -lcurand"
    AC_SUBST(STARPU_CURAND_LDFLAGS)
fi

# Peer transfers are only supported since CUDA 4.0
# Disable them if user explicity wants to disable them
AC_ARG_ENABLE(cuda_memcpy_peer, [AS_HELP_STRING([--disable-cuda-memcpy-peer], [do not allow peer transfers when using CUDA 4.0])],, [enable_cuda_memcpy_peer=$enable_cuda])
if test x$enable_cuda_memcpy_peer = xyes; then
    AC_DEFINE(STARPU_HAVE_CUDA_MEMCPY_PEER,[1],[Peer transfers are supported in CUDA])
fi

if test x$enable_cuda = xyes; then
    if test x$starpu_windows != xyes ; then
	STARPU_CUDA_LDFLAGS="$STARPU_CUDA_LDFLAGS -lstdc++"
    fi
    AC_SUBST(STARPU_CUDA_LDFLAGS)
    AC_SUBST(STARPU_CUFFT_LDFLAGS)
    AC_SUBST(STARPU_CUDA_CPPFLAGS)
fi

AC_ARG_VAR([NVCC], [CUDA compiler])
AC_ARG_VAR([NVCC_CC], [C compiler for CUDA compiler])
AC_ARG_VAR([NVCCFLAGS], [CUDA compiler flags])

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

have_valid_opencl=no
AC_DEFUN([STARPU_CHECK_OPENCL],
[
    __opencl_dir=$1
    __opencl_include_dir=$2
    __opencl_lib_dir=$3

    if test "$__opencl_dir" != "no" ; then
	AC_MSG_CHECKING(whether OpenCL is available in $__opencl_dir $__opencl_include_dir and $__opencl_lib_dir)
    else
	AC_MSG_CHECKING(whether OpenCL is available)
    fi
    AC_MSG_RESULT()

    if test "$__opencl_include_dir" = "no" -a "$__opencl_dir" != "no" ; then
        __opencl_include_dir="$__opencl_dir/include"
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
	    AC_HAVE_LIBRARY([OpenCL],[have_valid_opencl=yes],[have_valid_opencl=no])
	    unset ac_cv_lib_OpenCL_main
	else
	    AC_MSG_CHECKING(whether OpenCL is available in $__opencl_dir)
	    AC_MSG_RESULT()
	    AC_HAVE_LIBRARY([OpenCL],[have_valid_opencl=yes],[have_valid_opencl=no])
	    unset ac_cv_lib_OpenCL_main
	    if test "$have_valid_opencl" = "no" -a "$__opencl_dir" != "no" ; then
		for __cuda_libdir in lib64 lib lib/x86 lib/Win32 ; do
		    __opencl_lib_dir="$__opencl_dir/$__cuda_libdir"
		    AC_MSG_CHECKING(whether OpenCL is available in $__opencl_dir and $__opencl_lib_dir)
		    AC_MSG_RESULT()
		    LDFLAGS="${SAVED_LDFLAGS} -L$__opencl_lib_dir"
		    AC_HAVE_LIBRARY([OpenCL],[have_valid_opencl=yes],[have_valid_opencl=no])
		    unset ac_cv_lib_OpenCL_main
		    if test "$have_valid_opencl" = yes ; then
			break
		    fi
		done
	    else
		LDFLAGS="${SAVED_LDFLAGS}"
		AC_HAVE_LIBRARY([OpenCL],[have_valid_opencl=yes],[have_valid_opencl=no])
		unset ac_cv_lib_OpenCL_main
	    fi
        fi
    fi

    if test "$have_valid_opencl" = "yes" -a "$__opencl_include_dir" != "no"; then
        STARPU_OPENCL_CPPFLAGS="-I$__opencl_include_dir"
	AC_CHECK_HEADERS([CL/cl_ext.h])
    fi

    CPPFLAGS="${SAVED_CPPFLAGS}"
    LDFLAGS="${SAVED_LDFLAGS}"

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

AC_DEFUN([STARPU_LOOK_FOR_OPENCL],
[
    	if test "x$has_opencl_being_checked" != "xyes" ; then
    	    STARPU_CHECK_OPENCL("$opencl_dir", "$opencl_include_dir", "$opencl_lib_dir")
	    if test "$have_valid_opencl" = "no" ; then
            	for f in "/usr/local/cuda" "/c/cuda" "/cygdrive/c/cuda" "/opt/cuda" "$CUDA_ROOT" "$CUDA_PATH" "$CUDA_INC_PATH/.." "$CUDA_INSTALL_PATH" "$CUDA_TOOLKIT"; do
		    if test -n "$f" ; then
    			STARPU_CHECK_OPENCL("$f", "no", "no")
			if test "$have_valid_opencl" = "yes" ; then
			    break
			fi
		    fi
		done
	    fi
	    has_opencl_being_checked=yes
	fi
])

if test x$enable_opencl = xyes -o x$enable_opencl = xmaybe; then
   case $target in
        *-*-darwin*)
          AC_MSG_CHECKING(whether OpenCL is available)

          SAVED_LIBS=$LIBS
          LIBS="$LIBS -framework OpenCL"
          AC_LINK_IFELSE(
          [AC_LANG_PROGRAM([[
          #ifdef __APPLE_CC__
          #include <OpenCL/opencl.h>
          #else
          #include <CL/cl.h>
          #endif
          ]],
            [[return clSetKernelArg(0, 0, 0, 0); ]])],
          [AC_MSG_RESULT(yes)
            enable_opencl=yes
            have_valid_opencl=yes
            STARPU_OPENCL_CPPFLAGS=
            STARPU_OPENCL_LDFLAGS="-framework OpenCL"],
          [AC_MSG_RESULT(no)
             enable_opencl=no])
          LIBS=$SAVED_LIBS
          ;;
        *)
	  STARPU_LOOK_FOR_OPENCL()
	  # in case OpenCL was explicitely required, but is not available, this is an error
	  if test x$enable_opencl = xyes -a x$have_valid_opencl = xno; then
	    AC_MSG_ERROR([cannot find OpenCL])
	  fi
	  # now we enable OpenCL if and only if a proper setup is available
	  enable_opencl=$have_valid_opencl
          ;;
   esac
   save_LIBS="$LIBS"
   LIBS="$LIBS $STARPU_OPENCL_LDFLAGS"
   AC_CHECK_FUNCS([clEnqueueMarkerWithWaitList])
   LIBS="$save_LIBS"
fi

AC_MSG_CHECKING(whether OpenCL should be used)
AC_MSG_RESULT($enable_opencl)
AC_SUBST(STARPU_USE_OPENCL, $enable_opencl)
AM_CONDITIONAL(STARPU_USE_OPENCL, test x$enable_opencl = xyes)
if test x$enable_opencl = xyes ; then
	AC_DEFINE(STARPU_USE_OPENCL, [1], [OpenCL support is activated])
	STARPU_OPENCL_CPPFLAGS="${STARPU_OPENCL_CPPFLAGS} -DSTARPU_OPENCL_DATADIR=${datarootdir}/starpu/opencl -DCL_USE_DEPRECATED_OPENCL_1_1_APIS"
        AC_SUBST(STARPU_OPENCL_DATAdir, "$(eval echo ${datarootdir}/starpu/opencl/examples)")
        AC_SUBST(STARPU_OPENCL_CPPFLAGS)
        AC_SUBST(STARPU_OPENCL_LDFLAGS)
fi

###############################################################################
#                                                                             #
# General GPU settings                                                        #
#                                                                             #
###############################################################################
AC_MSG_CHECKING(whether asynchronous copy should be disabled)
AC_ARG_ENABLE(asynchronous-copy, [AS_HELP_STRING([--disable-asynchronous-copy],
			[disable asynchronous copy between CPU and GPU])],
			enable_asynchronous_copy=$enableval, enable_asynchronous_copy=yes)
disable_asynchronous_copy=no
if test x$enable_asynchronous_copy = xno ; then
   disable_asynchronous_copy=yes
fi
AC_MSG_RESULT($disable_asynchronous_copy)
if test x$disable_asynchronous_copy = xyes ; then
   AC_DEFINE([STARPU_DISABLE_ASYNCHRONOUS_COPY], [1], [Define to 1 to disable asynchronous copy between CPU and GPU devices])
fi

AC_MSG_CHECKING(whether asynchronous CUDA copy should be disabled)
AC_ARG_ENABLE(asynchronous-cuda-copy, [AS_HELP_STRING([--disable-asynchronous-cuda-copy],
			[disable asynchronous copy between CPU and CUDA devices])],
			enable_asynchronous_cuda_copy=$enableval, enable_asynchronous_cuda_copy=yes)
disable_asynchronous_cuda_copy=no
if test x$enable_asynchronous_cuda_copy = xno ; then
   disable_asynchronous_cuda_copy=yes
fi
AC_MSG_RESULT($disable_asynchronous_cuda_copy)
if test x$disable_asynchronous_cuda_copy = xyes ; then
   AC_DEFINE([STARPU_DISABLE_ASYNCHRONOUS_CUDA_COPY], [1], [Define to 1 to disable asynchronous copy between CPU and CUDA devices])
fi

AC_MSG_CHECKING(whether asynchronous OpenCL copy should be disabled)
AC_ARG_ENABLE(asynchronous-opencl-copy, [AS_HELP_STRING([--disable-asynchronous-opencl-copy],
			[disable asynchronous copy between CPU and OPENCL devices])],
			enable_asynchronous_opencl_copy=$enableval, enable_asynchronous_opencl_copy=yes)
disable_asynchronous_opencl_copy=no
if test x$enable_asynchronous_opencl_copy = xno ; then
   disable_asynchronous_opencl_copy=yes
fi
AC_MSG_RESULT($disable_asynchronous_opencl_copy)
if test x$disable_asynchronous_opencl_copy = xyes ; then
   AC_DEFINE([STARPU_DISABLE_ASYNCHRONOUS_OPENCL_COPY], [1], [Define to 1 to disable asynchronous copy between CPU and OpenCL devices])
fi

AC_MSG_CHECKING(whether asynchronous MIC copy should be disabled)
AC_ARG_ENABLE(asynchronous-mic-copy, [AS_HELP_STRING([--disable-asynchronous-mic-copy],
			[disable asynchronous copy between CPU and MIC devices])],
			enable_asynchronous_mic_copy=$enableval, enable_asynchronous_mic_copy=yes)
disable_asynchronous_mic_copy=no
if test x$enable_asynchronous_mic_copy = xno ; then
   disable_asynchronous_mic_copy=yes
fi
AC_MSG_RESULT($disable_asynchronous_mic_copy)
if test x$disable_asynchronous_mic_copy = xyes ; then
   AC_DEFINE([STARPU_DISABLE_ASYNCHRONOUS_MIC_COPY], [1], [Define to 1 to disable asynchronous copy between CPU and MIC devices])
fi

AC_MSG_CHECKING(whether asynchronous MPI Master Slave copy should be disabled)
AC_ARG_ENABLE(asynchronous-mpi-master-slave-copy, [AS_HELP_STRING([--disable-asynchronous-mpi-master-slave-copy],
			[disable asynchronous copy between MPI Master and MPI Slave devices])],
			enable_asynchronous_mpi_master_slave_copy=$enableval, enable_asynchronous_mpi_master_slave_copy=yes)
disable_asynchronous_mpi_master_slave_copy=no
if test x$enable_asynchronous_mpi_master_slave_copy = xno ; then
   disable_asynchronous_mpi_master_slave_copy=yes
fi
AC_MSG_RESULT($disable_asynchronous_mpi_master_slave_copy)
if test x$disable_asynchronous_mpi_master_slave_copy = xyes ; then
   AC_DEFINE([STARPU_DISABLE_ASYNCHRONOUS_MPI_MS_COPY], [1], [Define to 1 to disable asynchronous copy between MPI Master and MPI Slave devices])
fi

###############################################################################
#                                                                             #
#                                 MIC settings                                #
#                                                                             #
###############################################################################

# ignore these otions, only meant for mic-configure, but also passed here.
AC_ARG_ENABLE(native-mic)
AC_ARG_WITH(compiler)
AC_ARG_WITH(mic-param)
AC_ARG_WITH(host-param)

AC_MSG_CHECKING(maximum number of MIC devices)
AC_ARG_ENABLE(maxmicdev, [AS_HELP_STRING([--enable-maxmicdev=<number>],
			[maximum number of MIC devices])],
			nmaxmicdev=$enableval,
            [
             if test x$enable_mic = xyes; then
                 nmaxmicdev=4
             else
                 nmaxmicdev=0
             fi
            ])
AC_MSG_RESULT($nmaxmicdev)

AC_DEFINE_UNQUOTED(STARPU_MAXMICDEVS, [$nmaxmicdev],
	[maximum number of MIC devices])

AC_MSG_CHECKING(maximum number of MIC threads)
AC_ARG_ENABLE(maxmicthreads, [AS_HELP_STRING([--enable-maxmicthreads=<number>],
			[maximum number of MIC threads])],
			nmaxmicthreads=$enableval, nmaxmicthreads=120)
AC_MSG_RESULT($nmaxmicthread)

AC_DEFINE_UNQUOTED(STARPU_MAXMICCORES, [$nmaxmicthreads],
	[maximum number of MIC cores])

AC_ARG_WITH(coi-dir,
	[AS_HELP_STRING([--with-coi-dir=<path>],
	[specify the MIC's COI installation directory])],
	[coi_dir="$withval"],
	[coi_dir=no])

AC_ARG_WITH(coi-include-dir,
	[AS_HELP_STRING([--with-coi-include-dir=<path>],
	[specify where the MIC's COI headers are installed])],
	[coi_include_dir="$withval"],
	[coi_include_dir=no])

AC_ARG_WITH(coi-lib-dir,
	[AS_HELP_STRING([--with-coi-lib-dir=<path>],
	[specify where the MIC's COI libraries are installed])],
	[coi_lib_dir="$withval"],
	[coi_lib_dir=no])

AC_DEFUN([STARPU_CHECK_COI_RUNTIME],
[
    __coi_dir=$1
    __coi_include_dir=$2
    __coi_lib_dir=$3

    if test "$__coi_dir" != "no" -a "$__coi_dir" != "" ; then
	AC_MSG_CHECKING(whether MIC's COI runtime is available in $__coi_dir)
    else
	AC_MSG_CHECKING(whether MIC's COI runtime is available)
    fi
    AC_MSG_RESULT()

    if test "$__coi_include_dir" = "no" -a "$__coi_dir" != "no" ; then
        __coi_include_dir="${__coi_dir}/include"
    fi
    if test "$__coi_lib_dir" = "no" -a "$__coi_dir" != "no" ; then
        __coi_lib_dir="${__coi_dir}/lib"
    fi

    SAVED_CPPFLAGS="$CPPFLAGS"
    SAVED_LDFLAGS="$LDFLAGS"

    if test "$__coi_include_dir" != "no" ; then
        CPPFLAGS="${CPPFLAGS} -I$__coi_include_dir"
    fi
    if test "$__coi_lib_dir" != "no" ; then
	LDFLAGS="${LDFLAGS} -L$__coi_lib_dir ${STARPU_SCIF_LDFLAGS}"
    fi

    AC_CHECK_HEADER([source/COIEngine_source.h],[have_valid_coi=yes],[have_valid_coi=no])

    if test "$have_valid_coi" = "yes" ; then
	AC_HAVE_LIBRARY([$4],[have_valid_coi=yes],[have_valid_coi=no])

        if test "$have_valid_coi" = "no" ; then
            if test "$3" = "no" -a "$__coi_dir" != "no" ; then
		# ${__coi_dir}/lib didn't work, let's try with lib64
                __coi_lib_dir="$__coi_dir/lib64"
		LDFLAGS="${SAVED_LDFLAGS} -L$__coi_lib_dir"
	        AC_HAVE_LIBRARY([$4],[have_valid_coi=yes],[have_valid_coi=no])
            fi
        fi
    fi

    if test "$have_valid_coi" = "yes" -a "$__coi_include_dir" != "no"; then
        STARPU_COI_CPPFLAGS="-I$__coi_include_dir"
    fi

    if test "$have_valid_coi" = "yes" ; then
        if test "$__coi_lib_dir" != "no"; then
	    STARPU_COI_LDFLAGS="-L$__coi_lib_dir"
        fi
	STARPU_COI_LDFLAGS="${STARPU_COI_LDFLAGS} -l$4"
    fi

    CPPFLAGS="${SAVED_CPPFLAGS}"
    LDFLAGS="${SAVED_LDFLAGS}"
])

AC_ARG_WITH(scif-dir,
	[AS_HELP_STRING([--with-scif-dir=<path>],
	[specify the MIC's SCIF installation directory])],
	[scif_dir="$withval"],
	[scif_dir=no])

AC_ARG_WITH(scif-include-dir,
	[AS_HELP_STRING([--with-scif-include-dir=<path>],
	[specify where the MIC's SCIF headers are installed])],
	[scif_include_dir="$withval"],
	[scif_include_dir=no])

AC_ARG_WITH(scif-lib-dir,
	[AS_HELP_STRING([--with-scif-lib-dir=<path>],
	[specify where the MIC's SCIF libraries are installed])],
	[scif_lib_dir="$withval"],
	[scif_lib_dir=no])

AC_DEFUN([STARPU_CHECK_SCIF_RUNTIME],
[
    __scif_dir=$1
    __scif_include_dir=$2
    __scif_lib_dir=$3

    if test "$__scif_dir" != "no" -a "$__scif_dir" != "" ; then
	AC_MSG_CHECKING(whether MIC's SCIF runtime is available in $__scif_dir)
    else
	AC_MSG_CHECKING(whether MIC's SCIF runtime is available)
    fi
    AC_MSG_RESULT()

    if test "$__scif_include_dir" = "no" -a "$__scif_dir" != "no" ; then
        __scif_include_dir="${__scif_dir}/include"
    fi
    if test "$__scif_lib_dir" = "no" -a "$__scif_dir" != "no" ; then
        __scif_lib_dir="${__scif_dir}/lib"
    fi

    SAVED_CPPFLAGS="$CPPFLAGS"
    SAVED_LDFLAGS="$LDFLAGS"

    if test "$__scif_include_dir" != "no" ; then
        CPPFLAGS="${CPPFLAGS} -I$__scif_include_dir"
    fi
    if test "$__scif_lib_dir" != "no" ; then
	LDFLAGS="${LDFLAGS} -L$__scif_lib_dir"
    fi

#    AC_CHECK_HEADER([source/SCIFEngine_source.h],[have_valid_scif=yes],[have_valid_scif=no])

#    if test "$have_valid_scif" = "yes" ; then
	AC_HAVE_LIBRARY([scif],[have_valid_scif=yes],[have_valid_scif=no])

        if test "$have_valid_scif" = "no" ; then
            if test "$3" = "no" -a "$__scif_dir" != "no" ; then
		# ${__scif_dir}/lib didn't work, let's try with lib64
                __scif_lib_dir="$__scif_dir/lib64"
		LDFLAGS="${SAVED_LDFLAGS} -L$__scif_lib_dir"
	        AC_HAVE_LIBRARY([scif],[have_valid_scif=yes],[have_valid_scif=no])
            fi
        fi
#    fi

    if test "$have_valid_scif" = "yes" -a "$__scif_include_dir" != "no"; then
        STARPU_SCIF_CPPFLAGS="-I$__scif_include_dir"
    fi

    if test "$have_valid_scif" = "yes" ; then
        if test "$__scif_lib_dir" != "no"; then
	    STARPU_SCIF_LDFLAGS="-L$__scif_lib_dir"
        fi
	STARPU_SCIF_LDFLAGS="${STARPU_SCIF_LDFLAGS} -lscif"
    fi

    CPPFLAGS="${SAVED_CPPFLAGS}"
    LDFLAGS="${SAVED_LDFLAGS}"
])

if test x$enable_mic = xyes ; then

    STARPU_CHECK_SCIF_RUNTIME($scif_dir, $scif_include_dir, $scif_lib_dir)
    if test "$have_valid_scif" = "no" ; then
	AC_MSG_ERROR([cannot find MIC's SCIF runtime])
    fi

    case $host_vendor in
	*1om)
	    # We are cross-compiling.
	    # Let's have a look for the device runtime which lib has a different name
	    STARPU_CHECK_COI_RUNTIME($coi_dir, $coi_include_dir, $coi_lib_dir, coi_device)
	    ;;
	*)
	    STARPU_CHECK_COI_RUNTIME($coi_dir, $coi_include_dir, $coi_lib_dir, coi_host)
	    ;;
    esac

    if test "$have_valid_coi" = "no" ; then
	AC_MSG_ERROR([cannot find MIC's COI runtime])
    fi

    AC_SUBST(STARPU_COI_CPPFLAGS)
    AC_SUBST(STARPU_COI_LDFLAGS)
    AC_SUBST(STARPU_SCIF_CPPFLAGS)
    AC_SUBST(STARPU_SCIF_LDFLAGS)
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

AC_ARG_ENABLE(spinlock_check, [AS_HELP_STRING([--enable-spinlock-check], [enable spinlock check])], enable_spinlock_check=$enableval, enable_spinlock_check=no)
AC_ARG_ENABLE(fstack-protector-all, [AS_HELP_STRING([--disable-fstack-protector-all], [disable GCC option -fstack-protector-all])], enable_fstack_protector_all=$enableval, enable_fstack_protector_all=yes)

if test x$enable_debug = xyes; then
	AC_DEFINE(STARPU_DEBUG, [1], [enable debugging statements])
	CFLAGS="$CFLAGS -O0"
	CXXFLAGS="$CXXFLAGS -O0"
	FFLAGS="$FFLAGS -O0"
	FCFLAGS="$FCFLAGS -O0"
	enable_spinlock_check=yes
	if test x$GCC = xyes; then
		if test x$starpu_windows != xyes ; then
			if test x$enable_fstack_protector_all = xyes ; then
			   CFLAGS="$CFLAGS -fstack-protector-all"
			   CXXFLAGS="$CXXFLAGS -fstack-protector-all"
			   FFLAGS="$FFLAGS -fstack-protector-all"
			   FCFLAGS="$FCFLAGS -fstack-protector-all"
			fi
		fi
	fi
else
	CFLAGS="-O3 $CFLAGS"
	CXXFLAGS="-O3 $CXXFLAGS"
	FFLAGS="-O3 $FFLAGS"
	FCFLAGS="-O3 $FCFLAGS"
fi

AC_MSG_CHECKING(whether full gdb information should be enabled)
AC_ARG_ENABLE(full-gdb-information, [AS_HELP_STRING([--disable-full-gdb-information], [disable full gdb information])],
			enable_full_gdb_information=$enableval, enable_full_gdb_information=yes)
AC_MSG_RESULT($enable_full_gdb_information)
if test x$enable_full_gdb_information = xyes -a x$GCC = xyes; then
	CFLAGS+=" -gdwarf-2 -g3"
	CXXFLAGS+=" -gdwarf-2 -g3"
	FFLAGS+=" -gdwarf-2 -g3"
	FCFLAGS+=" -gdwarf-2 -g3"
	LDFLAGS+=" -gdwarf-2 -g3"
fi
CFLAGS+=" -g "
CXXFLAGS+=" -g "
FFLAGS+=" -g "
FCFLAGS+=" -g "
LDFLAGS+=" -g "

if test x$enable_spinlock_check = xyes; then
	AC_DEFINE(STARPU_SPINLOCK_CHECK, [1], [check spinlock use])
fi

AC_MSG_CHECKING(whether extra checks should be performed)
AC_ARG_ENABLE(fast, [AS_HELP_STRING([--enable-fast],
			[do not enforce assertions])],
			enable_fast=$enableval, enable_fast=no)
AC_MSG_RESULT($enable_fast)
if test x$enable_fast = xyes; then
	AC_DEFINE(STARPU_NO_ASSERT, [1], [disable assertions])
else
        # fortify gets really enabled only with optimizations, avoid enabling it
        # when they optimizations are not enabled, because with some glibc it
        # spews a lot of warnings.
	if test x$enable_debug != xyes; then
		if test x$GCC = xyes; then
			CPPFLAGS="$CPPFLAGS -D_FORTIFY_SOURCE=1"
		fi
	fi
fi

AC_MSG_CHECKING(whether debug messages should be displayed)
AC_ARG_ENABLE(verbose, [AS_HELP_STRING([--enable-verbose],
			[display verbose debug messages (--enable-verbose=extra increase the verbosity)])],
			enable_verbose=$enableval, enable_verbose=no)
AC_MSG_RESULT($enable_verbose)
if test x$enable_verbose = xyes; then
	AC_DEFINE(STARPU_VERBOSE, [1], [display verbose debug messages])
fi
if test x$enable_verbose = xextra; then
	AC_DEFINE(STARPU_VERBOSE, [1], [display verbose debug messages])
	AC_DEFINE(STARPU_EXTRA_VERBOSE, [1], [display verbose debug messages])
fi

AC_MSG_CHECKING(whether coverage testing should be enabled)
AC_ARG_ENABLE(coverage, [AS_HELP_STRING([--enable-coverage],
			[enable coverage checking])],
			enable_coverage=$enableval, enable_coverage=no)
AC_MSG_RESULT($enable_coverage)
AC_SUBST(COVERAGE, $enable_coverage)
AM_CONDITIONAL(STARPU_COVERAGE_ENABLED, [test "x$enable_coverage" = "xyes"])
if test x$enable_coverage = xyes; then
	CFLAGS="${CFLAGS} --coverage"
	CXXFLAGS="${CXXFLAGS} --coverage"
	FFLAGS="${FFLAGS} --coverage"
	FCFLAGS="${FCFLAGS} --coverage"
	LDFLAGS="${LDFLAGS} --coverage"
	LIBS="${LIBS} -lgcov"
fi

AC_MSG_CHECKING(whether coverity mode should be enabled)
AC_ARG_ENABLE(coverity, [AS_HELP_STRING([--enable-coverity], [enable coverity mode])],
			enable_coverity=$enableval, enable_coverity=no)
AC_MSG_RESULT($enable_coverity)
AM_CONDITIONAL(STARPU_COVERITY, test x$enable_coverity = xyes)

# shall we use FxT to generate trace of the execution ?
AC_MSG_CHECKING(whether FxT traces should be generated)
AC_ARG_WITH(fxt, [AS_HELP_STRING([--with-fxt[[=<dir>]]], [generate fxt traces])],
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

	if test x$use_fxt_from_system = xno; then
	    save_PKG_CONFIG_PATH="$PKG_CONFIG_PATH"
	    PKG_CONFIG_PATH="$fxtdir/lib/pkgconfig:$PKG_CONFIG_PATH"
	    PKG_CHECK_MODULES([FXT],  [fxt], , [
		AC_MSG_WARN([Old FxT without fxt.pc file, hoping link will succeed])
		FXT_CFLAGS="-I$fxtdir/include/ "
		FXT_LDFLAGS="-L$fxtdir/lib/"
		AC_ARG_VAR(FXT_LDFLAGS)
		FXT_LIBS="-lfxt"
	    ])
	    PKG_CONFIG_PATH="$save_PKG_CONFIG_PATH"
	else
	    PKG_CHECK_MODULES([FXT],  [fxt])
	fi
	save_LIBS="$LIBS"
	LIBS="$LIBS $FXT_LIBS"
	save_LDFLAGS="$LDFLAGS"
	LDFLAGS="$LDFLAGS $FXT_LDFLAGS"
	AC_CHECK_FUNCS([fxt_close])
	AC_CHECK_FUNCS([enable_fut_flush])
	AC_CHECK_FUNCS([fut_set_filename])
	AC_CHECK_FUNCS([fut_setup_flush_callback])
	LDFLAGS="$save_LDFLAGS"
	LIBS="$save_LIBS"
	save_CFLAGS="$CFLAGS"
	CFLAGS="$CFLAGS $FXT_CFLAGS"
	AC_CHECK_DECLS([enable_fut_flush], [], [], [[#include <fut.h>]])
	AC_CHECK_DECLS([fut_set_filename], [], [], [[#include <fut.h>]])
	AC_CHECK_DECLS([fut_setup_flush_callback], [], [], [[#include <fut.h>]])
	CFLAGS="$save_CFLAGS"

	if test x$enable_simgrid = xyes -a x$enable_shared = xno ; then
                # simgrid's SMPI needs fxt to be linked in statically for
                # variable privatization to work
		FXT_LIBS="$(pkg-config --variable=libdir fxt)/libfxt.a -Wl,--as-needed $(pkg-config --libs --static fxt) -Wl,--no-as-needed"
	fi

	##########################################
	# Poti is a library to generate paje trace files
	##########################################
	PKG_CHECK_MODULES([POTI], [poti], [have_valid_poti=yes], [have_valid_poti=no])
	AC_ARG_ENABLE(poti, [AS_HELP_STRING([--enable-poti],
				[Enable the use of the POTI library to generate Paje traces])],
				enable_poti=$enableval, enable_poti=no)
	if test x$enable_poti = xyes -a x$have_valid_poti = xyes ; then
		AC_DEFINE(STARPU_HAVE_POTI, [1], [Define to 1 if you have libpoti and it is meant to be used])
		save_LIBS="$LIBS"
		LIBS="$LIBS $POTI_LIBS"
		AC_CHECK_FUNCS([poti_init_custom poti_user_NewEvent])
		LIBS="$save_LIBS"
		FXT_CFLAGS="$FXT_CFLAGS $POTI_CFLAGS"
		FXT_LIBS="$FXT_LIBS $POTI_LIBS"
	fi
fi

AC_MSG_CHECKING(whether additional locking systems FxT traces should be enabled)
AC_ARG_ENABLE(fxt-lock, [AS_HELP_STRING([--enable-fxt-lock],
			[enable additional locking systems FxT traces])],
			enable_fxt_lock=$enableval, enable_fxt_lock=no)
AC_MSG_RESULT($enable_fxt_lock)
if  test x$enable_fxt_lock = xyes; then
	AC_DEFINE(STARPU_FXT_LOCK_TRACES, [1], [enable additional locking systems FxT traces])
fi

AC_MSG_CHECKING(whether performance debugging should be enabled)
AC_ARG_ENABLE(perf-debug, [AS_HELP_STRING([--enable-perf-debug],
			[enable performance debugging through gprof])],
			enable_perf_debug=$enableval, enable_perf_debug=no)
AC_MSG_RESULT($enable_perf_debug)
AC_SUBST(STARPU_PERF_DEBUG, $enable_perf_debug)
if test x$enable_perf_debug = xyes; then
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

AC_MSG_CHECKING(whether memory stats should be displayed)
AC_ARG_ENABLE(memory-stats, [AS_HELP_STRING([--enable-memory-stats],
			     [enable memory stats])],
			     enable_memory_stats=$enableval, enable_memory_stats=no)
AC_MSG_RESULT($enable_memory_stats)
if test x$enable_memory_stats = xyes; then
        AC_DEFINE(STARPU_MEMORY_STATS, [1], [enable memory stats])
fi

AC_ARG_ENABLE(glpk, [AS_HELP_STRING([--disable-glpk],
			     [disable using glpk for bound computation])],
			     enable_glpk=$enableval, enable_glpk=yes)
if test x$enable_glpk = xyes; then
	AC_CHECK_HEADERS([glpk.h], [AC_DEFINE([STARPU_HAVE_GLPK_H], [1], [Define to 1 if you have the <glpk.h> header file.])])
	STARPU_HAVE_LIBRARY(GLPK, [glpk])
fi

AC_ARG_WITH(ayudame1-include-dir,
	[AS_HELP_STRING([--with-ayudame1-include-dir=<path>],
	[specify where Ayudame version 1 headers are installed])],
	[
		ayudame1_include_dir="$withval"
		if test -n "$ayudame1_include_dir"; then
			CPPFLAGS="-I$ayudame1_include_dir $CPPFLAGS"
		fi
	], [ayudame1_include_dir=no])
AC_ARG_WITH(ayudame2-include-dir,
	[AS_HELP_STRING([--with-ayudame2-include-dir=<path>],
	[specify where Ayudame version 2 headers are installed])],
	[
		ayudame2_include_dir="$withval"
		if test -n "$ayudame2_include_dir"; then
			CPPFLAGS="-I$ayudame2_include_dir $CPPFLAGS"
		fi
	], [ayudame2_include_dir=no])

# Ayudame 1 header is capitalized
AC_CHECK_HEADERS([Ayudame.h])
AC_ARG_ENABLE(ayudame1, [AS_HELP_STRING([--disable-ayudame1],
				   [Do not use Ayudame lib version 1])],
				   enable_ayudame1=$enableval, enable_ayudame1=yes)
# Ayudame 2 header is lowercase
AC_CHECK_HEADERS([ayudame.h])
AC_ARG_ENABLE(ayudame2, [AS_HELP_STRING([--disable-ayudame2],
				   [Do not use Ayudame lib version 2])],
				   enable_ayudame2=$enableval, enable_ayudame2=yes)
if test x$enable_ayudame1 = xyes -a x$ac_cv_header_Ayudame_h = xyes; then
   AC_DEFINE([STARPU_USE_AYUDAME1], [1], [Define to 1 if Ayudame 1 is available and should be used])
   ayu_msg="yes, use version 1"
else
   if test x$enable_ayudame2 = xyes -a x$ac_cv_header_ayudame_h = xyes; then
      AC_DEFINE([STARPU_USE_AYUDAME2], [1], [Define to 1 if Ayudame 2 is available and should be used])
      ayu_msg="yes, use version 2"
   else
      ayu_msg="no"
   fi
fi

AM_CONDITIONAL([STARPU_USE_AYUDAME1], [test "x$enable_ayudame1" = "xyes"])
AM_CONDITIONAL([STARPU_USE_AYUDAME2], [test "x$enable_ayudame2" = "xyes"])

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

AC_MSG_CHECKING(maximum number of memory nodes to use per MPI rank)
AC_ARG_ENABLE(maxnodes, [AS_HELP_STRING([--enable-maxnodes=<nnodes>],
			[maximum number of memory nodes per MPI rank])],
			maxnodes=$enableval, maxnodes=0)

if test x$maxnodes = x0 ; then
	if test x$enable_simgrid = xyes ; then
		# We need the room for the virtual CUDA/OpenCL devices
		nodes=`expr 4 + $nmaxcudadev + $nmaxopencldev + $nmaxmicdev + 1 + $nmaxmpidev`
		if test $nodes -gt 32 ; then
			nodes=32
		fi
	else
		# We have one memory node shared by all CPU workers, one node per GPU
		# and per MIC device
		# we add nodes to use 2 memory disks
		nodes=`expr $nmaxnumanodes + 2`
		if test x$enable_cuda = xyes ; then
			# we could have used nmaxcudadev + 1, but this would certainly give an
			# odd number.
			nodes=`expr $nodes + $nmaxcudadev`
		fi
		if test x$enable_opencl = xyes ; then
			# we could have used nmaxcudadev + 1, but this would certainly give an
			# odd number.
			nodes=`expr $nodes + $nmaxopencldev`
		fi
		if test x$enable_mic = xyes ; then
			nodes=`expr $nodes + $nmaxmicdev`
		fi
		if test x$enable_rcce = xyes ; then
			# Only 1 memory node for the shared memory.
			nodes=`expr $nodes + 1`
		fi

		#nmaxmpidev = 0 if mpi master-slave is disabled
		nodes=`expr $nodes + $nmaxmpidev`
 	fi

	# set maxnodes to the next power of 2 greater than nodes
	maxnodes=1
	while test "$maxnodes" -lt "$nodes"
	do
		maxnodes=`expr $maxnodes \* 2`
	done
fi
if test $maxnodes -gt 32 ; then
	# FIXME: at least use uint64 so we can have 64 memory nodes
	AC_MSG_ERROR([selected number of nodes ($maxnodes) can not be greater than 32])
fi

AC_MSG_CHECKING(maximum number of memory nodes)
AC_MSG_RESULT($maxnodes)
AC_DEFINE_UNQUOTED(STARPU_MAXNODES, [$maxnodes],
		[maximum number of memory nodes])


AC_MSG_CHECKING(whether allocation cache should be used)
AC_ARG_ENABLE(allocation-cache, [AS_HELP_STRING([--disable-allocation-cache],
			[disable data allocation cache])],
			enable_allocation_cache=$enableval, enable_allocation_cache=yes)
AC_MSG_RESULT($enable_allocation_cache)
if test x$enable_allocation_cache = xyes; then
	AC_DEFINE(STARPU_USE_ALLOCATION_CACHE, [1], [enable data allocation cache])
fi

AC_ARG_WITH(perf-model-dir, [AS_HELP_STRING([--with-perf-model-dir=<dir>], [specify where performance models should be stored])],
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
if test x$enable_simgrid != xyes; then
	if test x$enable_cpu != xyes; then
		maxcpus=0
	fi
	if test x$enable_cuda != xyes; then
		nmaxcudadev=0
	fi
	if test x$enable_opencl != xyes; then
		nmaxopencldev=0
	fi
	if test x$enable_mic != xyes; then
		nmaxmicthreads=0
	fi
    #By default, if we cannot build mpi master-slave nmaxmpidev is set to zero.
    #But with the multiplication with maxcpus, we need to put it to one.
    if test x$build_mpi_master_slave != xyes; then
        nmaxmpidev=1
    fi
fi
nmaxworkers=`expr 16 \* \( \( \( $nmaxmpidev \* $maxcpus \) + $nmaxcudadev + $nmaxopencldev + $nmaxmicthreads + 15 \) / 16 \) `
AC_MSG_CHECKING(Maximum number of workers)
AC_MSG_RESULT($nmaxworkers)
AC_DEFINE_UNQUOTED(STARPU_NMAXWORKERS, [$nmaxworkers], [Maximum number of workers])

# Computes the maximun number of combined worker
nmaxcombinedworkers=`expr $maxcpus + $nmaxmicthreads`
AC_MSG_CHECKING(Maximum number of workers combinations)
AC_MSG_RESULT($nmaxcombinedworkers)
AC_DEFINE_UNQUOTED(STARPU_NMAX_COMBINEDWORKERS,
	[$nmaxcombinedworkers], [Maximum number of worker combinations])



# Computes the maximum number of implementations per arch
AC_MSG_CHECKING(maximum number of implementations)
AC_ARG_ENABLE(maximplementations, [AS_HELP_STRING([--enable-maximplementations=<number>],
		[maximum number of implementations])],
		maximplementations=$enableval, maximplementations=4)
AC_MSG_RESULT($maximplementations)
AC_DEFINE_UNQUOTED(STARPU_MAXIMPLEMENTATIONS, [$maximplementations],
		[maximum number of implementations])

# Enable LevelDB support if requested and the lib is found
AC_ARG_ENABLE(leveldb, [AS_HELP_STRING([--enable-leveldb],
				   [Enable linking with LevelDB if available])],
				   enable_leveldb=$enableval, enable_leveldb=no)
if  test x$enable_leveldb = xyes; then
AC_LANG_PUSH([C++])
AC_CHECK_HEADERS([leveldb/db.h], [AC_DEFINE([STARPU_HAVE_LEVELDB], [1], [Define to 1 if you have the <leveldb/db.h> header file.])])
STARPU_HAVE_LIBRARY(LEVELDB, [leveldb])
AC_LANG_POP([C++])
fi
AM_CONDITIONAL(STARPU_HAVE_LEVELDB, test  "x$enable_leveldb" = "xyes" -a "x$ac_cv_lib_leveldb_main" = "xyes")

# Defines the calibration heuristic for the history-based calibration of StarPU
AC_MSG_CHECKING(calibration heuristic of history-based StarPU calibrator)
AC_ARG_ENABLE(calibration-heuristic, [AS_HELP_STRING([--enable-calibration-heuristic=<number>],
			[Define the maximum authorized deviation of StarPU history-based calibrator.])],
			calibration_heuristic=$enableval, calibration_heuristic=50)
AC_MSG_RESULT($calibration_heuristic)
AC_DEFINE_UNQUOTED(STARPU_HISTORYMAXERROR, [$calibration_heuristic], [calibration heuristic value])


###############################################################################
#                                                                             #
#                             MP Common settings                              #
#                                                                             #
###############################################################################

AM_CONDITIONAL([STARPU_USE_MP], [test "x$enable_mic" = "xyes" -o "x$build_mpi_master_slave" = "xyes" -o "x$enable_rcce" = "xyes"])

AC_ARG_ENABLE([export-dynamic], [AS_HELP_STRING([--disable-export-dynamic],
			  [Prevent the linker from adding all symbols to the dynamic symbol table])], [], [])

if test x$enable_mic = xyes -o x$build_mpi_master_slave = xyes -o x$enable_rcce = xyes ; then
	AC_DEFINE(STARPU_USE_MP, [1], [Message-passing SINKs support
		  is enabled])

	if test x$enable_export_dynamic != xno ; then
		STARPU_EXPORT_DYNAMIC="-rdynamic"
	fi
fi

AC_SUBST(STARPU_EXPORT_DYNAMIC)

# Computes the maximum number of different kernels a message-passing sink
# can lookup for and launch.
AC_MSG_CHECKING(Maximum number of message-passing kernels)
AC_ARG_ENABLE(maxmpkernels, [AS_HELP_STRING([
	      -enable-maxmpkernels=<number>],
	      [maximum number of kernels a message-passing sink can lookup
	      for and execute])],
	      maxmpkernels=$enableval, maxmpkernels=10)
AC_MSG_RESULT($maxmpkernels)
AC_DEFINE_UNQUOTED(STARPU_MAXMPKERNELS, [$maxmpkernels],
		[maximum number of message-passing kernels])

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
	CFLAGS="$1" # -we10006"

	AC_LINK_IFELSE(
		AC_LANG_PROGRAM(
			[[]],
			[[
				AC_LANG_SOURCE([
					const char *hello = "Hello World";
				])
			]]
		),
		[
			m4_default_nblank([$2], [GLOBAL_AM_CFLAGS="$GLOBAL_AM_CFLAGS $1"])
			AC_MSG_RESULT(yes)
			option_available=1
		],
		[
			AC_MSG_RESULT(no)
			option_available=0
		]
	)
	CFLAGS="$SAVED_CFLAGS"
])

IS_SUPPORTED_CFLAG(-Wall)
IS_SUPPORTED_CFLAG(-Werror=implicit)
IS_SUPPORTED_CFLAG(-Werror=implicit-function-declaration)
if test x$enable_perf_debug = xyes; then
	IS_SUPPORTED_CFLAG(-no-pie)
	IS_SUPPORTED_CFLAG(-no-PIE)
	IS_SUPPORTED_CFLAG(-fno-pie)
fi

if test "x$STARPU_DEVEL" != x; then
	AC_DEFINE(STARPU_DEVEL, [1], [enable developer warnings])
	IS_SUPPORTED_CFLAG(-Wextra)
	IS_SUPPORTED_CFLAG(-Wunused)
	IS_SUPPORTED_CFLAG(-Wundef)
	IS_SUPPORTED_CFLAG(-Wshadow)
	IS_SUPPORTED_CFLAG(-Werror=pointer-arith)
	IS_SUPPORTED_CFLAG(-fno-common)
	if test x$enable_debug = xyes; then
		IS_SUPPORTED_CFLAG(-fno-optimize-sibling-calls)
	fi
fi
AM_CONDITIONAL([STARPU_DEVEL],[test "x$STARPU_DEVEL" != x])

AC_SUBST(GLOBAL_AM_CFLAGS)

# Same value as Automake's, for use in other places.
pkglibdir="\${libdir}/$PACKAGE"
AC_SUBST([pkglibdir])

###############################################################################
#                                                                             #
#                               Fortran                                       #
#                                                                             #
###############################################################################

AC_ARG_ENABLE(fortran, [AS_HELP_STRING([--disable-fortran],
			[disable build of fortran examples])],
			enable_build_fortran_requested=$enableval, enable_build_fortran_requested=yes)
use_mpi_fort=no
enable_build_fortran=no
if test "x$enable_build_fortran_requested" = "xyes" ; then
   if test "x$FC" != "x"; then
   	if $FC --version|grep -q 'GNU Fortran'; then
		 AC_LANG_PUSH([Fortran])
		 OLD_FCFLAGS="$FCFLAGS"
		 FCFLAGS="$FCFLAGS -cpp"
		 AC_COMPILE_IFELSE([AC_LANG_PROGRAM([], [[
#if __GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ < 9)
#error GFortran too old, version >= 4.9.x needed, Fortran examples will not be built
#endif
]]
                 )],
                 [enable_build_fortran="yes"],
                 [enable_build_fortran="no"])
		 FCFLAGS="$OLD_FCFLAGS"
		 AC_LANG_POP([Fortran])
                 if test "$enable_build_fortran" = "no" ; then
                   AC_MSG_WARN([GFortran too old, version >= 4.9.x needed, Fortran examples will not be built])
                 fi
	else
		if $FC -V 2>&1|grep -q 'Intel(R) Fortran'; then
			enable_build_fortran="yes"
			ifort_fc_version=`$FC -V 2>&1 |head -1|sed 's/.*Version //;s/ Build.*//'`
			ifort_maj_version=`echo $ifort_fc_version|cut -d. -f1`

			if test $ifort_maj_version -lt 16; then
				AC_MSG_WARN([Intel Fortran compiler $ifort_fc_version too old, version >= 2016.x needed, Fortran examples will not be built])
				enable_build_fortran="no"
			fi
		else
			if $FC -qversion 2>&1|grep -q 'IBM XL Fortran'; then
				xlf_fc_version=`$FC -V 2>&1 |tail -1|sed 's/.*Version: //'`
				xlf_maj_version=`echo $xlf_fc_version|cut -d. -f1`

				AC_MSG_WARN([IBM Fortran compiler $xlf_fc_version not validated with the native StarPU Fortran API, Fortran examples will not be built])
				enable_build_fortran="no"
			else
				AC_MSG_WARN(Fortran compiler has not been tested for StarPU native Fortran support)
				 enable_build_fortran="yes"
			fi
		fi
	fi
	if test "x$enable_build_fortran" = "xyes" ; then
		AC_DEFINE(STARPU_HAVE_FC, [1], [Define this if a Fortran compiler is available])
		if test x$build_mpi_lib = xyes -o x$build_nmad_lib = xyes -o x$build_mpi_master_slave = xyes ; then
			#Check MPIFORT
			if test x$enable_simgrid = xyes ; then
				DEFAULT_MPIFORT=smpifort
			else
				DEFAULT_MPIFORT=mpifort
			fi
			AC_ARG_WITH(mpifort, [AS_HELP_STRING([--with-mpifort=<mpifort name or path to mpifort>], [Name or path of the mpifort compiler])], [DEFAULT_MPIFORT=$withval])
			case $DEFAULT_MPIFORT in
				/*) mpifort_path="$DEFAULT_MPIFORT" ;;
				*)  AC_PATH_PROG(mpifort_path, $DEFAULT_MPIFORT, [no], [$simgrid_dir/bin:$PATH]) ;;
			esac
			# We test if the MPIFORT compiler exists
			if test ! -x $mpifort_path; then
				AC_MSG_RESULT(The mpifort compiler '$mpifort_path' does not have the execute permission)
				mpifort_path=no
			else
				OLD_CC=$CC
				CC=$mpicc_path
				AC_LINK_IFELSE(
						AC_LANG_PROGRAM(
							[[#include <mpi.h>]],
							[[AC_LANG_SOURCE([return MPI_Comm_f2c(0);])]]
							),
						[use_mpi_fort=yes],
						[use_mpi_fort=no]
						)
				CC=$OLD_CC
				if test "x$use_mpi_fort" = xyes; then
					AC_DEFINE([HAVE_MPI_COMM_F2C], [1], [Function MPI_Comm_f2c is available])
				fi
			fi

			AC_MSG_CHECKING(whether mpifort is available)
			AC_MSG_RESULT($mpifort_path)
			AC_SUBST(MPIFORT, $mpifort_path)

			if test x$mpifort_path != xno ; then
				MPIPATH=$(dirname $mpifort_path):$PATH
			else
				MPIPATH=$PATH
			fi
		fi
	fi
   fi
fi
if test "x$enable_build_fortran" = "xyes" ; then
   if test "x$FC" = "x" ; then
      enable_build_fortran="no"
   fi
fi

#We have MPI C/C++ compiler
if test x$build_mpi_master_slave = xyes; then
    #Check if we can compile fortran cases
    if test x$use_mpi_fort = xyes ; then
        F77LD=$mpifort_path
        FCLD=$mpifort_path
        F77=$mpifort_path
        FC=$mpifort_path
    else
        enable_build_fortran=no
    fi
fi


AM_CONDITIONAL([STARPU_HAVE_FC], [test "x$FC" != "x" -a "x$enable_build_fortran" = "xyes"])
AM_CONDITIONAL([STARPU_HAVE_F77], [test "x$F77" != "x" -a "x$enable_build_fortran" = "xyes"])
AM_CONDITIONAL([STARPU_HAVE_MPIFORT], [test "x$use_mpi_fort" = "xyes"])

########################################################################
#                                                                      #
#                            Cluster support                           #
#                                                                      #
########################################################################

AC_ARG_ENABLE(cluster, [AS_HELP_STRING([--enable-cluster], [build the cluster support])],
			enable_cluster=$enableval, enable_cluster=no)

AC_MSG_CHECKING(for cluster support)

if test x$enable_cluster = xyes; then
	AC_DEFINE(STARPU_CLUSTER, [1], [Define this to enable cluster support])
	AC_OPENMP
fi

AM_CONDITIONAL([STARPU_CLUSTER], [test "x$enable_cluster" = "xyes"])
AC_MSG_RESULT($enable_cluster)

###############################################################################
#                                                                             #
#                            OpenMP runtime support                           #
#                                                                             #
###############################################################################

AC_ARG_ENABLE(openmp, [AS_HELP_STRING([--enable-openmp],
			[build the OpenMP runtime support])],
			enable_openmp=$enableval, enable_openmp=no)

openmp_msg=""
if test x$starpu_windows = xyes ; then
   enable_openmp=no
   openmp_msg="disabled on windows"
fi
if test "x$use_mpi_master_slave" = "xyes" ; then
   enable_openmp=no
   openmp_msg="incompatibility with MPI master slave support"
fi

if test x$enable_openmp = xyes; then
	AC_DEFINE(STARPU_OPENMP, [1], [Define this to enable OpenMP runtime support])
fi

AC_MSG_CHECKING(for OpenMP runtime support)
AM_CONDITIONAL([STARPU_OPENMP], [test "x$enable_openmp" = "xyes"])
AC_MSG_RESULT($enable_openmp $openmp_msg)

if test x$enable_simgrid = xno; then
  if test -n "{OPENMP_CFLAGS}"; then
     CFLAGS="${CFLAGS} ${OPENMP_CFLAGS}"
     STARPU_OMP_LDFLAGS="${OPENMP_CFLAGS}"
  fi
fi
AM_CONDITIONAL([STARPU_HAVE_OPENMP],[test x$enable_simgrid = xno -a -n "$OPENMP_CFLAGS"])

###############################################################################
#                                                                             #
#                               SOCL interface                                #
#                                                                             #
###############################################################################

AC_ARG_ENABLE([socl],
  [AS_HELP_STRING([--enable-socl],
    [build the OpenCL interface (experimental)])],
  [enable_socl="$enableval"],
  [enable_socl="maybe"])

AC_MSG_CHECKING(for SOCL)

# in case SOCL was explicitely required, but is not available, this is an error
if test "x$enable_socl" = "xyes" -a "$have_valid_opencl" = "no" ; then
    AC_MSG_ERROR([SOCL cannot be enabled without OpenCL])
fi

# MPI Master Slave and SOCL are not compatible
if test "x$use_mpi_master_slave" = "xyes" ; then
   if test "x$enable_socl" = "xyes" ; then
      AC_MSG_ERROR([MPI Master-Slave and SOCL can not be used at the same time !])
   fi
   if test "x$enable_socl" = "xmaybe" ; then
     enable_socl=no
   fi
fi

# now we enable SOCL if and only if a proper setup is available
if test "x$enable_socl" = "xyes" -o "x$enable_socl" = "xmaybe" ; then
   build_socl=$have_valid_opencl
else
   build_socl=no
fi

AC_MSG_RESULT($build_socl)
AM_CONDITIONAL([STARPU_BUILD_SOCL], [test "x$build_socl" = "xyes"])
AM_CONDITIONAL([STARPU_USE_SOCL], [test "x$build_socl" = "xyes"])

if test "$build_socl" = "yes" ; then
   AC_CHECK_FUNCS([clGetExtensionFunctionAddressForPlatform])
   if test -n "$SOCL_OCL_LIB_OPENCL" -a -f "$SOCL_OCL_LIB_OPENCL" ; then
      run_socl_check=yes
      SOCL_OCL_LIB_OPENCL_DIR=$(dirname $SOCL_OCL_LIB_OPENCL)
      AC_SUBST(SOCL_OCL_LIB_OPENCL_DIR)
   else
      run_socl_check=no
   fi
else
   run_socl_check=no
fi
###############################################################################
#                                                                             #
#                                 Debugging                                   #
#                                                                             #
###############################################################################

AC_PATH_PROG([GDB], [gdb], [not-found])
if test "x$GDB" != "xnot-found"; then
   AC_DEFINE_UNQUOTED([STARPU_GDB_PATH], ["$GDB"],
     [Path to the GNU debugger.])
fi

###############################################################################
#                                                                             #
#                                  Examples                                   #
#                                                                             #
###############################################################################

AC_ARG_ENABLE(build-tests, [AS_HELP_STRING([--disable-build-tests],
			[disable building of tests])],
			enable_build_tests=$enableval, enable_build_tests=yes)
# check stuff for tests (todo)
AM_CONDITIONAL(STARPU_BUILD_TESTS, [test x$enable_build_tests != xno])
AC_ARG_ENABLE(build-examples, [AS_HELP_STRING([--disable-build-examples],
			[disable building of examples])],
			enable_build_examples=$enableval, enable_build_examples=yes)
# check stuff for examples (todo)
AM_CONDITIONAL(STARPU_BUILD_EXAMPLES, [test x$enable_build_examples != xno])
AC_ARG_ENABLE(opengl-render, [AS_HELP_STRING([--enable-opengl-render],
			[enable OpenGL rendering of some examples])],
			enable_opengl_render=$enableval, enable_opengl_render=no)

if test x$enable_opengl_render = xyes; then
	STARPU_CHECK_LIB(OPENGL_RENDER, glut, glutInit,,AC_MSG_ERROR([cannot find glut]))
	STARPU_CHECK_LIB(OPENGL_RENDER, GL, glXCreateContext,,AC_MSG_ERROR([cannot find GL]))
	STARPU_CHECK_LIB(OPENGL_RENDER, GLU, gluLookAt,,AC_MSG_ERROR([cannot find GLU]))

	AC_DEFINE(STARPU_OPENGL_RENDER, [1], [enable OpenGL rendering of some examples])
fi

AC_MSG_CHECKING(whether OpenGL rendering is enabled)
AC_SUBST(STARPU_OPENGL_RENDER, $enable_opengl_render)
AC_MSG_RESULT($enable_opengl_render)
AM_CONDITIONAL([STARPU_HAVE_OPENGL], [test "x$enable_opengl_render" = xyes])

AC_PATH_XTRA
if test "x$no_x" != "xyes"; then
	AC_DEFINE(STARPU_HAVE_X11, [1], [enable X11])
fi
AM_CONDITIONAL([STARPU_HAVE_X11], [test "x$no_x" != "xyes"])

# In case there are BLAS kernels that are used by the example applications
# we may specify which library to use. Note that this is not used for StarPU
# itself.

blas_lib=maybe
AC_ARG_ENABLE(blas-lib,
 [  --enable-blas-lib[=blaslibname]:
                      none [default]: no BLAS lib is used
                      atlas: use ATLAS library
                      goto: use GotoBLAS library
		      mkl: use MKL library (you may need to set specific CFLAGS and LDFLAGS with --with-mkl-cflags and --with-mkl-ldflags)],
 [
     if   test "x$enableval" = "xatlas" ; then
        blas_lib=atlas
     elif test "x$enableval" = "xgoto" ; then
        blas_lib=goto
     elif test "x$enableval" = "xopenblas" ; then
        blas_lib=openblas
     elif test "x$enableval" = "xnone" ; then
        blas_lib=none
     elif test "x$enableval" = "xmkl" ; then
        blas_lib=mkl
     elif test "x$enableval" = "xarmpl" ; then
        blas_lib=armpl
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
       STARPU_CHECK_LIB(BLAS, gfortran, main,,)
       STARPU_CHECK_LIB(BLAS, ifcore, main,,)
       # Perhaps that GotoBLAS2 is available instead (so that we have libgotoblas2.{so,a})
       STARPU_CHECK_LIB(BLAS, goto2, sgemm_,, [havegoto2=no], [$STARPU_BLAS_LDFLAGS])
       if test x$havegoto2 = xno; then
	   STARPU_CHECK_LIB(BLAS, goto, sgemm_,,AC_MSG_ERROR([cannot find goto lib]), [$STARPU_BLAS_LDFLAGS])
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
	STARPU_CHECK_LIB(BLAS, atlas, ATL_sgemm,,AC_MSG_ERROR([cannot find atlas lib]),)
	STARPU_CHECK_LIB(BLAS, cblas, cblas_sgemm,,AC_MSG_ERROR([cannot find atlas lib]),[-latlas])
	AC_DEFINE(STARPU_ATLAS, [1], [use STARPU_ATLAS library])
    fi
fi

if test x$blas_lib = xmaybe -o x$blas_lib = xopenblas; then
    PKG_CHECK_MODULES([OPENBLAS], [openblas],
    				  [PKG_CHECK_MODULES([BLAS_OPENBLAS], [blas-openblas],
				  				      [AC_DEFINE([STARPU_OPENBLAS], [1], [Define to 1 if you use the openblas library.])
					          		       AC_SUBST([STARPU_OPENBLAS], [1])
						          	       CFLAGS="${CFLAGS} ${OPENBLAS_CFLAGS} ${BLAS_OPENBLAS_CFLAGS} "
							               LIBS="${LIBS} ${OPENBLAS_LIBS} ${BLAS_OPENBLAS_LIBS} "
								       blas_lib=openblas
      								      ],
								      [ if test x$blas_lib = xopenblas; then
								      	   STARPU_CHECK_LIB(OPENBLAS, blas-openblas, cblas_sgemm,,AC_MSG_ERROR([cannot find blas-openblas lib]),[-lblas-openblas])
				  				      	   AC_DEFINE([STARPU_OPENBLAS], [1], [Define to 1 if you use the openblas library.])
					          		       	   AC_SUBST([STARPU_OPENBLAS], [1])
									fi
				  				      ])
				  ],
				  [ if test x$blas_lib = xopenblas; then
				       STARPU_CHECK_LIB(OPENBLAS, openblas, cblas_sgemm,,AC_MSG_ERROR([cannot find openblas lib]),[-lopenblas])
				       AC_DEFINE([STARPU_OPENBLAS], [1], [Define to 1 if you use the openblas library.])
				       AC_SUBST([STARPU_OPENBLAS], [1])
      				    fi
                                  ] )
fi

if test x$blas_lib = xmaybe -o x$blas_lib = xmkl; then
    # Should we use MKL ?
    if test -n "$MKLROOT" ; then
        CPPFLAGS="${CPPFLAGS} -I$MKLROOT/include"
	case $host_vendor in
	    *1om) mkl_plat=mic ;;
	    *)    mkl_plat=intel64 ;;
	esac
	SAVED_LIBS=$LIBS
	STARPU_BLAS_LDFLAGS="-L$MKLROOT/lib/$mkl_plat -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lm -lpthread -ldl"
	LIBS="$LIBS $STARPU_BLAS_LDFLAGS"
	AC_LINK_IFELSE(
		[AC_LANG_PROGRAM([[
			#include <mkl.h>
		]], [[ ]])],
		[ blas_lib=mkl ],
		[ STARPU_BLAS_LDFLAGS="" ],
	)
	LIBS=$SAVED_LIBS
    fi
    AC_ARG_WITH(mkl-cflags, [AS_HELP_STRING([--with-mkl-cflags], [specify MKL compilation flags])],
	[
        CPPFLAGS="${CPPFLAGS} $withval"
	blas_lib=mkl
	])

    AC_ARG_WITH(mkl-ldflags, [AS_HELP_STRING([--with-mkl-ldflags], [specify MKL linking flags])],
	[
	STARPU_BLAS_LDFLAGS="$withval"
	blas_lib=mkl
	])
    if test x$blas_lib = xmkl; then
	AC_DEFINE(STARPU_MKL, [1], [use MKL library])
    fi
fi

if test x$blas_lib = xmaybe -o x$blas_lib = xarmpl; then
    # Should we use ARMPL ?
    if test -n "$ARMPL_DIR" ; then
        CPPFLAGS="${CPPFLAGS} -I$ARMPL_INCLUDES"
	SAVED_LIBS=$LIBS
	STARPU_BLAS_LDFLAGS="-L$ARMPL_LIBRARIES -larmpl_lp64 -lgfortran -lm"
	LIBS="$LIBS $STARPU_BLAS_LDFLAGS"
	AC_LINK_IFELSE(
		[AC_LANG_PROGRAM([[
			#include <armpl.h>
		]], [[ ]])],
		[ blas_lib=armpl ],
		[ STARPU_BLAS_LDFLAGS="" ],
	)
	LIBS=$SAVED_LIBS
    fi
    AC_ARG_WITH(armpl-cflags, [AS_HELP_STRING([--with-armpl-cflags], [specify ARMPL compilation flags])],
	[
        CPPFLAGS="${CPPFLAGS} $withval"
	blas_lib=armpl
	])

    AC_ARG_WITH(armpl-ldflags, [AS_HELP_STRING([--with-armpl-ldflags], [specify ARMPL linking flags])],
	[
	STARPU_BLAS_LDFLAGS="$withval"
	blas_lib=armpl
	])
    if test x$blas_lib = xarmpl; then
	AC_DEFINE(STARPU_ARMPL, [1], [use ARMPL library])
    fi
fi

if test x$blas_lib = xmaybe; then
    #perhaps it is possible to use some BLAS lib from the system
    use_system_blas=no
    STARPU_SEARCH_LIBS(BLAS,[sgemm_],[blas],use_system_blas=yes,,)
    if test x$use_system_blas = xyes; then
        AC_DEFINE(STARPU_SYSTEM_BLAS, [1], [use refblas library])
	blas_lib=system
    elif test x"$BLAS_LIBS" != x; then
        AC_DEFINE(STARPU_SYSTEM_BLAS, [1], [use user defined library])
        STARPU_BLAS_LDFLAGS="$BLAS_LIBS"
        blas_lib=system
        AC_ARG_VAR([BLAS_LIBS], [linker flags for blas])
    else
	blas_lib=none
    fi
fi

if test x$blas_lib = xsystem; then
    AC_CHECK_HEADER([cblas.h], [have_cblas_h=yes], [have_cblas_h=no])
fi
AM_CONDITIONAL(STARPU_HAVE_CBLAS_H, test x$have_cblas_h = xyes)
if test x$have_cblas_h = xyes; then
    AC_DEFINE(STARPU_HAVE_CBLAS_H, [1], [The blas library has blas.h])
fi

AM_CONDITIONAL(STARPU_ATLAS_BLAS_LIB, test x$blas_lib = xatlas)
AM_CONDITIONAL(STARPU_GOTO_BLAS_LIB, test x$blas_lib = xgoto)
AM_CONDITIONAL(STARPU_MKL_BLAS_LIB, test x$blas_lib = xmkl)
AM_CONDITIONAL(STARPU_SYSTEM_BLAS_LIB, test x$blas_lib = xsystem)
AM_CONDITIONAL(STARPU_NO_BLAS_LIB, test x$blas_lib = xnone -a x$enable_simgrid = xno)
AC_SUBST(STARPU_BLAS_LDFLAGS)

AC_MSG_CHECKING(which BLAS lib should be used)
AC_MSG_RESULT($blas_lib)
AC_SUBST(BLAS_LIB,$blas_lib)

###############################################################################
#                                                                             #
#			 Multiple linear regression			      #
#                                                                             #
###############################################################################
AC_ARG_ENABLE(mlr, [AS_HELP_STRING([--enable-mlr],
			[Enable multiple linear regression models])],
			enable_mlr=$enableval, enable_mlr=no)
AC_ARG_ENABLE(mlr-system-blas, [AS_HELP_STRING([--enable-mlr-system-blas],
			[Make the multiple linear regression models use the system BLAS instead of min-dgels])],
			enable_mlr_blas=$enableval, enable_mlr_blas=no)

AC_MSG_CHECKING(whether multiple linear regression models are disabled)
if test x$enable_mlr = xyes -a "$starpu_windows" != "yes" ; then
   	AC_MSG_RESULT(no)
	install_min_dgels=no
	support_mlr=yes
   	STARPU_SEARCH_LIBS(LAPACK,[dgels_],[lapack],use_system_lapack=yes,,)
	if test x$blas_lib = xnone ; then
	   use_system_lapack=no
	fi
	if test x$enable_mlr_blas = xyes -a x$use_system_lapack = xyes; then
	   	AC_DEFINE(STARPU_MLR_MODEL, [1], [use reflapack library])
		LDFLAGS="-llapack $LDFLAGS"
	else
		if test x$enable_mlr_blas=xyes -a x$blas_lib = xmkl; then
		   	AC_DEFINE(STARPU_MLR_MODEL, [1], [use mkl library])
		else
			AC_MSG_CHECKING(whether min-dgels is linked)
			if test x"$DGELS_LIBS" != x; then
		   	   	AC_MSG_RESULT(yes)
        		   	AC_DEFINE(STARPU_MLR_MODEL, [1], [use user defined library])
			   	AC_ARG_VAR([DGELS_LIBS], [linker flags for lapack dgels])
			else
				AC_MSG_RESULT(no)
				AC_MSG_CHECKING(min-dgels source)
				if test "${cross_compiling}" != "no" ; then
					# Cross-compiling is not supported by min-dgels
					AC_MSG_RESULT(no)
					install_min_dgels=no
					support_mlr=no
				else
					AC_MSG_RESULT(yes)
					DGELS_LIBS="-Wl,--start-group $STARPU_BUILD_DIR/min-dgels/build/minlibblas.a $STARPU_BUILD_DIR/min-dgels/build/minlibdgels.a $STARPU_BUILD_DIR/min-dgels/build/minlibf2c.a -Wl,--end-group"
					AC_DEFINE(STARPU_MLR_MODEL, [1], [use user defined library])
					AC_DEFINE(STARPU_BUILT_IN_MIN_DGELS, [1], [use built-in min_dgels])
					AC_ARG_VAR([DGELS_LIBS], [linker flags for lapack dgels])
					install_min_dgels=yes
				fi
			fi
		fi
	fi
else
 	AC_MSG_RESULT(yes)
	install_min_dgels=no
	support_mlr=no
fi
AM_CONDITIONAL(STARPU_USE_MIN_DGELS, test x$install_min_dgels = xyes)

##########################################
# FFT                                    #
##########################################

have_fftw=no
have_fftwf=no
have_fftwl=no
fft_support=no

AC_ARG_ENABLE(starpufft, [AS_HELP_STRING([--disable-starpufft],
			[Disable build of StarPU-FFT])],
			enable_starpufft=$enableval,enable_starpufft=yes)

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

if test x$enable_starpufft = xyes -a \( \( x$enable_cpu = xyes -a x$have_fftw = xyes -a x$have_fftwf = xyes \) -o x$have_cufftdoublecomplex = xyes \); then
   fft_support=yes
fi
AM_CONDITIONAL(STARPU_BUILD_STARPUFFT, [test x$fft_support = xyes])

AC_ARG_ENABLE(starpufft-examples, [AS_HELP_STRING([--enable-starpufft-examples],
			[enable build of StarPU FFT examples])],
			enable_starpufft_examples=$enableval, enable_starpufft_examples=no)
AM_CONDITIONAL(STARPU_BUILD_STARPUFFT_EXAMPLES, [test x$enable_starpufft_examples = xyes])

##########################################
# hwloc                                  #
##########################################

have_valid_hwloc=no
SAVED_LDFLAGS="${LDFLAGS}"
SAVED_CPPFLAGS="${CPPFLAGS}"
SAVED_PKG_CONFIG_PATH="$PKG_CONFIG_PATH"
AC_ARG_WITH([hwloc],
	[AS_HELP_STRING([--without-hwloc], [Disable hwloc (enabled by default)])],
	[
		if test x$withval != xno; then
			if test "$withval" = "yes" ; then
				use_hwloc=yes
			else
				# use specified path
				if test ! -d "$withval" ; then
				   AC_MSG_ERROR("Directory specified for hwloc <$withval> does not exist")
				fi
				if test ! -d "$withval/lib/pkgconfig" ; then
				   AC_MSG_ERROR("Hwloc directory <$withval> does not have a subdirectory lib/pkgconfig")
				fi
				export PKG_CONFIG_PATH=$withval/lib/pkgconfig:$PKG_CONFIG_PATH
				use_hwloc=yes
			fi
		else
			use_hwloc=no
		fi
	],
	[
		use_hwloc=maybe
	])
AS_IF([test "$use_hwloc" != "no"],
      [PKG_CHECK_MODULES([HWLOC],[hwloc], [have_valid_hwloc=yes], [have_valid_hwloc=no])]
     )
AM_CONDITIONAL(STARPU_HAVE_HWLOC, test "x$have_valid_hwloc" = "xyes")
# in case hwloc was explicitely required, but is not available, this is an error
AS_IF([test "$use_hwloc" = "yes" -a "$have_valid_hwloc" = "no"],
      [AC_MSG_ERROR([cannot find hwloc or pkg-config])]
     )
# in case hwloc is not available but was not explicitely disabled, this is an error
AS_IF([test "$have_valid_hwloc" = "no" -a "$use_hwloc" != "no"],
      [AC_MSG_ERROR([libhwloc or pkg-config was not found on your system. If the target machine is hyperthreaded the performance may be impacted a lot.  It is strongly recommended to install libhwloc and pkg-config. However, if you really want to use StarPU without enabling libhwloc, please restart configure by specifying the option '--without-hwloc'.])]
     )

LDFLAGS="${HWLOC_LIBS} ${SAVED_LDFLAGS}"
CPPFLAGS="${HWLOC_CFLAGS} ${SAVED_CPPFLAGS}"

AS_IF([test "$have_valid_hwloc" = "yes"],
      [AC_DEFINE([STARPU_HAVE_HWLOC], [1], [Define to 1 if you have the hwloc library.])
       HWLOC_REQUIRES=hwloc
       AC_SUBST([STARPU_HAVE_HWLOC], [1])
       AC_CHECK_DECLS([hwloc_cuda_get_device_osdev_by_index], [], [], [[#include <hwloc/cuda.h>]])
      ])

AC_CHECK_FUNCS([hwloc_topology_dup])
AC_CHECK_FUNCS([hwloc_topology_set_components])
AM_CONDITIONAL(STARPU_HWLOC_HAVE_TOPOLOGY_DUP, test $ac_cv_func_hwloc_topology_dup = yes)

LDFLAGS="${SAVED_LDFLAGS}"
CPPFLAGS="${SAVED_CPPFLAGS}"
export PKG_CONFIG_PATH=$SAVED_PKG_CONFIG_PATH

AC_MSG_CHECKING(whether hwloc should be used)
AC_MSG_RESULT($have_valid_hwloc)
AC_SUBST(HWLOC_REQUIRES)

# is the header file f77.h available ?
AC_CHECK_HEADER([f77.h], [have_f77_h=yes], [have_f77_h=no])
AC_SUBST(STARPU_HAVE_F77_H, $have_f77_h)
AM_CONDITIONAL(STARPU_HAVE_F77_H, test x$have_f77_h = xyes)
if test x$have_f77_h = xyes; then
        AC_DEFINE([STARPU_HAVE_F77_H], [1], [Define to 1 if you have the <f77.h> header file.])
fi

AC_ARG_ENABLE(icc, [AS_HELP_STRING([--enable-icc],
				[Enable the compilation of specific ICC examples])],
				enable_icc=$enableval, enable_icc=yes)

ICC=""
if test "$enable_icc" = "yes" ; then
   # Check if icc is available
   AC_CHECK_PROGS([ICC], [icc])
fi

# If cuda and icc are both available, check they are compatible
if test "$enable_cuda" = "yes" -a "$ICC" != ""; then
   AC_MSG_CHECKING(whether CUDA and ICC are compatible)
   OLD_CC="$CC"
   CC="$ICC"
   OLD_CFLAGS="$CFLAGS"
   CFLAGS="-I$PWD/include -I$srcdir/include"
   AC_COMPILE_IFELSE(
       [AC_LANG_PROGRAM(
	   [[#include <cuda.h>
	   #include <starpu.h>]],
	   [[]]
	   )],
       AC_MSG_RESULT(yes),
       [ICC=""
           AC_MSG_RESULT(no)]
   )
   CC="$OLD_CC"
   CFLAGS="$OLD_CFLAGS"
fi

# Disable ICC on windows
if test "x$ICC" != "x" -a "$starpu_windows" = "yes" ; then
    ICC=""
fi

if test "x$ICC" != "x"; then
  AC_DEFINE(STARPU_HAVE_ICC, [1], [Define this if icc is available])
fi
AM_CONDITIONAL([STARPU_HAVE_ICC], [test "x$ICC" != "x"])

# Do not generate manpages for the tools if we do not have help2man
AC_CHECK_PROGS([HELP2MAN], [help2man])
# Disable on windows
if test "$starpu_windows" = "yes" ; then
    HELP2MAN=""
fi
AM_CONDITIONAL([STARPU_HAVE_HELP2MAN], [test "x$HELP2MAN" != "x"])

AC_CHECK_MEMBER([struct cudaDeviceProp.pciDomainID],
  AC_DEFINE([STARPU_HAVE_DOMAINID],[1],[Define to 1 if CUDA device properties include DomainID]),
  , [[#include <cuda_runtime_api.h>]])

AC_CHECK_MEMBER([struct cudaDeviceProp.pciBusID],
  AC_DEFINE([STARPU_HAVE_BUSID],[1],[Define to 1 if CUDA device properties include BusID]),
  , [[#include <cuda_runtime_api.h>]])

dnl Set this condition when Automake 1.11 or later is being used.
dnl Automake 1.11 introduced `silent-rules', hence the check.
m4_ifdef([AM_SILENT_RULES],
  AM_CONDITIONAL([STARPU_HAVE_AM111], [true]),
  AM_CONDITIONAL([STARPU_HAVE_AM111], [false]))

##########################################
# Resource Manager                       #
##########################################

starpurm_support=no
starpurm_dlb_support=no

AC_ARG_ENABLE(starpurm, [AS_HELP_STRING([--enable-starpurm], [enable resource management support])],
                    enable_starpurm=$enableval, enable_starpurm=no)
if test "x$enable_starpurm" != xno
then
	starpurm_support=yes

	AC_MSG_CHECKING(whether resource management debug messages should be displayed)
	AC_ARG_ENABLE(starpurm-verbose, [AS_HELP_STRING([--enable-starpurm-verbose],
		      [display resource management verbose debug messages])],
		      enable_starpurm_verbose=$enableval, enable_starpurm_verbose=no)
	AC_MSG_RESULT($enable_starpurm_verbose)
	if test x$enable_starpurm_verbose = xyes; then
		AC_DEFINE(STARPURM_VERBOSE, [1], [display resource management verbose debug messages])
	fi

	# DLB
	DLB_CFLAGS=""
	DLB_LIBS=""
	AC_ARG_ENABLE(dlb, [AS_HELP_STRING([--enable-dlb], [enable DLB support])],
	                    enable_dlb=$enableval, enable_dlb=no)
	if test "x$enable_dlb" != xno
	then
		AC_ARG_WITH(dlb-include-dir,
			[AS_HELP_STRING([--with-dlb-include-dir=<path>],
			[specify where DLB headers are installed])],
			[dlb_inc_dirs="$withval"], [dlb_inc_dirs=""])

		dlb_inc_dirs="${dlb_inc_dirs} /usr/include/dlb"

		dlb_incdir_found=no
		for dlb_incdir in $dlb_inc_dirs
		do
			if test -n "$dlb_incdir"
			then
				SAVED_CPPFLAGS="${CPPFLAGS}"
				CPPFLAGS=-I${dlb_incdir}
				AC_CHECK_HEADERS([dlb.h])
				if test "$ac_cv_header_dlb_h" = "yes"
				then
					CPPFLAGS="-I$dlb_incdir ${SAVED_CPPFLAGS}"
					DLB_CFLAGS="-I${dlb_incdir}"
					dlb_incdir_found=yes
					break
				else
					CPPFLAGS=${SAVED_CPPFLAGS}
				fi
				unset ac_cv_header_dlb_h
			fi
		done

		AC_ARG_WITH(dlb-lib-dir,
			[AS_HELP_STRING([--with-dlb-lib-dir=<path>],
			[specify where DLB libraries are installed])],
			[dlb_lib_dirs="$withval"], [dlb_lib_dirs=""])

		dlb_lib_dirs="${dlb_lib_dirs} /usr/lib/dlb"

		dlb_libdir_found=no
		for dlb_libdir in $dlb_lib_dirs
		do
			if test -n "$dlb_libdir"
			then
				SAVED_LDFLAGS="${LDFLAGS}"
				LDFLAGS=-L${dlb_libdir}
				AC_CHECK_LIB(dlb, [DLB_Init])
				if test "$ac_cv_lib_dlb_DLB_Init" = "yes"
				then
					LDFLAGS="-L${dlb_libdir} ${SAVED_LDFLAGS} ${STARPU_DLB_LDFLAGS}"
					DLB_LIBS="-L${dlb_libdir} -ldlb"
					dlb_libdir_found=yes
					break
				else
					LDFLAGS=${SAVED_LDFLAGS}
				fi
				unset ac_cv_lib_dlb_DLB_Init
			fi
		done

		SAVED_CPPFLAGS="${CPPFLAGS}"
		SAVED_CFLAGS="${CFLAGS}"
		SAVED_LDFLAGS="${LDFLAGS}"
		CPPFLAGS="$HWLOC_CPPFLAGS -D_GNU_SOURCE $CPPFLAGS"
		CFLAGS="$HWLOC_CFLAGS $CFLAGS"
		LIBS="$HWLOC_LIBS $LIBS"
		# check whether libhwloc has a dedicated glibc-sched.h include for conversion with glibc cpusets
		AC_CHECK_HEADERS([hwloc/glibc-sched.h])
		CPPFLAGS="$SAVED_CPPFLAGS"
		CFLAGS="$SAVED_CFLAGS"
		LIBS="$SAVED_LIBS"

		SAVED_CPPFLAGS="${CPPFLAGS}"
		SAVED_CFLAGS="${CFLAGS}"
		SAVED_LDFLAGS="${LDFLAGS}"
		CPPFLAGS="$STARPU_CPPFLAGS $CPPFLAGS"
		CFLAGS="$STARPU_CFLAGS $CFLAGS"
		LIBS="$STARPU_LIBS $LIBS"
		# check if StarPU implements starpu_worker_set_going_to_sleep_callback()
		if test x$enable_worker_cb = xyes ; then
			AC_DEFINE([STARPURM_STARPU_HAVE_WORKER_CALLBACKS], [1], [Define to 1 if StarPU has support for worker callbacks.])
		fi

		#AC_CHECK_FUNC([starpu_worker_set_going_to_sleep_callback],AC_DEFINE([STARPURM_STARPU_HAVE_WORKER_CALLBACKS], [1], [Define to 1 if StarPU has support for worker callbacks.]))
		CPPFLAGS="$SAVED_CPPFLAGS"
		CFLAGS="$SAVED_CFLAGS"
		LIBS="$SAVED_LIBS"

		if test "x$dlb_incdir_found" != "xyes" -o "x$dlb_libdir_found" != "xyes"
		then
			enable_dlb=no
		fi
	fi

	AC_MSG_CHECKING(whether DLB support should be enabled)
	AC_MSG_RESULT($enable_dlb)
	if test "x$enable_dlb" != "xno"
	then
		AC_DEFINE([STARPURM_HAVE_DLB], [1], [Define to 1 if dlb support is enabled.])
		starpurm_dlb_support=yes

		AC_MSG_CHECKING(whether DLB resource management debug messages should be displayed)
		AC_ARG_ENABLE(starpurm-dlb-verbose, [AS_HELP_STRING([--enable-starpurm-dlb-verbose],
			      [display resource management verbose debug messages])],
			      enable_starpurm_dlb_verbose=$enableval, enable_starpurm_dlb_verbose=no)
		AC_MSG_RESULT($enable_starpurm_dlb_verbose)
		if test x$enable_starpurm_dlb_verbose = xyes; then
			AC_DEFINE(STARPURM_DLB_VERBOSE, [1], [display DLB resource management verbose debug messages])
		fi

		AX_DLB_CALLBACK_ARG()
	fi
	AC_SUBST(DLB_CFLAGS)
	AC_SUBST(DLB_LIBS)
fi
AM_CONDITIONAL(STARPURM_HAVE_DLB, test x$starpurm_dlb_support == "xyes")
AM_CONDITIONAL(STARPU_BUILD_STARPURM, [test x$starpurm_support = xyes])

AC_ARG_ENABLE(starpurm-examples, [AS_HELP_STRING([--enable-starpurm-examples],
			[enable build of StarPU Resource Manager examples])],
			enable_starpurm_examples=$enableval, enable_starpurm_examples=no)
AM_CONDITIONAL(STARPU_BUILD_STARPURM_EXAMPLES, [test x$enable_starpurm_examples = xyes])

##########################################
# Documentation                          #
##########################################

AC_ARG_ENABLE(build-doc, [AS_HELP_STRING([--disable-build-doc],
			[disable building of documentation])],
			enable_build_doc=$enableval, enable_build_doc=yes)

AC_ARG_ENABLE(build-doc-pdf, [AS_HELP_STRING([--enable-build-doc-pdf],
			[enable building of PDF documentation])],
			enable_build_doc_pdf=$enableval, enable_build_doc_pdf=no)

# Check whether doxygen needed tools are installed
AC_PATH_PROG(doxygencommand, doxygen)
if test "$doxygencommand" = "" ; then
   enable_build_doc="no"
   enable_build_doc_pdf="no"
fi
AC_PATH_PROG(pdflatexcommand, pdflatex)
if test "$pdflatexcommand" = "" ; then
   enable_build_doc_pdf="no"
fi
AC_PATH_PROG(epstopdfcommand, epstopdf)
if test "$epstopdfcommand" = "" ; then
   enable_build_doc_pdf="no"
fi

available_doc="no"
if test -d "$srcdir/doc/doxygen/html" ; then
   enable_build_doc="no"
   available_doc="yes"
fi
available_doc_pdf="no"
if test -f "$srcdir/doc/doxygen/starpu.pdf" ; then
   enable_build_doc="no"
   enable_build_doc_pdf="no"
   available_doc_pdf="yes"
fi
AC_MSG_CHECKING(whether HTML documentation should be compiled)
AC_MSG_RESULT($enable_build_doc)
AC_MSG_CHECKING(whether HTML documentation is available)
AC_MSG_RESULT($available_doc)
AC_MSG_CHECKING(whether PDF documentation should be compiled)
AC_MSG_RESULT($enable_build_doc_pdf)
AC_MSG_CHECKING(whether PDF documentation is available)
AC_MSG_RESULT($available_doc_pdf)

AM_CONDITIONAL(STARPU_BUILD_DOC, [test x$enable_build_doc != xno])
AM_CONDITIONAL(STARPU_AVAILABLE_DOC, [test x$available_doc != xno])

AM_CONDITIONAL(STARPU_BUILD_DOC_PDF, [test x$enable_build_doc_pdf != xno])
AM_CONDITIONAL(STARPU_AVAILABLE_DOC_PDF, [test x$available_doc_pdf != xno])

###############################################################################
#                                                                             #
#                                Final settings                               #
#                                                                             #
###############################################################################

# these are the flags needed for linking libstarpu (and thus also for static linking)
LIBSTARPU_LDFLAGS="$STARPU_OPENCL_LDFLAGS $STARPU_CUDA_LDFLAGS $HWLOC_LIBS $FXT_LIBS $STARPU_COI_LDFLAGS $STARPU_SCIF_LDFLAGS $STARPU_RCCE_LDFLAGS $STARPU_LEVELDB_LDFLAGS $STARPU_GLPK_LDFLAGS $STARPU_LEVELDB_LDFLAGS $SIMGRID_LIBS $STARPU_BLAS_LDFLAGS $STARPU_OMP_LDFLAGS $DGELS_LIBS"
AC_SUBST([LIBSTARPU_LDFLAGS])

LIBSTARPU_LINK=libstarpu-$STARPU_EFFECTIVE_VERSION.la
AC_SUBST([LIBSTARPU_LINK])

if test "x$enable_shared" = xno; then
        # No .so, so application will unexpected have to know which -l to
        # use. Give them in .pc file.
	AC_DEFINE(STARPU_STATIC_ONLY, [1], [Only static compilation was made])
	STARPU_EXPORTED_LIBS="$STARPU_EXPORTED_LIBS $LDFLAGS $LIBS $LIBSTARPU_LDFLAGS"
fi
AC_SUBST(STARPU_EXPORTED_LIBS)

# File configuration
AC_CONFIG_COMMANDS([executable-scripts], [
  chmod +x tests/regression/regression.sh
  chmod +x tests/loader-cross.sh
  chmod +x tests/model-checking/starpu-mc.sh
  chmod +x examples/loader-cross.sh
  chmod +x examples/stencil/loader-cross.sh
  chmod +x tools/starpu_env
  chmod +x tools/starpu_codelet_profile
  chmod +x tools/starpu_codelet_histo_profile
  chmod +x tools/starpu_mpi_comm_matrix.py
  chmod +x tools/starpu_workers_activity
  chmod +x tools/starpu_paje_draw_histogram
  chmod +x tools/starpu_paje_state_stats
  chmod +x tools/starpu_paje_summary
  chmod +x tools/starpu_mlr_analysis
  chmod +x tools/starpu_paje_sort
  chmod +x tools/starpu_smpirun
  chmod +x doc/doxygen/doxygen_filter.sh
  chmod +x doc/doxygen_dev/doxygen_filter.sh
  mkdir -p tests/microbenchs
  test -e tests/microbenchs/tasks_data_overhead.sh || ln -sf $ac_abs_top_srcdir/tests/microbenchs/tasks_data_overhead.sh tests/microbenchs/
  test -e tests/microbenchs/sync_tasks_data_overhead.sh || ln -sf $ac_abs_top_srcdir/tests/microbenchs/sync_tasks_data_overhead.sh tests/microbenchs/
  test -e tests/microbenchs/async_tasks_data_overhead.sh || ln -sf $ac_abs_top_srcdir/tests/microbenchs/async_tasks_data_overhead.sh tests/microbenchs/
  test -e tests/microbenchs/tasks_size_overhead.sh || ln -sf $ac_abs_top_srcdir/tests/microbenchs/tasks_size_overhead.sh tests/microbenchs/
  test -e tests/microbenchs/tasks_size_overhead_sched.sh || ln -sf $ac_abs_top_srcdir/tests/microbenchs/tasks_size_overhead_sched.sh tests/microbenchs/
  test -e tests/microbenchs/tasks_size_overhead_scheds.sh || ln -sf $ac_abs_top_srcdir/tests/microbenchs/tasks_size_overhead_scheds.sh tests/microbenchs/
  test -e tests/microbenchs/tasks_size_overhead.gp || ln -sf $ac_abs_top_srcdir/tests/microbenchs/tasks_size_overhead.gp tests/microbenchs/
  test -e tests/microbenchs/microbench.sh || ln -sf $ac_abs_top_srcdir/tests/microbenchs/microbench.sh tests/microbenchs/
  test -e tests/microbenchs/parallel_dependent_homogeneous_tasks_data.sh || ln -sf $ac_abs_top_srcdir/tests/microbenchs/parallel_dependent_homogeneous_tasks_data.sh tests/microbenchs/
  test -e tests/microbenchs/parallel_independent_heterogeneous_tasks_data.sh || ln -sf $ac_abs_top_srcdir/tests/microbenchs/parallel_independent_heterogeneous_tasks_data.sh tests/microbenchs/
  test -e tests/microbenchs/parallel_independent_heterogeneous_tasks.sh || ln -sf $ac_abs_top_srcdir/tests/microbenchs/parallel_independent_heterogeneous_tasks.sh tests/microbenchs/
  test -e tests/microbenchs/parallel_independent_homogeneous_tasks_data.sh || ln -sf $ac_abs_top_srcdir/tests/microbenchs/parallel_independent_homogeneous_tasks_data.sh tests/microbenchs/
  test -e tests/microbenchs/parallel_independent_homogeneous_tasks.sh || ln -sf $ac_abs_top_srcdir/tests/microbenchs/parallel_independent_homogeneous_tasks.sh tests/microbenchs/
  test -e tests/microbenchs/parallel_redux_homogeneous_tasks_data.sh || ln -sf $ac_abs_top_srcdir/tests/microbenchs/parallel_redux_homogeneous_tasks_data.sh tests/microbenchs/
  test -e tests/microbenchs/parallel_redux_heterogeneous_tasks_data.sh || ln -sf $ac_abs_top_srcdir/tests/microbenchs/parallel_redux_heterogeneous_tasks_data.sh tests/microbenchs/
  test -e tests/microbenchs/bandwidth_scheds.sh || ln -sf $ac_abs_top_srcdir/tests/microbenchs/bandwidth_scheds.sh tests/microbenchs/
  mkdir -p tests/datawizard
  test -e tests/datawizard/locality.sh || ln -sf $ac_abs_top_srcdir/tests/datawizard/locality.sh tests/datawizard/
  mkdir -p tests/overlap
  test -e tests/overlap/overlap.sh || ln -sf $ac_abs_top_srcdir/tests/overlap/overlap.sh tests/overlap/
  mkdir -p tests/model-checking
  test -e tests/model-checking/prio_list.sh || ln -sf $ac_abs_top_srcdir/tests/model-checking/prio_list.sh tests/model-checking/
  test -e tests/model-checking/barrier.sh || ln -sf $ac_abs_top_srcdir/tests/model-checking/barrier.sh tests/model-checking/
  mkdir -p examples/heat
  test -e examples/heat/heat.sh || ln -sf $ac_abs_top_srcdir/examples/heat/heat.sh examples/heat/
  mkdir -p examples/lu
  test -e examples/lu/lu.sh || ln -sf $ac_abs_top_srcdir/examples/lu/lu.sh examples/lu/
  mkdir -p examples/cholesky
  test -e examples/cholesky/cholesky.sh || ln -sf $ac_abs_top_srcdir/examples/cholesky/cholesky.sh examples/cholesky/
  mkdir -p examples/mult
  test -e examples/mult/sgemm.sh || ln -sf $ac_abs_top_srcdir/examples/mult/sgemm.sh examples/mult/
  mkdir -p examples/scheduler
  test -e examples/scheduler/schedulers.sh || ln -sf $ac_abs_top_srcdir/examples/scheduler/schedulers.sh examples/scheduler
  test -e examples/scheduler/schedulers_context.sh || ln -sf $ac_abs_top_srcdir/examples/scheduler/schedulers_context.sh examples/scheduler
  test -e tools/starpu_paje_draw_histogram.R || ln -sf $ac_abs_top_srcdir/tools/starpu_paje_draw_histogram.R tools/starpu_paje_draw_histogram.R
  test -e tools/starpu_paje_state_stats.R || ln -sf $ac_abs_top_srcdir/tools/starpu_paje_state_stats.R tools/starpu_paje_state_stats.R
  test -e tools/starpu_trace_state_stats.py || ln -sf $ac_abs_top_srcdir/tools/starpu_trace_state_stats.py tools/starpu_trace_state_stats.py
  chmod +x tools/starpu_trace_state_stats.py
])

# Create links to ICD files in build/socl/vendors directory. SOCL will use this
# directory as the OCL_ICD_VENDORS directory
SOCL_VENDORS="vendors/install/socl.icd"
for icd in /etc/OpenCL/vendors/*.icd ; do
	if test -f $icd ; then
	        if test "$(basename $icd)" != "socl.icd" ; then
        		new_icd=$(basename $icd)
			AC_CONFIG_LINKS([socl/vendors/$new_icd:$icd])
			SOCL_VENDORS="$SOCL_VENDORS vendors/$new_icd"
		fi
        fi
done
AC_SUBST(SOCL_VENDORS)

AC_CONFIG_FILES(tests/regression/regression.sh tests/regression/profiles tests/regression/profiles.build.only)
AC_CONFIG_HEADER(src/common/config.h include/starpu_config.h starpurm/include/starpurm_config.h)

SANITIZE=$(echo $CFLAGS | grep sanitize)
AM_CONDITIONAL(STARPU_SANITIZE, test -n "$SANITIZE")

AC_OUTPUT([
	Makefile
	src/Makefile
	tools/Makefile
	tools/starpu_env
	tools/starpu_codelet_profile
	tools/starpu_codelet_histo_profile
	tools/starpu_mpi_comm_matrix.py
	tools/starpu_workers_activity
	tools/starpu_paje_draw_histogram
	tools/starpu_paje_state_stats
	tools/starpu_paje_summary
	tools/starpu_mlr_analysis
	tools/starpu_paje_sort
	tools/starpu_smpirun
	socl/Makefile
	socl/src/Makefile
	socl/examples/Makefile
	socl/vendors/socl.icd
	socl/vendors/install/socl.icd
	libstarpu.pc
	starpu-1.0.pc
	starpu-1.1.pc
	starpu-1.2.pc
	starpu-1.3.pc
	mpi/libstarpumpi.pc
	mpi/starpumpi-1.0.pc
	mpi/starpumpi-1.1.pc
	mpi/starpumpi-1.2.pc
	mpi/starpumpi-1.3.pc
	starpufft/Makefile
	starpufft/src/Makefile
	starpufft/tests/Makefile
	starpufft/libstarpufft.pc
	starpufft/starpufft-1.0.pc
	starpufft/starpufft-1.1.pc
	starpufft/starpufft-1.2.pc
	starpufft/starpufft-1.3.pc
	starpurm/Makefile
	starpurm/src/Makefile
	starpurm/examples/Makefile
	starpurm/starpurm-1.3.pc
	examples/Makefile
	examples/stencil/Makefile
	tests/Makefile
	tests/loader-cross.sh
	tests/model-checking/starpu-mc.sh
	examples/loader-cross.sh
	examples/stencil/loader-cross.sh
	mpi/Makefile
	mpi/src/Makefile
	mpi/tests/Makefile
	mpi/examples/Makefile
	sc_hypervisor/Makefile
	sc_hypervisor/src/Makefile
	sc_hypervisor/examples/Makefile
	doc/Makefile
	doc/doxygen/Makefile
	doc/doxygen/doxygen-config.cfg
	doc/doxygen/doxygen_filter.sh
	doc/doxygen_dev/Makefile
	doc/doxygen_dev/doxygen-config.cfg
	doc/doxygen_dev/doxygen_filter.sh
	tools/msvc/starpu_var.bat
	min-dgels/Makefile
])

AC_MSG_NOTICE([

	CPUs   enabled: $enable_cpu
	CUDA   enabled: $enable_cuda
	OpenCL enabled: $enable_opencl
	MIC    enabled: $enable_mic

	Compile-time limits
	(change these with --enable-maxcpus, --enable-maxcudadev,
	--enable-maxopencldev, --enable-maxmicdev, --enable-maxnodes,
        --enable-maxbuffers)
        (Note these numbers do not represent the number of detected
	devices, but the maximum number of devices StarPU can manage)

	Maximum number of CPUs:                     $maxcpus
	Maximum number of CUDA devices:             $nmaxcudadev
	Maximum number of OpenCL devices:           $nmaxopencldev
	Maximum number of MIC threads:              $nmaxmicthreads
	Maximum number of MPI master-slave devices: $nmaxmpidev
	Maximum number of memory nodes:             $maxnodes
	Maximum number of task buffers:             $nmaxbuffers

	GPU-GPU transfers: $enable_cuda_memcpy_peer
	Allocation cache:  $enable_allocation_cache

	Magma enabled:     $have_magma
	BLAS library:      $blas_lib
	hwloc:             $have_valid_hwloc
	FxT trace enabled: $use_fxt

        Documentation HTML:  $enable_build_doc
        Documentation PDF:   $enable_build_doc_pdf
        Examples:            $enable_build_examples

	StarPU Extensions:
	       StarPU MPI enabled:                            $build_mpi_lib
	       StarPU MPI(nmad) enabled:                      $build_nmad_lib
	       MPI test suite:                                $running_mpi_check
	       Master-Slave MPI enabled:                      $use_mpi_master_slave
	       FFT Support:                                   $fft_support
	       Resource Management enabled:                   $starpurm_support
	       OpenMP runtime support enabled:                $enable_openmp
	       Cluster support enabled:                       $enable_cluster
	       SOCL enabled:                                  $build_socl
               SOCL test suite:                               $run_socl_check
               Scheduler Hypervisor:                          $build_sc_hypervisor
               simgrid enabled:                               $enable_simgrid
               ayudame enabled:                               $ayu_msg
               HDF5 enabled:                                  $enable_hdf5
	       Native fortran support:                        $enable_build_fortran
	       Native MPI fortran support:                    $use_mpi_fort
	       Support for multiple linear regression models: $support_mlr
])

if test "$build_socl" = "yes" -a "$run_socl_check" = "no" ; then
	AC_MSG_NOTICE([
WARNING: SOCL test suite will not be run as the environment variable SOCL_OCL_LIB_OPENCL is not defined.
To run the tests, you need to install the OCL implementation of ICD
(https://forge.imag.fr/projects/ocl-icd/ or Debian package ocl-icd-libopencl1)
and set the variable SOCL_OCL_LIB_OPENCL to the location of the libOpenCL.so.])
fi

if test x"$have_valid_hwloc" = xno -a "$enable_simgrid" = "no" ; then
  AC_MSG_NOTICE([
WARNING: hwloc was not enabled.  If the target machine is hyperthreaded the
performance may be impacted a lot.  It is strongly recommended to install
hwloc])
fi

if test x"$starpu_windows" = xyes -a "x$STARPU_MS_LIB" = "x" ; then
  AC_MSG_NOTICE([
WARNING: lib was not found, you will not be able to build StarPU applications
with Microsoft Visual Studio. Add to your PATH the directories for MSVC, e.g
   c:\Program Files (x86)\Microsoft Visual Studio 11.0\Common7\IDE;
   c:\Program Files (x86)\Microsoft Visual Studio 11.0\VC\bin])
fi
