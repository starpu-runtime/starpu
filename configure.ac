AC_INIT([StarPU],0.1)

AC_MSG_NOTICE([Configuring StarPU])

AC_PROG_CC
AC_PROG_F77

AC_MSG_CHECKING(whether CPUs should be used)
AC_ARG_ENABLE(cpu, [AS_HELP_STRING([--disable-cpu],
			[do not use the CPU(s)])],
			enable_cpu=$enableval, enable_cpu=yes)
AC_MSG_RESULT($enable_cpu)
AC_SUBST(USE_CPU, $enable_cpu)

if test x$enable_cpu = xyes; then
	# This value is set quite randomly, but StarPU should not take more
	# core than there are in the system
	AC_SUBST(NMAXCORES, 16)
fi


AC_MSG_CHECKING(whether CUDA devices should be used)
AC_ARG_ENABLE(cuda, [AS_HELP_STRING([--enable-cuda],
			[use CUDA device(s)])],
			enable_cuda=$enableval, enable_cuda=no)
AC_MSG_RESULT($enable_cuda)
AC_SUBST(USE_CUDA, $enable_cuda)

AC_MSG_CHECKING(whether a dynamic library should be generated)
AC_ARG_ENABLE(dynamic, [AS_HELP_STRING([--enable-dynamic],
			[generate a dynamic library])],
			enable_dynamic=$enableval, enable_dynamic=no)
AC_MSG_RESULT($enable_dynamic)
AC_SUBST(DYNAMIC, $enable_dynamic)

AC_MSG_CHECKING(whether coverage testing should be enabled)
AC_ARG_ENABLE(coverage, [AS_HELP_STRING([--enable-coverage],
			[enable coverage checking])],
			enable_coverage=$enableval, enable_coverage=no)
AC_MSG_RESULT($enable_coverage)
AC_SUBST(COVERAGE, $enable_coverage)

AC_MSG_CHECKING(whether OpenGL rendering is enabled)
AC_ARG_ENABLE(opengl-render, [AS_HELP_STRING([--enable-opengl-render],
			[enable OpenGL rendering of some examples])],
			enable_opengl_render=$enableval, enable_opengl_render=no)
AC_MSG_RESULT($enable_opengl_render)
AC_SUBST(OPENGL_RENDER, $enable_opengl_render)


AC_MSG_CHECKING(whether priorities should be enabled)
AC_ARG_ENABLE(priority, [AS_HELP_STRING([--disable-priority],
			[do not use priorities])],
			enable_priority=$enableval, enable_priority=yes)
AC_MSG_RESULT($enable_priority)
AC_SUBST(USE_PRIO, $enable_priority)

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

if test x$use_fxt = xyes; then

AC_SUBST(USE_FXTDIR_FROM_USER,$use_fxt_from_system)
if test x$use_fxt_from_system = xno; then
	CPPFLAGS+=" -I$fxtdir/include/"
	LDFLAGS+=" -L$fxtdir/lib/"
fi
	AC_CHECK_LIB(fxt, fut_setup,,AC_MSG_ERROR([cannot find fxt lib]))
	AC_CHECK_HEADER([fxt/fxt.h],,AC_MSG_ERROR([cannot find headers for fxt]))
	AC_CHECK_HEADER([fxt/fut.h],,AC_MSG_ERROR([cannot find headers for fxt]))
fi

AC_MSG_CHECKING(whether performance debugging should be enabled)
AC_ARG_ENABLE(perf-debug, [AS_HELP_STRING([--enable-perf-debug],
			[enable performance debugging])],
			enable_perf_debug=$enableval, enable_perf_debug=no)
AC_MSG_RESULT($enable_perf_debug)
AC_SUBST(PERF_DEBUG, $enable_perf_debug)

# TODO : make it case insensitive ...
AC_MSG_CHECKING(which BLAS lib should be used)
blas_lib=none
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
AC_MSG_RESULT($blas_lib)
AC_SUBST(BLAS_LIB,$blas_lib)


AC_OUTPUT(examples/Makefile)
AC_OUTPUT(Makefile)
AC_OUTPUT(src/common/Makefile)
AC_OUTPUT(src/core/Makefile)
AC_OUTPUT(src/core/mechanisms/Makefile)
AC_OUTPUT(src/core/dependencies/Makefile)
AC_OUTPUT(src/drivers/Makefile)
AC_OUTPUT(examples/heat/Makefile)
AC_OUTPUT(examples/incrementer/Makefile)
