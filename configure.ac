AC_INIT([StarPU],0.1)

AC_MSG_NOTICE([Configuring StarPU])

AC_PROG_CC
AC_PROG_F77

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
			[enable OpenGL rending of examples' results])],
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
AC_ARG_ENABLE(fxt, [AS_HELP_STRING([--enable-fxt],
			[generate fxt traces])],
			use_fxt=$enableval, use_fxt=no)
AC_MSG_RESULT($use_fxt)
AC_SUBST(USE_FXT, $use_fxt)

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
