dnl This test is taken from libgfortran

dnl Check whether the target supports __sync_*_compare_and_swap.
AC_DEFUN([STARPU_CHECK_SYNC_BUILTINS], [
  AC_CACHE_CHECK([whether the target supports __sync_*_compare_and_swap],
		 ac_cv_have_sync_builtins, [
  AC_LINK_IFELSE([AC_LANG_PROGRAM([int foo, bar;],
			[bar = __sync_val_compare_and_swap(&foo, 0, 1);])],
			[ac_cv_have_sync_builtins=yes],
			[ac_cv_have_sync_builtins=no])])
  if test $ac_cv_have_sync_builtins = yes; then
    AC_DEFINE(HAVE_SYNC_BUILTINS, 1,
	      [Define to 1 if the target supports __sync_*_compare_and_swap])
  fi])
