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
    AC_DEFINE(STARPU_HAVE_SYNC_BUILTINS, 1,
	      [Define to 1 if the target supports __sync_*_compare_and_swap])
  fi])

dnl Check whether the target supports __sync_fetch_and_add.
AC_DEFUN([STARPU_CHECK_SYNC_FETCH_AND_ADD], [
  AC_CACHE_CHECK([whether the target supports __sync_fetch_and_add],
		 ac_cv_have_sync_fetch_and_add, [
  AC_LINK_IFELSE([AC_LANG_PROGRAM([int foo, bar;],
			[bar = __sync_fetch_and_add(&foo, 1);])],
			[ac_cv_have_sync_fetch_and_add=yes],
			[ac_cv_have_sync_fetch_and_add=no])])
  if test $ac_cv_have_sync_fetch_and_add = yes; then
    AC_DEFINE(STARPU_HAVE_SYNC_FETCH_AND_ADD, 1,
	      [Define to 1 if the target supports __sync_fetch_and_add])
  fi])

dnl Check whether the target supports __sync_fetch_and_or.
AC_DEFUN([STARPU_CHECK_SYNC_FETCH_AND_OR], [
  AC_CACHE_CHECK([whether the target supports __sync_fetch_and_or],
		 ac_cv_have_sync_fetch_and_or, [
  AC_LINK_IFELSE([AC_LANG_PROGRAM([int foo, bar;],
			[bar = __sync_fetch_and_or(&foo, 1);])],
			[ac_cv_have_sync_fetch_and_or=yes],
			[ac_cv_have_sync_fetch_and_or=no])])
  if test $ac_cv_have_sync_fetch_and_or = yes; then
    AC_DEFINE(STARPU_HAVE_SYNC_FETCH_AND_OR, 1,
	      [Define to 1 if the target supports __sync_fetch_and_or])
  fi])

dnl Check whether the target supports __sync_lock_test_and_set.
AC_DEFUN([STARPU_CHECK_SYNC_LOCK_TEST_AND_SET], [
  AC_CACHE_CHECK([whether the target supports __sync_lock_test_and_set],
		 ac_cv_have_sync_lock_test_and_set, [
  AC_LINK_IFELSE([AC_LANG_PROGRAM([int foo, bar;],
			[bar = __sync_lock_test_and_set(&foo, 1);])],
			[ac_cv_have_sync_lock_test_and_set=yes],
			[ac_cv_have_sync_lock_test_and_set=no])])
  if test $ac_cv_have_sync_lock_test_and_set = yes; then
    AC_DEFINE(STARPU_HAVE_SYNC_LOCK_TEST_AND_SET, 1,
	      [Define to 1 if the target supports __sync_lock_test_and_set])
  fi])
