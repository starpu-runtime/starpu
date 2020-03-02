# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2012-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

# Check whether the target supports __sync_val_compare_and_swap.
AC_DEFUN([STARPU_CHECK_SYNC_VAL_COMPARE_AND_SWAP], [
  AC_CACHE_CHECK([whether the target supports __sync_val_compare_and_swap],
		 ac_cv_have_sync_val_compare_and_swap, [
  AC_LINK_IFELSE([AC_LANG_PROGRAM([int foo, bar;],
			[bar = __sync_val_compare_and_swap(&foo, 0, 1);])],
			[ac_cv_have_sync_val_compare_and_swap=yes],
			[ac_cv_have_sync_val_compare_and_swap=no])])
  if test $ac_cv_have_sync_val_compare_and_swap = yes; then
    AC_DEFINE(STARPU_HAVE_SYNC_VAL_COMPARE_AND_SWAP, 1,
	      [Define to 1 if the target supports __sync_val_compare_and_swap])
  fi])

# Check whether the target supports __sync_bool_compare_and_swap.
AC_DEFUN([STARPU_CHECK_SYNC_BOOL_COMPARE_AND_SWAP], [
  AC_CACHE_CHECK([whether the target supports __sync_bool_compare_and_swap],
		 ac_cv_have_sync_bool_compare_and_swap, [
  AC_LINK_IFELSE([AC_LANG_PROGRAM([int foo, bar;],
			[bar = __sync_bool_compare_and_swap(&foo, 0, 1);])],
			[ac_cv_have_sync_bool_compare_and_swap=yes],
			[ac_cv_have_sync_bool_compare_and_swap=no])])
  if test $ac_cv_have_sync_bool_compare_and_swap = yes; then
    AC_DEFINE(STARPU_HAVE_SYNC_BOOL_COMPARE_AND_SWAP, 1,
	      [Define to 1 if the target supports __sync_bool_compare_and_swap])
  fi])

# Check whether the target supports __sync_fetch_and_add.
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

# Check whether the target supports __sync_fetch_and_or.
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

# Check whether the target supports __sync_lock_test_and_set.
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

# Check whether the target supports __atomic_compare_exchange_n.
AC_DEFUN([STARPU_CHECK_ATOMIC_COMPARE_EXCHANGE_N], [
  AC_CACHE_CHECK([whether the target supports __atomic_compare_exchange_n],
		 ac_cv_have_atomic_compare_exchange_n, [
  AC_LINK_IFELSE([AC_LANG_PROGRAM([int foo, bar, baz;],
			[baz = __atomic_compare_exchange_n(&foo, &bar, 1, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);])],
			[ac_cv_have_atomic_compare_exchange_n=yes],
			[ac_cv_have_atomic_compare_exchange_n=no])])
  if test $ac_cv_have_atomic_compare_exchange_n = yes; then
    AC_DEFINE(STARPU_HAVE_ATOMIC_COMPARE_EXCHANGE_N, 1,
	      [Define to 1 if the target supports __atomic_compare_exchange_n])
  fi])

# Check whether the target supports __atomic_exchange_n.
AC_DEFUN([STARPU_CHECK_ATOMIC_EXCHANGE_N], [
  AC_CACHE_CHECK([whether the target supports __atomic_exchange_n],
		 ac_cv_have_atomic_exchange_n, [
  AC_LINK_IFELSE([AC_LANG_PROGRAM([int foo, bar;],
			[bar = __atomic_exchange_n(&foo, 1, __ATOMIC_SEQ_CST);])],
			[ac_cv_have_atomic_exchange_n=yes],
			[ac_cv_have_atomic_exchange_n=no])])
  if test $ac_cv_have_atomic_exchange_n = yes; then
    AC_DEFINE(STARPU_HAVE_ATOMIC_EXCHANGE_N, 1,
	      [Define to 1 if the target supports __atomic_exchange_n])
  fi])

# Check whether the target supports __atomic_fetch_add.
AC_DEFUN([STARPU_CHECK_ATOMIC_FETCH_ADD], [
  AC_CACHE_CHECK([whether the target supports __atomic_fetch_add],
		 ac_cv_have_atomic_fetch_add, [
  AC_LINK_IFELSE([AC_LANG_PROGRAM([int foo, bar;],
			[bar = __atomic_fetch_add(&foo, 1, __ATOMIC_SEQ_CST);])],
			[ac_cv_have_atomic_fetch_add=yes],
			[ac_cv_have_atomic_fetch_add=no])])
  if test $ac_cv_have_atomic_fetch_add = yes; then
    AC_DEFINE(STARPU_HAVE_ATOMIC_FETCH_ADD, 1,
	      [Define to 1 if the target supports __atomic_fetch_add])
  fi])

# Check whether the target supports __atomic_fetch_or.
AC_DEFUN([STARPU_CHECK_ATOMIC_FETCH_OR], [
  AC_CACHE_CHECK([whether the target supports __atomic_fetch_or],
		 ac_cv_have_atomic_fetch_or, [
  AC_LINK_IFELSE([AC_LANG_PROGRAM([int foo, bar;],
			[bar = __atomic_fetch_or(&foo, 1, __ATOMIC_SEQ_CST);])],
			[ac_cv_have_atomic_fetch_or=yes],
			[ac_cv_have_atomic_fetch_or=no])])
  if test $ac_cv_have_atomic_fetch_or = yes; then
    AC_DEFINE(STARPU_HAVE_ATOMIC_FETCH_OR, 1,
	      [Define to 1 if the target supports __atomic_fetch_or])
  fi])

# Check whether the target supports __atomic_test_and_set.
AC_DEFUN([STARPU_CHECK_ATOMIC_TEST_AND_SET], [
  AC_CACHE_CHECK([whether the target supports __atomic_test_and_set],
		 ac_cv_have_atomic_test_and_set, [
  AC_LINK_IFELSE([AC_LANG_PROGRAM([int foo, bar;],
			[bar = __atomic_test_and_set(&foo, __ATOMIC_SEQ_CST);])],
			[ac_cv_have_atomic_test_and_set=yes],
			[ac_cv_have_atomic_test_and_set=no])])
  if test $ac_cv_have_atomic_test_and_set = yes; then
    AC_DEFINE(STARPU_HAVE_ATOMIC_TEST_AND_SET, 1,
	      [Define to 1 if the target supports __atomic_test_and_set])
  fi])

# Check whether the target supports __sync_synchronize.
AC_DEFUN([STARPU_CHECK_SYNC_SYNCHRONIZE], [
  AC_CACHE_CHECK([whether the target supports __sync_synchronize],
		 ac_cv_have_sync_synchronize, [
  AC_LINK_IFELSE([AC_LANG_PROGRAM(,
			[__sync_synchronize();])],
			[ac_cv_have_sync_synchronize=yes],
			[ac_cv_have_sync_synchronize=no])])
  if test $ac_cv_have_sync_synchronize = yes; then
    AC_DEFINE(STARPU_HAVE_SYNC_SYNCHRONIZE, 1,
	      [Define to 1 if the target supports __sync_synchronize])
  fi])
