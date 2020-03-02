# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2018-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

# Check whether DLB callbacks expect an user argument
AC_DEFUN([AX_DLB_CALLBACK_ARG],
[AC_MSG_CHECKING([whether DLB callbacks expect an user argument])
  AC_CACHE_VAL(ac_cv_dlb_callback_arg,dnl
  [AC_TRY_COMPILE(dnl
[#include <dlb_sp.h>
dlb_handler_t dlb_handle;
void _dlb_callback_disable_cpu(int cpuid, void *arg) {
  (void)cpuid;
  (void)arg;
}
void f(void) {
(void)DLB_CallbackSet_sp(dlb_handle, dlb_callback_disable_cpu, (dlb_callback_t)_dlb_callback_disable_cpu, 0);
}
],, ac_cv_dlb_callback_arg=yes, ac_cv_dlb_callback_arg=no)
  ])dnl AC_CACHE_VAL
  AC_MSG_RESULT([$ac_cv_dlb_callback_arg])
  if test $ac_cv_dlb_callback_arg = yes; then
    AC_DEFINE(STARPURM_HAVE_DLB_CALLBACK_ARG,1,[Define to 1 if DLB callbacks expect an user argument])
  fi
])
