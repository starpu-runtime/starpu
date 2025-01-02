! StarPU --- Runtime system for heterogeneous multicore architectures.
!
! Copyright (C) 2015-2025   University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
!
! StarPU is free software; you can redistribute it and/or modify
! it under the terms of the GNU Lesser General Public License as published by
! the Free Software Foundation; either version 2.1 of the License, or (at
! your option) any later version.
!
! StarPU is distributed in the hope that it will be useful, but
! WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
!
! See the GNU Lesser General Public License in COPYING.LGPL for more details.
!
PROGRAM init_01

  USE starpu_mod
  USE iso_c_binding

  IMPLICIT NONE

  INTEGER(KIND=C_INT) :: res

  res = starpu_init(C_NULL_PTR)
  IF (res /= 0) THEN
          STOP 77
  END IF
  CALL starpu_shutdown()
END PROGRAM init_01
