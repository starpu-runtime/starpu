! StarPU --- Runtime system for heterogeneous multicore architectures.
!
! Copyright (C) 2015-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
! Copyright (C) 2015       ONERA
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
MODULE nf_types

  USE iso_c_binding

  TYPE type_numpar
     REAL(KIND=C_DOUBLE)                        :: coeff
     INTEGER(KIND=C_INT)                        :: Neq_max
  END TYPE type_numpar

  TYPE type_mesh_elt
     INTEGER(KIND=C_INT)                        :: Ng, Np
     REAL(KIND=C_DOUBLE),POINTER,DIMENSION(:,:) :: ro, dro
     REAL(KIND=C_DOUBLE),POINTER,DIMENSION(:,:) :: basis
     TYPE(C_PTR)                                :: ro_h, dro_h, basis_h
  END TYPE type_mesh_elt

  TYPE type_mesh
     TYPE(type_mesh_elt), POINTER, DIMENSION(:) :: elt
  END TYPE type_mesh

END MODULE nf_types
