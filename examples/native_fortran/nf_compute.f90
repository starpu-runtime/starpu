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
! Computation kernels for the simulation

MODULE nf_compute

  USE nf_types
  USE fstarpu_mod
  USE iso_c_binding

  IMPLICIT NONE

CONTAINS

  !--------------------------------------------------------------!
  SUBROUTINE init_element(ro,dro,basis,Neq_max,Np,Ng,i)
    INTEGER(KIND=C_INT),INTENT(IN)                           :: Neq_max,Np,Ng,i
    REAL(KIND=C_DOUBLE),DIMENSION(:,:),POINTER,INTENT(INOUT) :: ro,basis,dro
    !Local variables
    INTEGER(KIND=C_INT)                                      :: n,nb,neq

    DO nb=1,Np
       DO neq= 1,Neq_max
          ro(neq,nb)  = 0.01*(nb+neq)*i
       END DO
    END DO

    DO nb=1,Np
       DO neq= 1,Neq_max
          dro(neq,nb) = 0.05*(nb-neq)*i
       END DO
    END DO

    DO n=1,Ng
       DO nb=1,Np
          basis(nb,n) = 0.05*(n+nb)*i
       END DO
    END DO

  END SUBROUTINE init_element

  !--------------------------------------------------------------!
  RECURSIVE SUBROUTINE loop_element_cpu_fortran(buffers, cl_args) BIND(C)
    TYPE(C_PTR), VALUE, INTENT(IN)              :: buffers, cl_args

    INTEGER(KIND=C_INT)                         :: Neq_max,Np,Ng
    REAL(KIND=C_DOUBLE),DIMENSION(:,:),POINTER  :: ro,dro,basis
    REAL(KIND=C_DOUBLE),TARGET                  :: coeff

    Neq_max = fstarpu_matrix_get_nx(buffers, 0)
    Np = fstarpu_matrix_get_nx(buffers, 2)
    Ng = fstarpu_matrix_get_ny(buffers, 2)

    CALL fstarpu_unpack_arg(cl_args,(/ c_loc(coeff) /))

    CALL c_f_pointer(fstarpu_matrix_get_ptr(buffers, 0), ro, shape=[Neq_max,Np])
    CALL c_f_pointer(fstarpu_matrix_get_ptr(buffers, 1), dro, shape=[Neq_max,Np])
    CALL c_f_pointer(fstarpu_matrix_get_ptr(buffers, 2), basis, shape=[Np,Ng])

    CALL loop_element_cpu(ro,dro,basis,coeff,Neq_max,Ng,Np)
  END SUBROUTINE loop_element_cpu_fortran

  !--------------------------------------------------------------!
  RECURSIVE SUBROUTINE loop_element_cpu(ro,dro,basis,coeff,Neq_max,Ng,Np)
    REAL(KIND=C_DOUBLE),INTENT(IN)                           :: coeff
    INTEGER(KIND=C_INT),INTENT(IN)                           :: Neq_max,Ng,Np
    REAL(KIND=C_DOUBLE),DIMENSION(:,:),POINTER,INTENT(IN)    :: ro,basis
    REAL(KIND=C_DOUBLE),DIMENSION(:,:),POINTER,INTENT(INOUT) :: dro
    !Local variables
    REAL(KIND=C_DOUBLE)                                      :: coeff2,r
    INTEGER(KIND=C_INT)                                      :: n,nb,neq

    DO n=1,Ng
       r = 0.
       DO nb=1,Np
          DO neq= 1,Neq_max
             r = r + basis(nb,n) * ro(neq,nb)
          ENDDO
       ENDDO

       coeff2 = r + coeff

       DO nb=1,Np
          DO neq = 1,Neq_max
             dro(neq,nb) = coeff2 + dro(neq,nb)
          ENDDO
       ENDDO
    ENDDO

  END SUBROUTINE loop_element_cpu

  !--------------------------------------------------------------!
  RECURSIVE SUBROUTINE copy_element_cpu_fortran(buffers, cl_args) BIND(C)
    TYPE(C_PTR), VALUE, INTENT(IN)              :: buffers, cl_args

    INTEGER(KIND=C_INT)                         :: Neq_max,Np
    REAL(KIND=C_DOUBLE),DIMENSION(:,:),POINTER  :: ro,dro

    Neq_max = fstarpu_matrix_get_nx(buffers, 0)
    Np = fstarpu_matrix_get_ny(buffers, 0)

    CALL c_f_pointer(fstarpu_matrix_get_ptr(buffers, 0), ro, shape=[Neq_max,Np])
    CALL c_f_pointer(fstarpu_matrix_get_ptr(buffers, 1), dro, shape=[Neq_max,Np])

    CALL copy_element_cpu(ro,dro)

  END SUBROUTINE copy_element_cpu_fortran

  !--------------------------------------------------------------!
  RECURSIVE SUBROUTINE copy_element_cpu(ro,dro)
    REAL(KIND=C_DOUBLE),DIMENSION(:,:),POINTER,INTENT(INOUT) :: ro
    REAL(KIND=C_DOUBLE),DIMENSION(:,:),POINTER,INTENT(IN)    :: dro

    ro = ro + dro

  END SUBROUTINE copy_element_cpu

END MODULE nf_compute
