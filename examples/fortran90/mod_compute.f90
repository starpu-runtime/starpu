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

MODULE mod_compute

  USE mod_types
  USE starpu_mod
  USE mod_interface
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
  RECURSIVE SUBROUTINE loop_element_cpu_fortran(coeff,Neq_max,Np,Ng, &
       &   ro_ptr,dro_ptr,basis_ptr) BIND(C)
    INTEGER(KIND=C_INT),VALUE                  :: Neq_max,Np,Ng
    REAL(KIND=C_DOUBLE),VALUE                  :: coeff
    TYPE(C_PTR)                                :: ro_ptr,dro_ptr,basis_ptr
    !Local variables
    REAL(KIND=C_DOUBLE),DIMENSION(:,:),POINTER :: ro,dro,basis

    CALL C_F_POINTER(ro_ptr,ro,[Neq_max,Np])
    CALL C_F_POINTER(dro_ptr,dro,[Neq_max,Np])
    CALL C_F_POINTER(basis_ptr,basis,[Np,Ng])

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
  RECURSIVE SUBROUTINE copy_element_cpu_fortran(Neq_max,Np, &
       &   ro_ptr,dro_ptr) BIND(C)
    INTEGER(KIND=C_INT),VALUE                  :: Neq_max,Np
    TYPE(C_PTR)                                :: ro_ptr,dro_ptr
    !Local variables
    REAL(KIND=C_DOUBLE),DIMENSION(:,:),POINTER :: ro,dro

    CALL C_F_POINTER(ro_ptr,ro,[Neq_max,Np])
    CALL C_F_POINTER(dro_ptr,dro,[Neq_max,Np])

    CALL copy_element_cpu(ro,dro)

  END SUBROUTINE copy_element_cpu_fortran

  !--------------------------------------------------------------!
  RECURSIVE SUBROUTINE copy_element_cpu(ro,dro)
    REAL(KIND=C_DOUBLE),DIMENSION(:,:),POINTER,INTENT(INOUT) :: ro
    REAL(KIND=C_DOUBLE),DIMENSION(:,:),POINTER,INTENT(IN)    :: dro

    ro = ro + dro

  END SUBROUTINE copy_element_cpu

END MODULE mod_compute
