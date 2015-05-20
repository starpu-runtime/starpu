! StarPU --- Runtime system for heterogeneous multicore architectures.
!
! Copyright (C) 2015  ONERA
! Copyright (C) 2015  Inria
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

MODULE mod_compute

  USE mod_types
  USE mod_starpu
  USE mod_interface
  USE iso_c_binding

  IMPLICIT NONE

CONTAINS

  !--------------------------------------------------------------!
  SUBROUTINE init_element(ro,dro,basis,Neq_max,Np,Ng,i)
    INTEGER,INTENT(IN)                        :: Neq_max,Np,Ng,i
    REAL,DIMENSION(:,:),POINTER,INTENT(INOUT) :: ro,basis,dro
    !Local variables
    INTEGER                                   :: n,nb,neq

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
  SUBROUTINE loop_element_cpu_fortran(coeff,Neq_max,Np,Ng, &
       &   ro_ptr,dro_ptr,basis_ptr) BIND(C) 
    INTEGER(KIND=C_INT),VALUE                 :: Neq_max,Np,Ng
    REAL(KIND=C_DOUBLE),VALUE                 :: coeff
    TYPE(C_PTR)                               :: ro_ptr,dro_ptr,basis_ptr
    !Local variables 
    REAL,DIMENSION(:,:),POINTER               :: ro,dro,basis

    CALL C_F_POINTER(ro_ptr,ro,[Neq_max,Np])
    CALL C_F_POINTER(dro_ptr,dro,[Neq_max,Np])
    CALL C_F_POINTER(basis_ptr,basis,[Np,Ng])
  
    CALL loop_element_cpu(ro,dro,basis,coeff,Neq_max,Ng,Np)

  END SUBROUTINE loop_element_cpu_fortran

  !--------------------------------------------------------------!
  SUBROUTINE loop_element_cpu(ro,dro,basis,coeff,Neq_max,Ng,Np)
    REAL,INTENT(IN)                           :: coeff
    INTEGER,INTENT(IN)                        :: Neq_max,Ng,Np
    REAL,DIMENSION(:,:),POINTER,INTENT(IN)    :: ro,basis
    REAL,DIMENSION(:,:),POINTER,INTENT(INOUT) :: dro
    !Local variables
    REAL                                      :: coeff2,r
    INTEGER                                   :: n,nb,neq

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
  SUBROUTINE copy_element_cpu_fortran(Neq_max,Np, &
       &   ro_ptr,dro_ptr) BIND(C) 
    INTEGER(KIND=C_INT),VALUE                 :: Neq_max,Np
    TYPE(C_PTR)                               :: ro_ptr,dro_ptr
    !Local variables 
    REAL,DIMENSION(:,:),POINTER               :: ro,dro

    CALL C_F_POINTER(ro_ptr,ro,[Neq_max,Np])
    CALL C_F_POINTER(dro_ptr,dro,[Neq_max,Np])
  
    CALL copy_element_cpu(ro,dro)

  END SUBROUTINE copy_element_cpu_fortran

  !--------------------------------------------------------------!
  SUBROUTINE copy_element_cpu(ro,dro)
    REAL,DIMENSION(:,:),POINTER,INTENT(INOUT) :: ro
    REAL,DIMENSION(:,:),POINTER,INTENT(IN)    :: dro

    ro = ro + dro

  END SUBROUTINE copy_element_cpu

END MODULE mod_compute
