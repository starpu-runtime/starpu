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
! This is an example of Fortran90 program making use of StarPU.
! It registers a few matrices for each element of a domain, performs
! update computations on them, and checks the result.

PROGRAM f90_example

  USE mod_types
  USE starpu_mod
  USE mod_interface
  USE mod_compute
  USE iso_c_binding

  IMPLICIT NONE

  TYPE(type_mesh)                :: mesh
  TYPE(type_numpar)              :: numpar
  TYPE(type_mesh_elt),POINTER    :: elt   => NULL()
  INTEGER(KIND=C_INT)            :: i,Nelt,res,cpus
  INTEGER(KIND=C_INT)            :: starpu_maj,starpu_min,starpu_rev
  INTEGER(KIND=C_INT)            :: neq,ng,nb,it,it_tot
  REAL(KIND=C_DOUBLE)            :: r, coeff2

  !Initialization with arbitrary data
  Nelt           = 2
  it_tot = 2
  numpar%Neq_max = 5
  numpar%coeff   = 1.0
  ALLOCATE(mesh%elt(Nelt))
  DO i = 1,Nelt
     elt => mesh%elt(i)
     elt%Ng  = 4
     elt%Np  = 2
     ALLOCATE(elt%ro(numpar%Neq_max,elt%Np))
     ALLOCATE(elt%dro(numpar%Neq_max,elt%Np))
     ALLOCATE(elt%basis(elt%Np,elt%Ng))
     CALL init_element(elt%ro,elt%dro,elt%basis,numpar%Neq_max,elt%Np,elt%Ng,i)
  ENDDO

  !Initialization of StarPU
  res = starpu_my_init_c()
  IF (res == -19) THEN
     STOP 77
  END IF
  call starpu_get_version(starpu_maj,starpu_min,starpu_rev)
  WRITE(6,'(a,i4,a,i4,a,i4)')      "StarPU version: ", starpu_maj , "." , starpu_min , "." , starpu_rev
  cpus = starpu_cpu_worker_get_count()
  IF (cpus == 0) THEN
     CALL starpu_shutdown()
     STOP 77
  END IF

  !Registration of elements
  DO i = 1,Nelt
     elt => mesh%elt(i)
     CALL starpu_register_element_c(numpar%Neq_max,elt%Np,elt%Ng,elt%ro,elt%dro, &
                                    elt%basis,elt%ro_h,elt%dro_h,elt%basis_h)
  ENDDO
  !Compute
  DO it = 1,it_tot

     ! compute new dro for each element
     DO i = 1,Nelt
        elt => mesh%elt(i)
        CALL starpu_loop_element_task_c(numpar%coeff,elt%ro_h,elt%dro_h,elt%basis_h)
     ENDDO
     ! sync (if needed by the algorithm)
     CALL starpu_task_wait_for_all()

     ! - - - - -

     ! copy dro to ro for each element
     DO i = 1,Nelt
        elt => mesh%elt(i)
         CALL starpu_copy_element_task_c(elt%ro_h,elt%dro_h)
     ENDDO
     ! sync (if needed by the algorithm)
     CALL starpu_task_wait_for_all()

  ENDDO
  !Unregistration of elements
  DO i = 1,Nelt
     elt => mesh%elt(i)
     CALL starpu_unregister_element_c(elt%ro_h,elt%dro_h,elt%basis_h)
  ENDDO

  !Terminate StarPU, no task can be submitted after
  CALL starpu_shutdown()

  !Check data with StarPU
  WRITE(6,'(a)') " "
  WRITE(6,'(a)') " %%%% RESULTS STARPU %%%% "
  WRITE(6,'(a)') " "
  DO i = 1,Nelt
     WRITE(6,'(a,i4,a)')      " elt ", i , "  ;  elt%ro = "
     WRITE(6,'(10(1x,F11.2))') mesh%elt(i)%ro
     WRITE(6,'(a)')           " ------------------------ "
  ENDDO

  !Same compute without StarPU
  DO i = 1,Nelt
     elt => mesh%elt(i)
     CALL init_element(elt%ro,elt%dro,elt%basis,numpar%Neq_max,elt%Np,elt%Ng,i)
  ENDDO

  DO it = 1, it_tot
     DO i = 1,Nelt
        elt => mesh%elt(i)
        CALL loop_element_cpu(elt%ro,elt%dro,elt%basis,numpar%coeff,numpar%Neq_max,elt%Ng,elt%Np)
        elt%ro = elt%ro + elt%dro
     ENDDO
  ENDDO

  WRITE(6,'(a)') " "
  WRITE(6,'(a)') " %%%% RESULTS VERIFICATION %%%% "
  WRITE(6,'(a)') " "

  DO i = 1,Nelt
     WRITE(6,'(a,i4,a)')      " elt ", i , "  ;  elt%ro = "
     WRITE(6,'(10(1x,F11.2))') mesh%elt(i)%ro
     WRITE(6,'(a)')           " ------------------------ "
  ENDDO

  WRITE(6,'(a)') " "

  !Deallocation
  DO i = 1,Nelt
     elt => mesh%elt(i)
     DEALLOCATE(elt%ro)
     DEALLOCATE(elt%dro)
     DEALLOCATE(elt%basis)
  ENDDO
  DEALLOCATE(mesh%elt)

END PROGRAM f90_example
