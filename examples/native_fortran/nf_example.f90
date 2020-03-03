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

  USE nf_types
  USE fstarpu_mod
  USE nf_compute
  USE iso_c_binding

  IMPLICIT NONE

  TYPE(type_mesh)                :: mesh
  TYPE(type_numpar),TARGET       :: numpar
  TYPE(type_mesh_elt),POINTER    :: elt   => NULL()
  INTEGER(KIND=C_INT)            :: i,Nelt,res,cpus
  INTEGER(KIND=C_INT)            :: starpu_maj,starpu_min,starpu_rev
  INTEGER(KIND=C_INT)            :: neq,ng,nb,it,it_tot
  REAL(KIND=C_DOUBLE)            :: r, coeff2
  REAL(KIND=C_DOUBLE),TARGET     :: flops

  TYPE(C_PTR) :: cl_loop_element = C_NULL_PTR ! loop codelet
  TYPE(C_PTR) :: cl_copy_element = C_NULL_PTR ! copy codelet

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
  res = fstarpu_init(C_NULL_PTR)
  IF (res == -19) THEN
     STOP 77
  END IF
  CALL fstarpu_get_version(starpu_maj,starpu_min,starpu_rev)
  WRITE(6,'(a,i4,a,i4,a,i4)')      "StarPU version: ", starpu_maj , "." , starpu_min , "." , starpu_rev
  cpus = fstarpu_cpu_worker_get_count()
  IF (cpus == 0) THEN
     CALL fstarpu_shutdown()
     STOP 77
  END IF

  cl_loop_element = fstarpu_codelet_allocate()
  CALL fstarpu_codelet_add_cpu_func(cl_loop_element, C_FUNLOC(loop_element_cpu_fortran))
  CALL fstarpu_codelet_add_buffer(cl_loop_element, FSTARPU_R)
  CALL fstarpu_codelet_add_buffer(cl_loop_element, FSTARPU_RW)
  CALL fstarpu_codelet_add_buffer(cl_loop_element, FSTARPU_R)
  CALL fstarpu_codelet_set_name(cl_loop_element, C_CHAR_"LOOP_ELEMENT"//C_NULL_CHAR)

  cl_copy_element = fstarpu_codelet_allocate()
  CALL fstarpu_codelet_add_cpu_func(cl_copy_element, C_FUNLOC(copy_element_cpu_fortran))
  CALL fstarpu_codelet_add_buffer(cl_copy_element, FSTARPU_RW)
  CALL fstarpu_codelet_add_buffer(cl_copy_element, FSTARPU_R)
  CALL fstarpu_codelet_set_name(cl_copy_element, C_CHAR_"COPY_ELEMENT"//C_NULL_CHAR)

  !Registration of elements
  DO i = 1,Nelt
     elt => mesh%elt(i)
     call fstarpu_matrix_data_register(elt%ro_h, 0, c_loc(elt%ro), numpar%Neq_max, numpar%Neq_max, elt%Np, c_sizeof(elt%ro(1,1)))
     call fstarpu_matrix_data_register(elt%dro_h, 0, c_loc(elt%dro), numpar%Neq_max, numpar%Neq_max, elt%Np, c_sizeof(elt%dro(1,1)))
     call fstarpu_matrix_data_register(elt%basis_h, 0, c_loc(elt%basis), elt%Np, elt%Np, elt%Ng, c_sizeof(elt%basis(1,1)))
  ENDDO
  !Compute
  DO it = 1,it_tot

     ! compute new dro for each element
     DO i = 1,Nelt
        elt => mesh%elt(i)
        flops = elt%Ng * ( (elt%Np * numpar%Neq_max * 2) + 1 + elt%Np * numpar%Neq_max)
        CALL fstarpu_insert_task((/ cl_loop_element,    &
                FSTARPU_VALUE, c_loc(numpar%coeff), FSTARPU_SZ_C_DOUBLE, &
                FSTARPU_R, elt%ro_h,                 &
                FSTARPU_RW, elt%dro_h,                &
                FSTARPU_R, elt%basis_h,              &
                FSTARPU_FLOPS, c_loc(flops),         &
                C_NULL_PTR /))
     ENDDO
     ! sync (if needed by the algorithm)
     CALL fstarpu_task_wait_for_all()

     ! - - - - -

     ! copy dro to ro for each element
     DO i = 1,Nelt
        elt => mesh%elt(i)
        CALL fstarpu_insert_task((/ cl_copy_element,    &
                FSTARPU_RW, elt%ro_h,                 &
                FSTARPU_R, elt%dro_h,                &
                C_NULL_PTR /))
     ENDDO
     ! sync (if needed by the algorithm)
     CALL fstarpu_task_wait_for_all()

  ENDDO
  !Unregistration of elements
  DO i = 1,Nelt
     elt => mesh%elt(i)
     CALL fstarpu_data_unregister(elt%ro_h)
     CALL fstarpu_data_unregister(elt%dro_h)
     CALL fstarpu_data_unregister(elt%basis_h)
  ENDDO

  !Terminate StarPU, no task can be submitted after
  CALL fstarpu_shutdown()

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
  CALL fstarpu_codelet_free(cl_loop_element)
  CALL fstarpu_codelet_free(cl_copy_element)
  DO i = 1,Nelt
     elt => mesh%elt(i)
     DEALLOCATE(elt%ro)
     DEALLOCATE(elt%dro)
     DEALLOCATE(elt%basis)
  ENDDO
  DEALLOCATE(mesh%elt)

END PROGRAM f90_example
