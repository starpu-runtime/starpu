! StarPU --- Runtime system for heterogeneous multicore architectures.
!
! Copyright (C) 2016-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
program nf_mpi_redux
  use iso_c_binding
  use fstarpu_mod
  use fstarpu_mpi_mod

  implicit none

  integer, target                         :: ret, np, i, j, trial
  type(c_ptr)                             :: work_cl, task_rw_cl,task_red_cl, task_ini_cl
  character(kind=c_char,len=*), parameter :: name=C_CHAR_"task"//C_NULL_CHAR
  character(kind=c_char,len=*), parameter :: namered=C_CHAR_"task_red"//C_NULL_CHAR
  character(kind=c_char,len=*), parameter :: nameini=C_CHAR_"task_ini"//C_NULL_CHAR
  real(kind(1.d0)), target                :: a,tmp
  real(kind(1.d0)), target, allocatable   :: b(:)
  integer(kind=8)                         :: tag, err
  type(c_ptr)                             :: ahdl
  type(c_ptr), target, allocatable        :: bhdl(:)
  type(c_ptr)                             :: task_mode, codelet_mode
  integer, target                         :: comm_world,comm_w_rank, comm_size
  integer(c_int), target                  :: w_node, nworkers, work_coef

  call fstarpu_fxt_autostart_profiling(0)
  ret = fstarpu_init(c_null_ptr)
  ret = fstarpu_mpi_init(1)

  comm_world = fstarpu_mpi_world_comm()
  comm_w_rank  = fstarpu_mpi_world_rank()
  comm_size  = fstarpu_mpi_world_size()
  if (comm_size.lt.2) then
    write(*,'(" ")')
    write(*,'("This application is meant to run with at least two nodes (found ",i4," ; i am ",i4,").")') comm_size, comm_w_rank
    stop 2
  end if
  allocate(b(comm_size-1), bhdl(comm_size-1))
  nworkers = fstarpu_worker_get_count()
  if (nworkers.lt.1) then
    write(*,'(" ")')
    write(*,'("This application is meant to run with at least one worker per node.")')
    stop 2
  end if

  ! allocate and reduction codelets
  task_red_cl = fstarpu_codelet_allocate()
  call fstarpu_codelet_set_name(task_red_cl, namered)
  call fstarpu_codelet_add_cpu_func(task_red_cl,C_FUNLOC(cl_cpu_task_red))
  call fstarpu_codelet_add_buffer(task_red_cl, FSTARPU_RW.ior.FSTARPU_COMMUTE)
  call fstarpu_codelet_add_buffer(task_red_cl, FSTARPU_R)

  task_ini_cl = fstarpu_codelet_allocate()
  call fstarpu_codelet_set_name(task_ini_cl, nameini)
  call fstarpu_codelet_add_cpu_func(task_ini_cl,C_FUNLOC(cl_cpu_task_ini))
  call fstarpu_codelet_add_buffer(task_ini_cl, FSTARPU_W)

  work_coef=2

  do trial=1,2

    if (trial.eq.2) then
          write(*,*) "Using STARPU_MPI_REDUX"
          codelet_mode = FSTARPU_RW.ior.FSTARPU_COMMUTE
          task_mode = FSTARPU_MPI_REDUX
    else if (trial.eq.1) then
          write(*,*) "Using STARPU_REDUX"
          codelet_mode = FSTARPU_REDUX
          task_mode = FSTARPU_REDUX
    end if
    ! allocate and fill codelet structs
    work_cl = fstarpu_codelet_allocate()
    call fstarpu_codelet_set_name(work_cl, name)
    call fstarpu_codelet_add_cpu_func(work_cl, C_FUNLOC(cl_cpu_task))
    call fstarpu_codelet_add_buffer(work_cl, codelet_mode)
    call fstarpu_codelet_add_buffer(work_cl, FSTARPU_R)
    err = fstarpu_mpi_barrier(comm_world)

    if(comm_w_rank.eq.0) then
      write(*,'(" ")')
      a = 1.0
      write(*,*) "init a = ", a
    else
      b(comm_w_rank) = 1.0 / (comm_w_rank + 1.0)
      write(*,*) "init b_",comm_w_rank,"=", b(comm_w_rank)
    end if

    err = fstarpu_mpi_barrier(comm_world)

    tag = 0
    if(comm_w_rank.eq.0) then
      call fstarpu_variable_data_register(ahdl, 0, c_loc(a),c_sizeof(a))
      do i=1,comm_size-1
          call fstarpu_variable_data_register(bhdl(i), -1, c_null_ptr,c_sizeof(b(i)))
      end do
    else
      call fstarpu_variable_data_register(ahdl, -1, c_null_ptr,c_sizeof(a))
      do i=1,comm_size-1
        if (i.eq.comm_w_rank) then
          call fstarpu_variable_data_register(bhdl(i), 0, c_loc(b(i)),c_sizeof(b(i)))
        else
          call fstarpu_variable_data_register(bhdl(i), -1, c_null_ptr,c_sizeof(b(i)))
        end if
      end do
    end if
    call fstarpu_mpi_data_register(ahdl,  tag,  0)
    do i=1,comm_size-1
       call fstarpu_mpi_data_register(bhdl(i), tag+i,i)
    end do

    tag = tag + comm_size

    call fstarpu_data_set_reduction_methods(ahdl,task_red_cl,task_ini_cl)

    err = fstarpu_mpi_barrier(comm_world)


    call fstarpu_fxt_start_profiling()
    do w_node=1,comm_size-1
      do i=1,work_coef*nworkers
        call fstarpu_mpi_task_insert( (/ c_loc(comm_world),   &
               work_cl,                                         &
               task_mode, ahdl,                            &
               FSTARPU_R, bhdl(w_node),                      &
               FSTARPU_EXECUTE_ON_NODE, c_loc(w_node),          &
               C_NULL_PTR /))
      end do
    end do
    call fstarpu_mpi_redux_data(comm_world, ahdl)
    err = fstarpu_mpi_wait_for_all(comm_world)

    if(comm_w_rank.eq.0) then
      tmp = 0
      do w_node=1,comm_size-1
        tmp = tmp + 1.0 / (w_node+1.0)
      end do
      write(*,*) 'computed result ---> ',a, "expected =",&
        1.0 + (comm_size-1.0)*(comm_size)/2.0 + work_coef*nworkers*((comm_size-1.0)*3.0 + tmp)
    end if
    err = fstarpu_mpi_barrier(comm_world)
    call fstarpu_data_unregister(ahdl)
    do w_node=1,comm_size-1
      call fstarpu_data_unregister(bhdl(w_node))
    end do
    call fstarpu_codelet_free(work_cl)

  end do

  call fstarpu_fxt_stop_profiling()
  call fstarpu_codelet_free(task_red_cl)
  call fstarpu_codelet_free(task_ini_cl)


  err = fstarpu_mpi_shutdown()
  call fstarpu_shutdown()
  deallocate(b, bhdl)
  stop 0

contains

  recursive subroutine cl_cpu_task (buffers, cl_args) bind(C)
    use iso_c_binding       ! C interfacing module
    use fstarpu_mod         ! StarPU interfacing module
    implicit none

    type(c_ptr), value, intent(in) :: buffers, cl_args ! cl_args is unused
    integer(c_int) :: ret, worker_id
    integer        :: comm_rank
    integer, target :: i
    real(kind(1.d0)), pointer :: a, b
    real(kind(1.d0))          :: old_a

    worker_id = fstarpu_worker_get_id()
    comm_rank  = fstarpu_mpi_world_rank()

    call c_f_pointer(fstarpu_variable_get_ptr(buffers, 0), a)
    call c_f_pointer(fstarpu_variable_get_ptr(buffers, 1), b)
    call fstarpu_sleep(real(0.01, c_float))
    old_a = a
    a = old_a + 3.0 + b
    write(*,*) "task   (c_w_rank:",comm_rank," worker_id:",worker_id,") from ",old_a,"to",a

    return
  end subroutine cl_cpu_task

  recursive subroutine cl_cpu_task_red (buffers, cl_args) bind(C)
    use iso_c_binding       ! C interfacing module
    use fstarpu_mod         ! StarPU interfacing module
    implicit none

    type(c_ptr), value, intent(in) :: buffers, cl_args ! cl_args is unused
    integer(c_int) :: ret, worker_id
    integer, target                         :: comm_rank
    real(kind(1.d0)), pointer :: as, ad
    real(kind(1.d0))           :: old_ad
    worker_id = fstarpu_worker_get_id()
    comm_rank  = fstarpu_mpi_world_rank()
    call c_f_pointer(fstarpu_variable_get_ptr(buffers, 0), ad)
    call c_f_pointer(fstarpu_variable_get_ptr(buffers, 1), as)
    old_ad = ad
    ad = ad + as
    call fstarpu_sleep(real(0.01, c_float))
    write(*,*) "red_cl (c_w_rank:",comm_rank,"worker_id:",worker_id,")",as, old_ad, ' ---> ',ad

    return
  end subroutine cl_cpu_task_red

  recursive subroutine cl_cpu_task_ini (buffers, cl_args) bind(C)
    use iso_c_binding       ! C interfacing module
    use fstarpu_mod         ! StarPU interfacing module
    implicit none

    type(c_ptr), value, intent(in) :: buffers, cl_args
        ! cl_args is unused
    integer(c_int) :: ret, worker_id
    integer, target                         :: comm_rank
    real(kind(1.d0)), pointer :: a
    worker_id = fstarpu_worker_get_id()
    comm_rank  = fstarpu_mpi_world_rank()
    call c_f_pointer(fstarpu_variable_get_ptr(buffers, 0), a)
    call fstarpu_sleep(real(0.005, c_float))
    ! As this codelet is run by each worker in the REDUX mode case
    ! this initialization makes salient the number of copies spawned
    write(*,*) "ini_cl (c_w_rank:",comm_rank,"worker_id:",worker_id,") set to", comm_rank
    a = comm_rank
    return
  end subroutine cl_cpu_task_ini

end program
