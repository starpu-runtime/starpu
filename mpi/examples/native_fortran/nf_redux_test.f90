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
program main
  use iso_c_binding
  use fstarpu_mod
  use fstarpu_mpi_mod

  implicit none

  integer, target                         :: ret, np, i, j
  type(c_ptr)                             :: task_cl, task_rw_cl, task_red_cl, task_ini_cl
  character(kind=c_char,len=*), parameter :: name=C_CHAR_"task"//C_NULL_CHAR
  character(kind=c_char,len=*), parameter :: namered=C_CHAR_"task_red"//C_NULL_CHAR
  character(kind=c_char,len=*), parameter :: nameini=C_CHAR_"task_ini"//C_NULL_CHAR
  real(kind(1.d0)), target                :: a1, a2, b1, b2
  integer(kind=8)                          :: tag, err
  type(c_ptr)                             :: a1hdl, a2hdl, b1hdl, b2hdl
  integer, target                         :: comm, comm_world, comm_w_rank, comm_size
  integer(c_int), target                  :: w_node

  call fstarpu_fxt_autostart_profiling(0)
  ret = fstarpu_init(c_null_ptr)
  ret = fstarpu_mpi_init(1)

  comm_world = fstarpu_mpi_world_comm()
  comm_w_rank  = fstarpu_mpi_world_rank()
  comm_size  = fstarpu_mpi_world_size()
  if (comm_size.ne.4) then
    write(*,'(" ")')
    write(*,'("This application is meant to run with 4 MPI")')
    stop 1
  end if
  err   = fstarpu_mpi_barrier(comm_world)

  if(comm_w_rank.eq.0) then
    write(*,'(" ")')
    a1 = 1.0
    write(*,*) "init_a1", a1
    b1 = 0.5
    write(*,*) "init b1", b1
  end if
  if(comm_w_rank.eq.1) then
    write(*,'(" ")')
    a2 = 2.0
    write(*,*) "init_a2", a2
    b2 = 0.8
    write(*,*) "init b2", b2
  end if

  ! allocate and fill codelet structs
  task_cl = fstarpu_codelet_allocate()
  call fstarpu_codelet_set_name(task_cl, name)
  call fstarpu_codelet_add_cpu_func(task_cl, C_FUNLOC(cl_cpu_task))
  call fstarpu_codelet_add_buffer(task_cl, FSTARPU_REDUX)
  call fstarpu_codelet_add_buffer(task_cl, FSTARPU_R)

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

  err = fstarpu_mpi_barrier(comm_world)

  tag = 0
  if(comm_w_rank.eq.0) then
        call fstarpu_variable_data_register(a1hdl, 0, c_loc(a1),c_sizeof(a1))
        call fstarpu_variable_data_register(b1hdl, 0, c_loc(b1),c_sizeof(b1))
  else
        call fstarpu_variable_data_register(a1hdl, -1, c_null_ptr,c_sizeof(a1))
        call fstarpu_variable_data_register(b1hdl, -1, c_null_ptr,c_sizeof(b1))
  end if
  call fstarpu_mpi_data_register(a1hdl,tag,0)
  call fstarpu_mpi_data_register(b1hdl, tag+1,0)

  tag = tag + 2
  if(comm_w_rank.eq.1) then
        call fstarpu_variable_data_register(a2hdl, 0, c_loc(a2),c_sizeof(a2))
        call fstarpu_variable_data_register(b2hdl, 0, c_loc(b2),c_sizeof(b2))
  else
        call fstarpu_variable_data_register(a2hdl, -1, c_null_ptr,c_sizeof(a2))
        call fstarpu_variable_data_register(b2hdl, -1, c_null_ptr,c_sizeof(b2))
  end if
  call fstarpu_mpi_data_register(a2hdl,tag,1)
  call fstarpu_mpi_data_register(b2hdl, tag+1, 1)
  tag = tag + 2

  call fstarpu_data_set_reduction_methods(a1hdl, task_red_cl,task_ini_cl)
  call fstarpu_data_set_reduction_methods(a2hdl, task_red_cl,task_ini_cl)

  err = fstarpu_mpi_barrier(comm_world)

  call fstarpu_fxt_start_profiling()

  w_node = 3
  comm = comm_world
  call fstarpu_mpi_task_insert( (/ c_loc(comm),   &
             task_cl,                                         &
             FSTARPU_REDUX, a1hdl,                            &
             FSTARPU_R, b1hdl,                                &
             FSTARPU_EXECUTE_ON_NODE, c_loc(w_node),          &
             C_NULL_PTR /))
  w_node = 2
  comm = comm_world
  call fstarpu_mpi_task_insert( (/ c_loc(comm),   &
             task_cl,                                         &
             FSTARPU_REDUX, a2hdl,                            &
             FSTARPU_R, b2hdl,                                &
             FSTARPU_EXECUTE_ON_NODE, c_loc(w_node),          &
             C_NULL_PTR /))

  call fstarpu_mpi_redux_data(comm_world, a1hdl)
  call fstarpu_mpi_redux_data(comm_world, a2hdl)
  ! write(*,*) "waiting all tasks ..."
  err = fstarpu_mpi_wait_for_all(comm_world)

  if(comm_w_rank.eq.0) then
     write(*,*) 'computed result ---> ',a1, "expected =",4.5
  end if
  if(comm_w_rank.eq.1) then
     write(*,*) 'computed result ---> ',a2, "expected=",5.8
  end if
  call fstarpu_data_unregister(a1hdl)
  call fstarpu_data_unregister(a2hdl)
  call fstarpu_data_unregister(b1hdl)
  call fstarpu_data_unregister(b2hdl)

  call fstarpu_fxt_stop_profiling()
  call fstarpu_codelet_free(task_cl)
  call fstarpu_codelet_free(task_red_cl)
  call fstarpu_codelet_free(task_ini_cl)


  err = fstarpu_mpi_shutdown()
  call fstarpu_shutdown()

  stop

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
    a = 3.0 + b
    write(*,*) "task   (c_w_rank:",comm_rank,") from ",old_a,"to",a

    return
  end subroutine cl_cpu_task

  recursive subroutine cl_cpu_task_red (buffers, cl_args) bind(C)
    use iso_c_binding       ! C interfacing module
    use fstarpu_mod         ! StarPU interfacing module
    implicit none

    type(c_ptr), value, intent(in) :: buffers, cl_args ! cl_args is unused
    integer(c_int) :: ret
    integer, target                         :: comm_rank
    real(kind(1.d0)), pointer :: as, ad
    real(kind(1.d0))           :: old_ad

    comm_rank  = fstarpu_mpi_world_rank()
    call c_f_pointer(fstarpu_variable_get_ptr(buffers, 0), ad)
    call c_f_pointer(fstarpu_variable_get_ptr(buffers, 1), as)
    old_ad = ad
    ad = ad + as
    call fstarpu_sleep(real(0.01, c_float))
    write(*,*) "red_cl (c_w_rank:",comm_rank,")",as, old_ad, ' ---> ',ad

    return
  end subroutine cl_cpu_task_red

  recursive subroutine cl_cpu_task_ini (buffers, cl_args) bind(C)
    use iso_c_binding       ! C interfacing module
    use fstarpu_mod         ! StarPU interfacing module
    implicit none

    type(c_ptr), value, intent(in) :: buffers, cl_args
        ! cl_args is unused
    integer(c_int) :: ret
    integer, target                         :: comm_rank
    real(kind(1.d0)), pointer :: a

    comm_rank  = fstarpu_mpi_world_rank()
    call c_f_pointer(fstarpu_variable_get_ptr(buffers, 0), a)
    call fstarpu_sleep(real(0.005, c_float))
    a = 0.0
    write(*,*) "ini_cl (c_w_rank:",comm_rank,")"
    return
  end subroutine cl_cpu_task_ini

end program main
