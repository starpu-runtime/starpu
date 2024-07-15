! StarPU --- Runtime system for heterogeneous multicore architectures.
!
! Copyright (C) 2024-2024  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
program nf_context
  use iso_c_binding
  use fstarpu_mod
  use fstarpu_mpi_mod
  implicit none

  type block
     real(kind(1.d0)), allocatable :: x(:,:)
     type(c_ptr)                   :: h
  end type block

  integer(c_int), target                :: ncpus, nworkers,  ctx
  integer, target                       :: ret, i, j, m, n, mpi_rank, mpi_size, mpi_comm
  integer(c_int), allocatable           :: workers(:)
  type(block), allocatable, target      :: a(:)
  character(kind=c_char,len=6), target  :: ctx_policy_eager  = C_CHAR_"eager"//C_NULL_CHAR
  type(c_ptr)                           :: task_cl   = c_null_ptr, ptr
  real(kind(1.d0))                      :: dtype
  integer(c_int64_t)                    :: tag

  interface
     recursive subroutine task_cpu_func(buffers, cl_arg) bind(C)
       use iso_c_binding
       type(c_ptr), value        :: cl_arg
       type(c_ptr), value        :: buffers
     end subroutine task_cpu_func
  end interface

  interface
     subroutine strtoptr(s, ptr) bind(c)
       use iso_c_binding
       type(c_ptr) :: ptr
       character(kind=c_char), dimension(*) :: s
     end subroutine strtoptr
  end interface

  m  = 2
  n  = 2

  ret = fstarpu_init(c_null_ptr)
  ret = fstarpu_mpi_init(1)
  mpi_comm = fstarpu_mpi_world_comm()
  mpi_size = fstarpu_mpi_world_size()
  mpi_rank = fstarpu_mpi_world_rank()

  nworkers = fstarpu_worker_get_count()
  allocate(workers(nworkers))
  ! ret = fstarpu_mpi_barrier(mpi_comm)
  if(mpi_rank.eq.0) write(*,'("Starpu inited (",i1,").  Has ",i2," CPU workers")')ret, nworkers
  ! ret = fstarpu_mpi_barrier(mpi_comm)

  task_cl = fstarpu_codelet_allocate()
  call fstarpu_codelet_add_cpu_func(task_cl, C_FUNLOC(task_cpu_func))
  call fstarpu_codelet_set_variable_nbuffers(task_cl)
  call fstarpu_codelet_set_name(task_cl, C_CHAR_"my task"//C_NULL_CHAR)

  i = fstarpu_worker_get_ids_by_type(FSTARPU_CPU_WORKER, workers, nworkers)
  ! write(*,*)workers(1:nworkers)
  call strtoptr(ctx_policy_eager, ptr)
  ctx = fstarpu_sched_ctx_create(workers, nworkers,   &
        C_CHAR_"qrm_ctx"//C_NULL_CHAR,                &
        (/ FSTARPU_SCHED_CTX_POLICY_NAME, ptr,        &
        c_null_ptr /))

  allocate(a(mpi_size))

  tag = 0
  do i=1, mpi_size
     if(mpi_rank.eq.i-1) then
        allocate(a(i)%x(m,n))
        a(i)%h = c_null_ptr
        call random_number(a(i)%x)
        call fstarpu_matrix_data_register(a(i)%h, 0,  &
             c_loc(a(i)%x),                           &
             m,                                       &
             m,                                       &
             n,                                       &
             c_sizeof(dtype))
     else
        call fstarpu_matrix_data_register(a(i)%h, -1, &
             c_null_ptr,                              &
             m,                                       &
             m,                                       &
             n,                                       &
             c_sizeof(dtype))
     end if
     tag = tag+1
     call fstarpu_mpi_data_register_comm(a(i)%h, &
          tag, i-1, mpi_comm)
  end do

  ret = fstarpu_mpi_barrier(mpi_comm)

  do i=1, mpi_size
     if(mpi_rank.eq.i-1) then
        call fstarpu_mpi_task_insert( [ c_loc(mpi_comm),       &
             task_cl,                                          &
             FSTARPU_VALUE, c_loc(mpi_rank), FSTARPU_SZ_C_INT, &
             FSTARPU_RW, a(i)%h,                               &
             FSTARPU_SCHED_CTX, c_loc(ctx),                    &
             C_NULL_PTR ])
     end if
  end do

  ! if only wait for tasks in ctx I have a segfault
  !call fstarpu_task_wait_for_all_in_ctx(ctx)

  ! if wait for all tasks (regardless of ctx) it works
  call fstarpu_task_wait_for_all()

  ret = fstarpu_mpi_barrier(mpi_comm)
  if(mpi_rank.eq.0) write(*,'("Yuppi, all the tasks in ctx",i1," ave finished!")')ctx


  call fstarpu_codelet_free(task_cl     );   task_cl   = c_null_ptr
  call fstarpu_shutdown()
  ret = fstarpu_mpi_shutdown()

  stop

end program nf_context


subroutine f_sleep(t)
  implicit none
  integer :: t_start, t_end, t_rate
  real(kind(1.d0))     :: ta, t
  call system_clock(t_start)
  do
     call system_clock(t_end, t_rate)
     ta = real(t_end-t_start)/real(t_rate)
     if(ta.gt.t) return
  end do
end subroutine f_sleep


recursive subroutine task_cpu_func(buffers, cl_arg) bind(C)
  use fstarpu_mod
  implicit none

  type(c_ptr), value        :: cl_arg
  type(c_ptr), value        :: buffers

  real(kind(1.d0)), pointer :: x(:,:)
  integer, target           :: mpi_rank
  integer                   :: mx, nx, ldx

  call fstarpu_unpack_arg(cl_arg,(/c_loc(mpi_rank)/))

  mx   = fstarpu_matrix_get_nx(buffers, 0)
  nx   = fstarpu_matrix_get_ny(buffers, 0)
  ldx  = fstarpu_matrix_get_ld(buffers, 0)

  call c_f_pointer(fstarpu_matrix_get_ptr(buffers, 0), x, shape=(/ldx,nx/))

  write(*,'("Here I am ",3(i3,2x))')mx, nx, mpi_rank
  x = real(mpi_rank, kind(1.d0))

  return
end subroutine task_cpu_func
