! StarPU --- Runtime system for heterogeneous multicore architectures.
!
! Copyright (C) 2016-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
program nf_sched_ctx
  use iso_c_binding       ! C interfacing module
  use fstarpu_mod         ! StarPU interfacing module
  implicit none

  type(c_ptr) :: cl1   ! a pointer for a codelet structure
  type(c_ptr) :: cl2   ! a pointer for another codelet structure
  integer(c_int) :: err   ! return status for fstarpu_init
  integer(c_int) :: ncpu  ! number of cpus workers

  ! list of cpu worker ids
  integer(c_int), dimension(:), allocatable :: procs

  ! sub-list of cpu worker ids for sched context 1
  integer(c_int) :: nprocs1, nprocs2
  integer(c_int), dimension(:), allocatable :: procs1, procs2
  integer(c_int) :: ctx1, ctx2

  ! needed to be able to call c_loc on it, to get a ptr to the string
  character(kind=c_char,len=11), target :: ctx_policy = C_NULL_CHAR

  integer(c_int),parameter :: n = 4
  integer(c_int) :: i
  integer(c_int), target :: arg_id
  integer(c_int), target :: arg_ctx
  integer(kind=c_int), parameter :: color1  = int(z'ff0000', kind=c_int)
  integer(kind=c_int), parameter :: color2  = int(z'00ff00', kind=c_int)

  ! initialize StarPU with default settings
  err = fstarpu_init(C_NULL_PTR)
  if (err == -19) then
     stop 76
  end if

  ! stop there if no CPU worker available
  ncpu = fstarpu_cpu_worker_get_count()
  if (ncpu == 0) then
     call fstarpu_shutdown()
     stop 77
  end if

  ! allocate and fill codelet structs
  cl1 = fstarpu_codelet_allocate()
  call fstarpu_codelet_set_name(cl1, C_CHAR_"TASK1"//C_NULL_CHAR)
  call fstarpu_codelet_add_cpu_func(cl1, C_FUNLOC(cl1_cpu_func))
  call fstarpu_codelet_set_color(cl1, color1)

  ! allocate and fill codelet structs
  cl2 = fstarpu_codelet_allocate()
  call fstarpu_codelet_set_name(cl2, C_CHAR_"TASK2"//C_NULL_CHAR)
  call fstarpu_codelet_add_cpu_func(cl2, C_FUNLOC(cl2_cpu_func))
  call fstarpu_codelet_set_color(cl2, color2)

  ! get the list of CPU worker ids
  allocate(procs(ncpu))
  err = fstarpu_worker_get_ids_by_type(FSTARPU_CPU_WORKER, procs, ncpu)

  ! split the workers in two sets
  write(*,*)ncpu,' cpus available'
  nprocs1 = max(ncpu/2,1);
  allocate(procs1(nprocs1))
  write(*,*) "procs1:"
  do i=1,nprocs1
     procs1(i) = procs(i)
     write(*,*) '   ',i, procs1(i)
  end do

  nprocs2 = max(ncpu - nprocs1,1);
  allocate(procs2(nprocs2))
  write(*,*) "procs2:"
  do i=1,nprocs2
     procs2(i) = procs(i+min(nprocs1,ncpu-nprocs2))
     write(*,*) '   ',i, procs2(i)
  end do

  ! create sched context 1 with default policy, by giving a NULL policy name
  ctx1 = fstarpu_sched_ctx_create(procs1, nprocs1,  &
       C_CHAR_"ctx1"//C_NULL_CHAR, &
       (/ FSTARPU_SCHED_CTX_POLICY_NAME, c_loc(ctx_policy), c_null_ptr /) &
       )

  call fstarpu_sched_ctx_display_workers(ctx1)

  ! create sched context 1 with default policy, by giving a NULL policy name
  ctx2 = fstarpu_sched_ctx_create(procs2, nprocs2,  &
       C_CHAR_"ctx2"//C_NULL_CHAR, &
       (/ FSTARPU_SCHED_CTX_POLICY_NAME, c_loc(ctx_policy), c_null_ptr /) &
       )

  call fstarpu_sched_ctx_display_workers(ctx2)

  ! submit a task on context 1
  arg_id = 1000
  arg_ctx = ctx1
  call fstarpu_insert_task((/ cl1, &
       FSTARPU_VALUE, c_loc(arg_id), FSTARPU_SZ_C_INT, &
       FSTARPU_SCHED_CTX, c_loc(arg_ctx), &
       C_NULL_PTR /))

  ! submit a task on context 2
  arg_id = 1000
  arg_ctx = ctx2
  call fstarpu_insert_task((/ cl2, &
       FSTARPU_VALUE, c_loc(arg_id), FSTARPU_SZ_C_INT, &
       FSTARPU_SCHED_CTX, c_loc(arg_ctx), &
       C_NULL_PTR /))

  ! mark submission process as completed on context 1
  call fstarpu_sched_ctx_finished_submit(ctx1)

  ! mark submission process as completed on context 2
  call fstarpu_sched_ctx_finished_submit(ctx2)

  ! wait for completion of all tasks
  call fstarpu_task_wait_for_all()

  ! deallocate both contexts
  call fstarpu_sched_ctx_delete(ctx1)
  call fstarpu_sched_ctx_delete(ctx2)

  deallocate(procs1)
  deallocate(procs2)

  ! free codelet structure
  call fstarpu_codelet_free(cl1)
  call fstarpu_codelet_free(cl2)

  ! shut StarPU down
  call fstarpu_shutdown()

  stop

contains
  recursive subroutine cl1_cpu_func (buffers, cl_args) bind(C)
    use iso_c_binding       ! C interfacing module
    use fstarpu_mod         ! StarPU interfacing module
    implicit none

    interface
       function sleep(s) bind(C)
         use iso_c_binding
         integer(c_int) :: sleep
         integer(c_int), value, intent(in) :: s
       end function sleep
    end interface

    type(c_ptr), value, intent(in) :: buffers, cl_args ! cl_args is unused
    integer(c_int),target :: id
    integer(c_int) :: worker_id
    integer(c_int) :: ret

    call fstarpu_unpack_arg(cl_args,(/ c_loc(id) /))
    ret = sleep(1)
    worker_id = fstarpu_worker_get_id()

    write(*,*) "task1:", id, ", worker_id:", worker_id
  end subroutine cl1_cpu_func

  recursive subroutine cl2_cpu_func (buffers, cl_args) bind(C)
    use iso_c_binding       ! C interfacing module
    use fstarpu_mod         ! StarPU interfacing module
    implicit none

    interface
       function sleep(s) bind(C)
         use iso_c_binding
         integer(c_int) :: sleep
         integer(c_int), value, intent(in) :: s
       end function sleep
    end interface

    type(c_ptr), value, intent(in) :: buffers, cl_args ! cl_args is unused
    integer(c_int),target :: id
    integer(c_int) :: worker_id
    integer(c_int) :: ret

    call fstarpu_unpack_arg(cl_args,(/ c_loc(id) /))
    ret = sleep(1)
    worker_id = fstarpu_worker_get_id()

    write(*,*) "task2:", id, ", worker_id:", worker_id
  end subroutine cl2_cpu_func



end program nf_sched_ctx
