! StarPU --- Runtime system for heterogeneous multicore architectures.
!
! Copyright (C) 2016-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
        use nf_sched_ctx_cl
        implicit none

        type(c_ptr) :: cl1   ! a pointer for a codelet structure
        type(c_ptr) :: cl2   ! a pointer for another codelet structure
        integer(c_int) :: err   ! return status for fstarpu_init
        integer(c_int) :: ncpu  ! number of cpus workers

        ! list of cpu worker ids
        integer(c_int), dimension(:), allocatable :: procs

        ! sub-list of cpu worker ids for sched context 1
        integer(c_int) :: nprocs1
        integer(c_int), dimension(:), allocatable :: procs1
        integer(c_int) :: ctx1


        ! sub-list of cpu worker ids for sched context 2
        integer(c_int) :: nprocs2
        integer(c_int), dimension(:), allocatable :: procs2
        integer(c_int) :: ctx2

        ! needed to be able to call c_loc on it, to get a ptr to the string
        character(kind=c_char,len=6), target :: ctx2_policy = C_CHAR_"eager"//C_NULL_CHAR

        integer(c_int),parameter :: n = 20
        integer(c_int) :: i
        integer(c_int), target :: arg_id
        integer(c_int), target :: arg_ctx

        ! initialize StarPU with default settings
        err = fstarpu_init(C_NULL_PTR)
        if (err == -19) then
                stop 77
        end if

        ! stop there if no CPU worker available
        ncpu = fstarpu_cpu_worker_get_count()
        if (ncpu == 0) then
                call fstarpu_shutdown()
                stop 77
        end if

        ! actually we really need at least 2 CPU workers such to allocate 2 non overlapping contexts
        if (ncpu < 2) then
                call fstarpu_shutdown()
                stop 77
        end if

        ! allocate and fill codelet structs
        cl1 = fstarpu_codelet_allocate()
        call fstarpu_codelet_set_name(cl1, C_CHAR_"sched_ctx_cl1"//C_NULL_CHAR)
        call fstarpu_codelet_add_cpu_func(cl1, C_FUNLOC(cl_cpu_func_sched_ctx))

        ! allocate and fill codelet structs
        cl2 = fstarpu_codelet_allocate()
        call fstarpu_codelet_set_name(cl2, C_CHAR_"sched_ctx_cl2"//C_NULL_CHAR)
        call fstarpu_codelet_add_cpu_func(cl2, C_FUNLOC(cl_cpu_func_sched_ctx))

        ! get the list of CPU worker ids
        allocate(procs(ncpu))
        err = fstarpu_worker_get_ids_by_type(FSTARPU_CPU_WORKER, procs, ncpu)

        ! split the workers in two sets

        nprocs1 = ncpu/2;
        allocate(procs1(nprocs1))
        write(*,*) "procs1:"
        do i=1,nprocs1
                procs1(i) = procs(i)
                write(*,*) i, procs1(i)
        end do

        nprocs2 = ncpu - nprocs1
        allocate(procs2(nprocs2))
        write(*,*) "procs2:"
        do i=1,nprocs2
                procs2(i) = procs(nprocs1+i)
                write(*,*) i, procs2(i)
        end do
        deallocate(procs)

        ! create sched context 1 with default policy, by giving a NULL policy name
        ctx1 = fstarpu_sched_ctx_create(procs1, nprocs1,  &
            C_CHAR_"ctx1"//C_NULL_CHAR, &
            (/ FSTARPU_SCHED_CTX_POLICY_NAME, c_null_ptr, c_null_ptr /) &
            )

        ! create sched context 2 with a user selected policy name
        ctx2 = fstarpu_sched_ctx_create(procs2, nprocs2,  &
            C_CHAR_"ctx2"//C_NULL_CHAR, &
            (/ FSTARPU_SCHED_CTX_POLICY_NAME, c_loc(ctx2_policy), c_null_ptr /))

        ! set inheritor context 
        call fstarpu_sched_ctx_set_inheritor(ctx2, ctx1);

        call fstarpu_sched_ctx_display_workers(ctx1)
        call fstarpu_sched_ctx_display_workers(ctx2)

        do i = 1, n
                ! submit a task on context 1
                arg_id = 1*1000 + i
                arg_ctx = ctx1
                call fstarpu_insert_task((/ cl1, &
                        FSTARPU_VALUE, c_loc(arg_id), FSTARPU_SZ_C_INT, &
                        FSTARPU_SCHED_CTX, c_loc(arg_ctx), &
                    C_NULL_PTR /))
        end do

        do i = 1, n
                ! now submit a task on context 2
                arg_id = 2*1000 + i
                arg_ctx = ctx2
                call fstarpu_insert_task((/ cl2, &
                        FSTARPU_VALUE, c_loc(arg_id), FSTARPU_SZ_C_INT, &
                        FSTARPU_SCHED_CTX, c_loc(arg_ctx), &
                    C_NULL_PTR /))
        end do

        ! mark submission process as completed on context 2
        call fstarpu_sched_ctx_finished_submit(ctx2)

        do i = 1, n
                ! now submit a task on context 1 again
                arg_id = 1*10000 + i
                arg_ctx = ctx1
                call fstarpu_insert_task((/ cl1, &
                        FSTARPU_VALUE, c_loc(arg_id), FSTARPU_SZ_C_INT, &
                        FSTARPU_SCHED_CTX, c_loc(arg_ctx), &
                    C_NULL_PTR /))
        end do

        ! mark submission process as completed on context 1
        call fstarpu_sched_ctx_finished_submit(ctx1)

        ! wait for completion of all tasks
        call fstarpu_task_wait_for_all()

        ! show how to add some workers from a context to another
        call fstarpu_sched_ctx_add_workers(procs1, nprocs1, ctx2)

        ! deallocate both contexts
        call fstarpu_sched_ctx_delete(ctx2)
        call fstarpu_sched_ctx_delete(ctx1)

        deallocate(procs2)
        deallocate(procs1)

        ! free codelet structure
        call fstarpu_codelet_free(cl1)
        call fstarpu_codelet_free(cl2)

        ! shut StarPU down
        call fstarpu_shutdown()

end program nf_sched_ctx

