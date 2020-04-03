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
program nf_mm
        use iso_c_binding       ! C interfacing module
        use fstarpu_mod         ! StarPU interfacing module
        use fstarpu_mpi_mod     ! StarPU-MPI interfacing module
        use nf_mm_cl
        implicit none

        logical, parameter :: verbose = .false.
        integer(c_int) :: comm_size, comm_rank
        integer(c_int), target :: comm_world
        integer(c_int) :: N = 16, BS = 4, NB
        real(kind=c_double),allocatable,target :: A(:,:), B(:,:), C(:,:)
        type(c_ptr),allocatable :: dh_A(:), dh_B(:), dh_C(:,:)
        type(c_ptr) :: cl_mm
        type(c_ptr) :: task
        integer(c_int) :: ncpu
        integer(c_int) :: ret
        integer(c_int) :: row, col
        integer(c_int) :: b_row, b_col
        integer(c_int) :: mr, rank
        integer(c_int64_t) :: tag

        ret = fstarpu_init(C_NULL_PTR)
        if (ret == -19) then
                stop 77
        else if (ret /= 0) then
                stop 1
        end if

        ret = fstarpu_mpi_init(1)
        print *,"fstarpu_mpi_init status:", ret
        if (ret /= 0) then
                stop 1
        end if

        ! stop there if no CPU worker available
        ncpu = fstarpu_cpu_worker_get_count()
        if (ncpu == 0) then
                call fstarpu_shutdown()
                stop 77
        end if

        comm_world = fstarpu_mpi_world_comm()
        comm_size = fstarpu_mpi_world_size()
        comm_rank = fstarpu_mpi_world_rank()

        if (comm_size < 2) then
                call fstarpu_shutdown()
                ret = fstarpu_mpi_shutdown()
                stop 77
        end if

        ! TODO: process app's argc/argv
        NB = N/BS

        ! allocate and initialize codelet
        cl_mm = fstarpu_codelet_allocate()
        call fstarpu_codelet_set_name(cl_mm, c_char_"nf_mm_cl"//c_null_char)
        call fstarpu_codelet_add_cpu_func(cl_mm, C_FUNLOC(cl_cpu_mult))
        call fstarpu_codelet_add_buffer(cl_mm, FSTARPU_R)
        call fstarpu_codelet_add_buffer(cl_mm, FSTARPU_R)
        call fstarpu_codelet_add_buffer(cl_mm, FSTARPU_RW)

        ! allocate matrices
        if (comm_rank == 0) then
                allocate(A(N,N))
                allocate(B(N,N))
                allocate(C(N,N))
        end if

        ! init matrices
        if (comm_rank == 0) then
                do col=1,N
                do row=1,N
                if (row == col) then
                        A(row,col) = 2
                else
                        A(row,col) = 0
                end if
                B(row,col) = row*N+col
                C(row,col) = 0
                end do
                end do

                if (verbose) then
                        print *,"A"
                        call mat_disp(A)
                        print *,"B"
                        call mat_disp(B)
                        print *,"C"
                        call mat_disp(C)
                end if
        end if

        ! allocate data handles
        allocate(dh_A(NB))
        allocate(dh_B(NB))
        allocate(dh_C(NB,NB))

        ! register matrices
        if (comm_rank == 0) then
                mr = 0 ! TODO: use STARPU_MAIN_RAM constant
        else
                mr = -1
        end if
        tag = 0

        do b_row=1,NB
                if (comm_rank == 0) then
                        call fstarpu_matrix_data_register(dh_A(b_row), mr, &
                                c_loc( A(1+(b_row-1)*BS,1) ), N, BS, N, c_sizeof(A(1,1)))
                else
                        call fstarpu_matrix_data_register(dh_A(b_row), mr, &
                                c_null_ptr, N, BS, N, c_sizeof(A(1,1)))
                end if
                call fstarpu_mpi_data_register(dh_A(b_row), tag, 0)
                tag = tag+1
        end do

        do b_col=1,NB
                if (comm_rank == 0) then
                        call fstarpu_matrix_data_register(dh_B(b_col), mr, &
                                c_loc( B(1,1+(b_col-1)*BS) ), N, N, BS, c_sizeof(B(1,1)))
                else
                        call fstarpu_matrix_data_register(dh_B(b_col), mr, &
                                c_null_ptr, N, N, BS, c_sizeof(B(1,1)))
                end if
                call fstarpu_mpi_data_register(dh_B(b_col), tag, 0)
                tag = tag+1
        end do

        do b_col=1,NB
        do b_row=1,NB
                if (comm_rank == 0) then
                        call fstarpu_matrix_data_register(dh_C(b_row,b_col), mr, &
                                c_loc( C(1+(b_row-1)*BS,1+(b_col-1)*BS) ), N, BS, BS, c_sizeof(C(1,1)))
                else
                        call fstarpu_matrix_data_register(dh_C(b_row,b_col), mr, &
                                c_null_ptr, N, BS, BS, c_sizeof(C(1,1)))
                end if
                call fstarpu_mpi_data_register(dh_C(b_row,b_col), tag, 0)
                tag = tag+1
        end do
        end do

        ! distribute matrix C
        do b_col=1,NB
        do b_row=1,NB
        rank = modulo(b_row+b_col, comm_size)
        call fstarpu_mpi_data_migrate(comm_world, dh_c(b_row,b_col), rank)
        end do
        end do

        do b_col=1,NB
           do b_row=1,NB
              task = fstarpu_mpi_task_build((/ c_loc(comm_world), cl_mm, &
                   				FSTARPU_R,  dh_A(b_row), &
                                                FSTARPU_R,  dh_B(b_col), &
                                                FSTARPU_RW, dh_C(b_row,b_col), &
                                                C_NULL_PTR /))
              if (c_associated(task)) then
                 ret = fstarpu_task_submit(task)
              endif
              call fstarpu_mpi_task_post_build((/ c_loc(comm_world), cl_mm, &
                   				FSTARPU_R,  dh_A(b_row), &
                                                FSTARPU_R,  dh_B(b_col), &
                                                FSTARPU_RW, dh_C(b_row,b_col), &
                                                C_NULL_PTR /))
           end do
        end do

        call fstarpu_task_wait_for_all()

        ! undistribute matrix C
        do b_col=1,NB
        do b_row=1,NB
        call fstarpu_mpi_data_migrate(comm_world, dh_c(b_row,b_col), 0)
        end do
        end do

        ! unregister matrices
        do b_row=1,NB
                call fstarpu_data_unregister(dh_A(b_row))
        end do

        do b_col=1,NB
                call fstarpu_data_unregister(dh_B(b_col))
        end do

        do b_col=1,NB
        do b_row=1,NB
                call fstarpu_data_unregister(dh_C(b_row,b_col))
        end do
        end do

        ! check result
        if (comm_rank == 0) then
                if (verbose) then
                        print *,"final C"
                        call mat_disp(C)
                end if

                do col=1,N
                do row=1,N
                if (abs(C(row,col) - 2*(row*N+col)) > 1.0) then
                        print *, "check failed"
                        stop 1
                end if
                end do
                end do
        end if

        ! free handles
        deallocate(dh_A)
        deallocate(dh_B)
        deallocate(dh_C)

        ! free matrices
        if (comm_rank == 0) then
                deallocate(A)
                deallocate(B)
                deallocate(C)
        end if
        call fstarpu_codelet_free(cl_mm)
        call fstarpu_shutdown()

        ret = fstarpu_mpi_shutdown()
        print *,"fstarpu_mpi_shutdown status:", ret
        if (ret /= 0) then
                stop 1
        end if
end program nf_mm
