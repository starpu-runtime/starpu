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
program nf_matrix
        use iso_c_binding       ! C interfacing module
        use fstarpu_mod         ! StarPU interfacing module
        use nf_codelets
        implicit none

        real(8), dimension(:,:), allocatable, target :: ma
        integer, dimension(:,:), allocatable, target :: mb
        integer :: i,j

        type(c_ptr) :: cl_mat   ! a pointer for the codelet structure
        type(c_ptr) :: dh_ma    ! a pointer for the 'ma' vector data handle
        type(c_ptr) :: dh_mb    ! a pointer for the 'mb' vector data handle
        integer(c_int) :: err   ! return status for fstarpu_init
        integer(c_int) :: ncpu  ! number of cpus workers

        allocate(ma(5,6))
        do i=1,5
        do j=1,6
        ma(i,j) = (i*10)+j
        end do
        end do

        allocate(mb(7,8))
        do i=1,7
        do j=1,8
        mb(i,j) = (i*10)+j
        end do
        end do

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

        ! allocate an empty codelet structure
        cl_mat = fstarpu_codelet_allocate()

        ! set the codelet name
        call fstarpu_codelet_set_name(cl_mat, C_CHAR_"my_mat_codelet"//C_NULL_CHAR)

        ! add a CPU implementation function to the codelet
        call fstarpu_codelet_add_cpu_func(cl_mat, C_FUNLOC(cl_cpu_func_mat))

        ! add a Read-only mode data buffer to the codelet
        call fstarpu_codelet_add_buffer(cl_mat, FSTARPU_R)

        ! add a Read-Write mode data buffer to the codelet
        call fstarpu_codelet_add_buffer(cl_mat, FSTARPU_RW)

        ! register 'ma', a vector of real(8) elements
        !dh_ma = fstarpu_matrix_data_register(c_loc(ma), 5, 5, 6, c_sizeof(ma(1,1)), 0)
        call fstarpu_matrix_data_register(dh_ma, 0, c_loc(ma), 5, 5, 6, c_sizeof(ma(1,1)))

        ! register 'mb', a vector of integer elements
        call fstarpu_matrix_data_register(dh_mb, 0, c_loc(mb), 7, 7, 8, c_sizeof(mb(1,1)))

        ! insert a task with codelet cl_mat, and vectors 'ma' and 'mb'
        !
        ! Note: The array argument must follow the layout:
        !   (/
        !     <codelet_ptr>,
        !     [<argument_type> [<argument_value(s)],]
        !     . . .
        !     C_NULL_PTR
        !   )/
        call fstarpu_insert_task((/ cl_mat, FSTARPU_R, dh_ma, FSTARPU_RW, dh_mb, C_NULL_PTR /))

        ! wait for task completion
        call fstarpu_task_wait_for_all()

        ! unregister 'ma'
        call fstarpu_data_unregister(dh_ma)

        ! unregister 'mb'
        call fstarpu_data_unregister(dh_mb)

        ! free codelet structure
        call fstarpu_codelet_free(cl_mat)

        ! shut StarPU down
        call fstarpu_shutdown()

        deallocate(mb)
        deallocate(ma)

end program nf_matrix
