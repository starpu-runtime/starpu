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
program nf_partition
        use iso_c_binding       ! C interfacing module
        use fstarpu_mod         ! StarPU interfacing module
        use nf_partition_cl
        implicit none

        real(8), dimension(:,:), allocatable, target :: ma
        integer, target :: i,j

        type(c_ptr) :: cl_partition   ! a pointer for the codelet structure
        type(c_ptr) :: dh_ma    ! a pointer for the 'ma' vector data handle
        type(c_ptr) :: dh_sub   ! a pointer for the sub-data handle
        integer(c_int) :: err   ! return status for fstarpu_init
        integer(c_int) :: ncpu  ! number of cpus workers
        type(c_ptr) :: filter_M
        type(c_ptr) :: filter_N

        integer(c_int), parameter :: ma_M = 64
        integer(c_int), parameter :: ma_M_parts = 32
        integer(c_int), parameter :: ma_N = 16
        integer(c_int), parameter :: ma_N_parts = 4

        allocate(ma(ma_M,ma_N))
        do i=1,ma_M
        do j=1,ma_N
        ma(i,j) = (i*100)+j
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
        cl_partition = fstarpu_codelet_allocate()

        ! set the codelet name
        call fstarpu_codelet_set_name(cl_partition, C_CHAR_"my_part_codelet"//C_NULL_CHAR)

        ! add a CPU implementation function to the codelet
        call fstarpu_codelet_add_cpu_func(cl_partition, C_FUNLOC(cl_partition_func))

        ! add a Read-Write mode data buffer to the codelet
        call fstarpu_codelet_add_buffer(cl_partition, FSTARPU_RW)

        ! register 'ma', a vector of real(8) elements
        !dh_ma = fstarpu_matrix_data_register(c_loc(ma), ma_M, ma_M, ma_N, c_sizeof(ma(1,1)), 0)
        call fstarpu_matrix_data_register(dh_ma, 0, c_loc(ma), ma_M, ma_M, ma_N, c_sizeof(ma(1,1)))

        ! allocate partitioning filters
        filter_M = fstarpu_df_alloc_matrix_filter_block()
        call fstarpu_data_filter_set_nchildren(filter_M, ma_M_parts)

        filter_N = fstarpu_df_alloc_matrix_filter_vertical_block()
        call fstarpu_data_filter_set_nchildren(filter_N, ma_N_parts)

        ! apply partitioning
        call fstarpu_data_map_filters(dh_ma, 2, (/ filter_M, filter_N /))

        do i=0,ma_M_parts-1
        do j=0,ma_N_parts-1
                ! get partitioned tile
                dh_sub = fstarpu_data_get_sub_data (dh_ma, 2, (/i, j/))

                ! Note: The array argument must follow the layout:
                !   (/
                !     <codelet_ptr>,
                !     [<argument_type> [<argument_value(s)],]
                !     . . .
                !     C_NULL_PTR
                !   )/
                call fstarpu_insert_task((/ cl_partition, FSTARPU_RW, dh_sub, C_NULL_PTR /))
        end do
        end do

        ! wait for task completion
        call fstarpu_task_wait_for_all()

        ! unpartition 'ma'
        call fstarpu_data_unpartition(dh_ma, 0)

        ! unregister 'ma'
        call fstarpu_data_unregister(dh_ma)

        ! free data filter structures
        call fstarpu_data_filter_free(filter_N)
        call fstarpu_data_filter_free(filter_M)

        ! free codelet structure
        call fstarpu_codelet_free(cl_partition)

        ! shut StarPU down
        call fstarpu_shutdown()

        deallocate(ma)

end program nf_partition
