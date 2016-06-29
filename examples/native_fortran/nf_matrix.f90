program nf_matrix
        use iso_c_binding       ! C interfacing module
        use fstarpu_mod         ! StarPU interfacing module
        use codelets
        implicit none

        real(8), dimension(:,:), allocatable, target :: ma
        integer, dimension(:,:), allocatable, target :: mb
        integer :: i,j

        type(c_ptr) :: cl_mat   ! a pointer for the codelet structure
        type(c_ptr) :: dh_ma    ! a pointer for the 'ma' vector data handle
        type(c_ptr) :: dh_mb    ! a pointer for the 'mb' vector data handle
        integer(c_int) :: err   ! return status for fstarpu_init

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

        ! allocate an empty codelet structure
        cl_mat = fstarpu_codelet_allocate()

        ! add a CPU implementation function to the codelet
        call fstarpu_codelet_add_cpu_func(cl_mat, C_FUNLOC(cl_cpu_func_mat))

        ! add a Read-only mode data buffer to the codelet
        call fstarpu_codelet_add_buffer(cl_mat, FSTARPU_R)

        ! add a Read-Write mode data buffer to the codelet
        call fstarpu_codelet_add_buffer(cl_mat, FSTARPU_RW)

        ! register 'ma', a vector of real(8) elements
        dh_ma = fstarpu_matrix_data_register(c_loc(ma), 5, 5, 6, c_sizeof(ma(1,1)), 0)

        ! register 'mb', a vector of integer elements
        dh_mb = fstarpu_matrix_data_register(c_loc(mb), 7, 7, 8, c_sizeof(mb(1,1)), 0)

        ! insert a task with codelet cl_mat, and vectors 'ma' and 'mb'
        !
        ! Note: The array argument must follow the layout:
        !   (/
        !     <codelet_ptr>,
        !     [<argument_type> [<argument_value(s)],]
        !     . . .
        !     C_NULL_PTR
        !   )/
        !
        ! Note: The argument type for data handles is FSTARPU_DATA, regardless
        ! of the buffer access mode (specified in the codelet)
        call fstarpu_insert_task((/ cl_mat, FSTARPU_DATA, dh_ma, FSTARPU_DATA, dh_mb, C_NULL_PTR /))

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
