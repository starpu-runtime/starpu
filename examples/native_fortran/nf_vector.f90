! StarPU --- Runtime system for heterogeneous multicore architectures.
!
! Copyright (C) 2016  Inria
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

program nf_vector
        use iso_c_binding       ! C interfacing module
        use fstarpu_mod         ! StarPU interfacing module
        use nf_codelets
        implicit none

        real(8), dimension(:), allocatable, target :: va
        integer, dimension(:), allocatable, target :: vb
        integer :: i

        type(c_ptr) :: cl_vec   ! a pointer for the codelet structure
        type(c_ptr) :: dh_va    ! a pointer for the 'va' vector data handle
        type(c_ptr) :: dh_vb    ! a pointer for the 'vb' vector data handle
        integer(c_int) :: err   ! return status for fstarpu_init

        allocate(va(5))
        va = (/ (i,i=1,5) /)

        allocate(vb(7))
        vb = (/ (i,i=1,7) /)

        ! initialize StarPU with default settings
        err = fstarpu_init(C_NULL_PTR)
        if (err == -19) then
                stop 77
        end if

        ! allocate an empty codelet structure
        cl_vec = fstarpu_codelet_allocate()

        ! add a CPU implementation function to the codelet
        call fstarpu_codelet_add_cpu_func(cl_vec, C_FUNLOC(cl_cpu_func_vec))

        ! add a Read-only mode data buffer to the codelet
        call fstarpu_codelet_add_buffer(cl_vec, FSTARPU_R)

        ! add a Read-Write mode data buffer to the codelet
        call fstarpu_codelet_add_buffer(cl_vec, FSTARPU_RW)

        ! register 'va', a vector of real(8) elements
        dh_va = fstarpu_vector_data_register(c_loc(va), 1+ubound(va,1)-lbound(va,1), c_sizeof(va(lbound(va,1))), 0)

        ! register 'vb', a vector of integer elements
        dh_vb = fstarpu_vector_data_register(c_loc(vb), 1+ubound(vb,1)-lbound(vb,1), c_sizeof(vb(lbound(vb,1))), 0)

        ! insert a task with codelet cl_vec, and vectors 'va' and 'vb'
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
        call fstarpu_insert_task((/ cl_vec, FSTARPU_DATA, dh_va, FSTARPU_DATA, dh_vb, C_NULL_PTR /))

        ! wait for task completion
        call fstarpu_task_wait_for_all()

        ! unregister 'va'
        call fstarpu_data_unregister(dh_va)

        ! unregister 'vb'
        call fstarpu_data_unregister(dh_vb)

        ! free codelet structure
        call fstarpu_codelet_free(cl_vec)

        ! shut StarPU down
        call fstarpu_shutdown()

        deallocate(vb)
        deallocate(va)

end program nf_vector

