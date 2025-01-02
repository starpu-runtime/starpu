! StarPU --- Runtime system for heterogeneous multicore architectures.
!
! Copyright (C) 2016-2025   University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
program nf_vector
        use iso_c_binding       ! C interfacing module
        use fstarpu_mod         ! StarPU interfacing module
        use nf_codelets
        implicit none

        real(8), dimension(:), allocatable, target :: va
        integer, dimension(:), allocatable, target :: vb
        integer :: i

        type(c_ptr) :: perfmodel_vec   ! a pointer for the perfmodel structure
        type(c_ptr) :: cl_vec   ! a pointer for the codelet structure
        type(c_ptr) :: dh_va    ! a pointer for the 'va' vector data handle
        type(c_ptr) :: dh_vb    ! a pointer for the 'vb' vector data handle
        integer(c_int) :: err   ! return status for fstarpu_init
        integer(c_int) :: ncpu  ! number of cpus workers
        integer(c_int) :: bool_ret

        allocate(va(5))
        va = (/ (i,i=1,5) /)

        allocate(vb(7))
        vb = (/ (i,i=1,7) /)

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

        ! illustrate use of pause/resume/is_paused
        bool_ret = fstarpu_is_paused()
        if (bool_ret /= 0) then
                stop 1
        end if

        call fstarpu_pause

        bool_ret = fstarpu_is_paused()
        if (bool_ret == 0) then
                stop 1
        end if

        call fstarpu_resume

        bool_ret = fstarpu_is_paused()
        if (bool_ret /= 0) then
                stop 1
        end if

        ! allocate an empty perfmodel structure
        perfmodel_vec = fstarpu_perfmodel_allocate()

        ! set the perfmodel symbol
        call fstarpu_perfmodel_set_symbol(perfmodel_vec, C_CHAR_"my_vec_sym"//C_NULL_CHAR)

        ! set the perfmodel type
        call fstarpu_perfmodel_set_type(perfmodel_vec, FSTARPU_HISTORY_BASED)

        ! allocate an empty codelet structure
        cl_vec = fstarpu_codelet_allocate()

        ! set the codelet name
        call fstarpu_codelet_set_name(cl_vec, C_CHAR_"my_vec_codelet"//C_NULL_CHAR)

        ! set the codelet perfmodel
        call fstarpu_codelet_set_model(cl_vec, perfmodel_vec)

        ! add a CPU implementation function to the codelet
        call fstarpu_codelet_add_cpu_func(cl_vec, C_FUNLOC(cl_cpu_func_vec))

        ! optionally set 'where' field to CPU only
        call fstarpu_codelet_set_where(cl_vec, FSTARPU_CPU)

        ! set 'type' field to SEQ (for demonstration purpose)
        call fstarpu_codelet_set_type(cl_vec, FSTARPU_SEQ)

        ! set 'max_parallelism' field to 1 (for demonstration purpose)
        call fstarpu_codelet_set_max_parallelism(cl_vec, 1)

        ! add a Read-only mode data buffer to the codelet
        call fstarpu_codelet_add_buffer(cl_vec, FSTARPU_R)

        ! add a Read-Write mode data buffer to the codelet
        call fstarpu_codelet_add_buffer(cl_vec, FSTARPU_RW.ior.FSTARPU_LOCALITY)

        ! register 'va', a vector of real(8) elements
        call fstarpu_vector_data_register(dh_va, 0, c_loc(va), 1+ubound(va,1)-lbound(va,1), c_sizeof(va(lbound(va,1))))

        ! register 'vb', a vector of integer elements
        call fstarpu_vector_data_register(dh_vb, 0, c_loc(vb), 1+ubound(vb,1)-lbound(vb,1), c_sizeof(vb(lbound(vb,1))))

        ! insert a task with codelet cl_vec, and vectors 'va' and 'vb'
        !
        ! Note: The array argument must follow the layout:
        !   (/
        !     <codelet_ptr>,
        !     [<argument_type> [<argument_value(s)],]
        !     . . .
        !     C_NULL_PTR
        !   )/
        call fstarpu_task_insert((/ cl_vec, &
                FSTARPU_R, dh_va, &
                FSTARPU_RW.ior.FSTARPU_LOCALITY, dh_vb, &
                FSTARPU_EXECUTE_WHERE, FSTARPU_CPU, & ! for illustration, not required here
                C_NULL_PTR /))

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

        ! free perfmodel structure (must be called after fstarpu_shutdown)
        call fstarpu_perfmodel_free(perfmodel_vec)

        deallocate(vb)
        deallocate(va)

end program nf_vector

