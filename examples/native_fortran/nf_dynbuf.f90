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
program nf_dynbuf
        use iso_c_binding       ! C interfacing module
        use fstarpu_mod         ! StarPU interfacing module
        use nf_dynbuf_cl
        implicit none

        type(c_ptr) :: cl_dynbuf_big   ! a pointer for the codelet structure
        type(c_ptr) :: dh_var
        type(c_ptr) :: descrs_var
        integer(c_int),target :: nbuffers
        integer(c_int) :: err   ! return status for fstarpu_init
        integer(c_int) :: ncpu  ! number of cpus workers

        integer(c_int),target :: var
        integer(c_int) :: i

        var = 42

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
        cl_dynbuf_big = fstarpu_codelet_allocate()
        call fstarpu_codelet_set_name(cl_dynbuf_big, C_CHAR_"dummy_big_kernel"//C_NULL_CHAR)
        call fstarpu_codelet_add_cpu_func(cl_dynbuf_big, C_FUNLOC(cl_cpu_func_dynbuf_big))

        write(*,*) "FSTARPU_NMAXBUFS",FSTARPU_NMAXBUFS
        nbuffers = FSTARPU_NMAXBUFS+1
        call fstarpu_codelet_set_nbuffers(cl_dynbuf_big, nbuffers)

        call fstarpu_variable_data_register(dh_var, 0, c_loc(var), c_sizeof(var))

        descrs_var = fstarpu_data_descr_array_alloc(nbuffers)
        do i=0,nbuffers-1
                call fstarpu_data_descr_array_set(descrs_var, i, dh_var, FSTARPU_RW)
        end do
        call fstarpu_insert_task((/ cl_dynbuf_big, &
                FSTARPU_VALUE, c_loc(nbuffers), FSTARPU_SZ_C_INT, &
                FSTARPU_DATA_MODE_ARRAY, descrs_var, c_loc(nbuffers), &
                C_NULL_PTR /))
        call fstarpu_task_wait_for_all()

        call fstarpu_data_descr_array_free(descrs_var)
        call fstarpu_data_unregister(dh_var)

        ! free codelet structure
        call fstarpu_codelet_free(cl_dynbuf_big)

        ! shut StarPU down
        call fstarpu_shutdown()

end program nf_dynbuf
