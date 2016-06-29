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

module fstarpu_mod
        use iso_c_binding

        integer(c_int), bind(C) :: FSTARPU_R
        integer(c_int), bind(C) :: FSTARPU_W
        integer(c_int), bind(C) :: FSTARPU_RW
        integer(c_int), bind(C) :: FSTARPU_SCRATCH
        integer(c_int), bind(C) :: FSTARPU_REDUX

        type(c_ptr), bind(C) :: FSTARPU_DATA
        type(c_ptr), bind(C) :: FSTARPU_VALUE

        type(c_ptr), bind(C) :: FSTARPU_SZ_INT4
        type(c_ptr), bind(C) :: FSTARPU_SZ_INT8

        type(c_ptr), bind(C) :: FSTARPU_SZ_REAL4
        type(c_ptr), bind(C) :: FSTARPU_SZ_REAL8
        interface
                ! == starpu.h ==

                subroutine fstarpu_conf_init(conf) bind(C,name="starpu_conf_init")
                        use iso_c_binding, only: c_ptr
                        type(c_ptr), value, intent(in) :: conf
                end subroutine fstarpu_conf_init

                ! starpu_init: see fstarpu_init
                ! starpu_initialize: see fstarpu_init

                subroutine fstarpu_pause() bind(C,name="starpu_pause")
                end subroutine fstarpu_pause

                subroutine fstarpu_resume() bind(C,name="starpu_resume")
                end subroutine fstarpu_resume

                subroutine fstarpu_shutdown () bind(C,name="starpu_shutdown")
                end subroutine fstarpu_shutdown

                ! starpu_topology_print

                subroutine fstarpu_asynchronous_copy_disabled() bind(C,name="starpu_asynchronous_copy_disabled")
                end subroutine fstarpu_asynchronous_copy_disabled

                subroutine fstarpu_asynchronous_cuda_copy_disabled() bind(C,name="starpu_asynchronous_cuda_copy_disabled")
                end subroutine fstarpu_asynchronous_cuda_copy_disabled

                subroutine fstarpu_asynchronous_opencl_copy_disabled() bind(C,name="starpu_asynchronous_opencl_copy_disabled")
                end subroutine fstarpu_asynchronous_opencl_copy_disabled

                subroutine fstarpu_asynchronous_mic_copy_disabled() bind(C,name="starpu_asynchronous_mic_copy_disabled")
                end subroutine fstarpu_asynchronous_mic_copy_disabled

                subroutine fstarpu_display_stats() bind(C,name="starpu_display_stats")
                end subroutine fstarpu_display_stats

                subroutine fstarpu_get_version(major,minor,release) bind(C,name="starpu_get_version")
                        use iso_c_binding, only: c_int
                        integer(c_int), intent(out) :: major,minor,release
                end subroutine fstarpu_get_version

                function fstarpu_cpu_worker_get_count() bind(C,name="starpu_cpu_worker_get_count")
                        use iso_c_binding, only: c_int
                        integer(c_int)              :: fstarpu_cpu_worker_get_count
                end function fstarpu_cpu_worker_get_count

                ! == starpu_task.h ==

                ! starpu_tag_declare_deps
                ! starpu_tag_declare_deps_array
                ! starpu_task_declare_deps_array
                ! starpu_tag_wait
                ! starpu_tag_wait_array
                ! starpu_tag_notify_from_apps
                ! starpu_tag_restart
                ! starpu_tag_remove
                ! starpu_task_init
                ! starpu_task_clean
                ! starpu_task_create
                ! starpu_task_destroy
                ! starpu_task_submit
                ! starpu_task_submit_to_ctx
                ! starpu_task_finished
                ! starpu_task_wait

                subroutine fstarpu_task_wait_for_all () bind(C,name="starpu_task_wait_for_all")
                end subroutine fstarpu_task_wait_for_all

                ! starpu_task_wait_for_n_submitted
                ! starpu_task_wait_for_all_in_ctx
                ! starpu_task_wait_for_n_submitted_in_ctx
                ! starpu_task_wait_for_no_ready
                ! starpu_task_nready
                ! starpu_task_nsubmitted
                ! starpu_codelet_init
                ! starpu_codelet_display_stats
                ! starpu_task_get_current
                ! starpu_parallel_task_barrier_init
                ! starpu_parallel_task_barrier_init_n
                ! starpu_task_dup
                ! starpu_task_set_implementation
                ! starpu_task_get_implementation
                ! --

                function fstarpu_codelet_allocate () bind(C)
                        use iso_c_binding, only: c_ptr
                        type(c_ptr) :: fstarpu_codelet_allocate
                end function fstarpu_codelet_allocate

                subroutine fstarpu_codelet_free (cl) bind(C)
                        use iso_c_binding, only: c_ptr
                        type(c_ptr), value, intent(in) :: cl
                end subroutine fstarpu_codelet_free

                subroutine fstarpu_codelet_add_cpu_func (cl, f_ptr) bind(C)
                        use iso_c_binding, only: c_ptr, c_funptr
                        type(c_ptr), value, intent(in) :: cl
                        type(c_funptr), value, intent(in) :: f_ptr
                end subroutine fstarpu_codelet_add_cpu_func

                subroutine fstarpu_codelet_add_cuda_func (cl, f_ptr) bind(C)
                        use iso_c_binding, only: c_ptr, c_funptr
                        type(c_ptr), value, intent(in) :: cl
                        type(c_funptr), value, intent(in) :: f_ptr
                end subroutine fstarpu_codelet_add_cuda_func

                subroutine fstarpu_codelet_add_opencl_func (cl, f_ptr) bind(C)
                        use iso_c_binding, only: c_ptr, c_funptr
                        type(c_ptr), value, intent(in) :: cl
                        type(c_funptr), value, intent(in) :: f_ptr
                end subroutine fstarpu_codelet_add_opencl_func

                subroutine fstarpu_codelet_add_buffer (cl, mode) bind(C)
                        use iso_c_binding, only: c_ptr, c_int
                        type(c_ptr), value, intent(in) :: cl
                        integer(c_int), value, intent(in) :: mode
                end subroutine fstarpu_codelet_add_buffer

                function fstarpu_vector_data_register(vector, nx, elt_size, ram) bind(C)
                        use iso_c_binding, only: c_ptr, c_int, c_size_t
                        type(c_ptr) :: fstarpu_vector_data_register
                        type(c_ptr), value, intent(in) :: vector
                        integer(c_int), value, intent(in) :: nx
                        integer(c_size_t), value, intent(in) :: elt_size
                        integer(c_int), value, intent(in) :: ram
                end function fstarpu_vector_data_register

                function fstarpu_vector_get_ptr(buffers, i) bind(C)
                        use iso_c_binding, only: c_ptr, c_int
                        type(c_ptr) :: fstarpu_vector_get_ptr
                        type(c_ptr), value, intent(in) :: buffers
                        integer(c_int), value, intent(in) :: i
                end function fstarpu_vector_get_ptr

                function fstarpu_vector_get_nx(buffers, i) bind(C)
                        use iso_c_binding, only: c_ptr, c_int
                        integer(c_int) :: fstarpu_vector_get_nx
                        type(c_ptr), value, intent(in) :: buffers
                        integer(c_int), value, intent(in) :: i
                end function fstarpu_vector_get_nx

                function fstarpu_matrix_data_register(matrix, ldy, ny, nx, elt_size, ram) bind(C)
                        use iso_c_binding, only: c_ptr, c_int, c_size_t
                        type(c_ptr) :: fstarpu_matrix_data_register
                        type(c_ptr), value, intent(in) :: matrix
                        integer(c_int), value, intent(in) :: ldy
                        integer(c_int), value, intent(in) :: ny
                        integer(c_int), value, intent(in) :: nx
                        integer(c_size_t), value, intent(in) :: elt_size
                        integer(c_int), value, intent(in) :: ram
                end function fstarpu_matrix_data_register

                function fstarpu_matrix_get_ptr(buffers, i) bind(C)
                        use iso_c_binding, only: c_ptr, c_int
                        type(c_ptr) :: fstarpu_matrix_get_ptr
                        type(c_ptr), value, intent(in) :: buffers
                        integer(c_int), value, intent(in) :: i
                end function fstarpu_matrix_get_ptr

                function fstarpu_matrix_get_ld(buffers, i) bind(C)
                        use iso_c_binding, only: c_ptr, c_int
                        integer(c_int) :: fstarpu_matrix_get_ld
                        type(c_ptr), value, intent(in) :: buffers
                        integer(c_int), value, intent(in) :: i
                end function fstarpu_matrix_get_ld

                function fstarpu_matrix_get_ny(buffers, i) bind(C)
                        use iso_c_binding, only: c_ptr, c_int
                        integer(c_int) :: fstarpu_matrix_get_ny
                        type(c_ptr), value, intent(in) :: buffers
                        integer(c_int), value, intent(in) :: i
                end function fstarpu_matrix_get_ny

                function fstarpu_matrix_get_nx(buffers, i) bind(C)
                        use iso_c_binding, only: c_ptr, c_int
                        integer(c_int) :: fstarpu_matrix_get_nx
                        type(c_ptr), value, intent(in) :: buffers
                        integer(c_int), value, intent(in) :: i
                end function fstarpu_matrix_get_nx

                subroutine fstarpu_data_unregister (dh) bind(C,name="starpu_data_unregister")
                        use iso_c_binding, only: c_ptr
                        type(c_ptr), value, intent(in) :: dh
                end subroutine fstarpu_data_unregister

                subroutine fstarpu_insert_task(arglist) bind(C)
                        use iso_c_binding, only: c_ptr
                        type(c_ptr), dimension(:), intent(in) :: arglist
                end subroutine fstarpu_insert_task

                subroutine fstarpu_unpack_arg(cl_arg,bufferlist) bind(C)
                        use iso_c_binding, only: c_ptr
                        type(c_ptr), value, intent(in) :: cl_arg
                        type(c_ptr), dimension(:), intent(in) :: bufferlist
                end subroutine fstarpu_unpack_arg

        end interface

        contains

                function fstarpu_init (conf) bind(C)
                        use iso_c_binding
                        integer(c_int) :: fstarpu_init
                        type(c_ptr), value, intent(in) :: conf

                        integer(4) :: FSTARPU_SZ_INT4_dummy
                        integer(8) :: FSTARPU_SZ_INT8_dummy
                        real(4) :: FSTARPU_SZ_REAL4_dummy
                        real(8) :: FSTARPU_SZ_REAL8_dummy

                        ! Note: Referencing global C constants from Fortran has
                        ! been found unreliable on some architectures, notably
                        ! on Darwin. The get_integer/get_pointer_constant
                        ! scheme is a workaround to that issue.

                        interface
                                ! These functions are not exported to the end user
                                function fstarpu_get_integer_constant(s) bind(C)
                                        use iso_c_binding, only: c_int,c_char
                                        integer(c_int) :: fstarpu_get_integer_constant
                                        character(kind=c_char) :: s
                                end function fstarpu_get_integer_constant

                                function fstarpu_get_pointer_constant(s) bind(C)
                                        use iso_c_binding, only: c_ptr,c_char
                                        type(c_ptr) :: fstarpu_get_pointer_constant
                                        character(kind=c_char) :: s
                                end function fstarpu_get_pointer_constant

                                function fstarpu_init_internal (conf) bind(C,name="starpu_init")
                                        use iso_c_binding, only: c_ptr,c_int
                                        integer(c_int) :: fstarpu_init_internal
                                        type(c_ptr), value :: conf
                                end function fstarpu_init_internal
                        end interface

                        ! Initialize Fortran integer constants from C peers
                        FSTARPU_R = fstarpu_get_integer_constant(C_CHAR_"FSTARPU_R"//C_NULL_CHAR)
                        FSTARPU_W = fstarpu_get_integer_constant(C_CHAR_"FSTARPU_W"//C_NULL_CHAR)
                        FSTARPU_RW = fstarpu_get_integer_constant(C_CHAR_"FSTARPU_RW"//C_NULL_CHAR)
                        FSTARPU_SCRATCH = fstarpu_get_integer_constant(C_CHAR_"FSTARPU_SCRATCH"//C_NULL_CHAR)
                        FSTARPU_REDUX = fstarpu_get_integer_constant(C_CHAR_"FSTARPU_REDUX"//C_NULL_CHAR)
                        ! Initialize Fortran 'pointer' constants from C peers
                        FSTARPU_DATA = fstarpu_get_pointer_constant(C_CHAR_"FSTARPU_DATA"//C_NULL_CHAR)
                        FSTARPU_VALUE = fstarpu_get_pointer_constant(C_CHAR_"FSTARPU_VALUE"//C_NULL_CHAR)
                        ! Initialize size constants as 'c_ptr'
                        FSTARPU_SZ_INT4 = transfer(int(c_sizeof(FSTARPU_SZ_INT4_dummy),kind=c_intptr_t),C_NULL_PTR)
                        FSTARPU_SZ_INT8 = transfer(int(c_sizeof(FSTARPU_SZ_INT8_dummy),kind=c_intptr_t),C_NULL_PTR)
                        FSTARPU_SZ_REAL4 = transfer(int(c_sizeof(FSTARPU_SZ_REAL4_dummy),kind=c_intptr_t),C_NULL_PTR)
                        FSTARPU_SZ_REAL8 = transfer(int(c_sizeof(FSTARPU_SZ_REAL8_dummy),kind=c_intptr_t),C_NULL_PTR)
                        ! Initialize StarPU
                        if (c_associated(conf)) then 
                                fstarpu_init = fstarpu_init_internal(conf)
                        else
                                fstarpu_init = fstarpu_init_internal(C_NULL_PTR)
                        end if
                end function fstarpu_init

                function fstarpu_csizet_to_cptr(i) bind(C)
                        use iso_c_binding
                        type(c_ptr) :: fstarpu_csizet_to_cptr
                        integer(c_size_t) :: i
                        fstarpu_csizet_to_cptr = transfer(int(i,kind=c_intptr_t),C_NULL_PTR)
                end function fstarpu_csizet_to_cptr

                function fstarpu_int_to_cptr(i) bind(C)
                        use iso_c_binding
                        type(c_ptr) :: fstarpu_int_to_cptr
                        integer :: i
                        fstarpu_int_to_cptr = transfer(int(i,kind=c_intptr_t),C_NULL_PTR)
                end function fstarpu_int_to_cptr
end module fstarpu_mod
