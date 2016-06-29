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

        interface
                subroutine fstarpu_shutdown () bind(C,name="starpu_shutdown")
                end subroutine fstarpu_shutdown

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
                        use iso_c_binding
                        type(c_ptr) :: fstarpu_vector_data_register
                        type(c_ptr), value, intent(in) :: vector
                        integer(c_int), value, intent(in) :: nx
                        integer(c_size_t), value, intent(in) :: elt_size
                        integer(c_int), value, intent(in) :: ram
                end function fstarpu_vector_data_register

                function fstarpu_vector_get_ptr(buffers, i) bind(C)
                        use iso_c_binding
                        type(c_ptr) :: fstarpu_vector_get_ptr
                        type(c_ptr), value, intent(in) :: buffers
                        integer(c_int), value, intent(in) :: i
                end function fstarpu_vector_get_ptr

                function fstarpu_vector_get_nx(buffers, i) bind(C)
                        use iso_c_binding
                        integer(c_int) :: fstarpu_vector_get_nx
                        type(c_ptr), value, intent(in) :: buffers
                        integer(c_int), value, intent(in) :: i
                end function fstarpu_vector_get_nx

                function fstarpu_matrix_data_register(matrix, ldy, ny, nx, elt_size, ram) bind(C)
                        use iso_c_binding
                        type(c_ptr) :: fstarpu_matrix_data_register
                        type(c_ptr), value, intent(in) :: matrix
                        integer(c_int), value, intent(in) :: ldy
                        integer(c_int), value, intent(in) :: ny
                        integer(c_int), value, intent(in) :: nx
                        integer(c_size_t), value, intent(in) :: elt_size
                        integer(c_int), value, intent(in) :: ram
                end function fstarpu_matrix_data_register

                function fstarpu_matrix_get_ptr(buffers, i) bind(C)
                        use iso_c_binding
                        type(c_ptr) :: fstarpu_matrix_get_ptr
                        type(c_ptr), value, intent(in) :: buffers
                        integer(c_int), value, intent(in) :: i
                end function fstarpu_matrix_get_ptr

                function fstarpu_matrix_get_ld(buffers, i) bind(C)
                        use iso_c_binding
                        integer(c_int) :: fstarpu_matrix_get_ld
                        type(c_ptr), value, intent(in) :: buffers
                        integer(c_int), value, intent(in) :: i
                end function fstarpu_matrix_get_ld

                function fstarpu_matrix_get_ny(buffers, i) bind(C)
                        use iso_c_binding
                        integer(c_int) :: fstarpu_matrix_get_ny
                        type(c_ptr), value, intent(in) :: buffers
                        integer(c_int), value, intent(in) :: i
                end function fstarpu_matrix_get_ny

                function fstarpu_matrix_get_nx(buffers, i) bind(C)
                        use iso_c_binding
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

                subroutine fstarpu_task_wait_for_all () bind(C,name="starpu_task_wait_for_all")
                end subroutine fstarpu_task_wait_for_all

        end interface

        contains

                subroutine fstarpu_init ()

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

                                subroutine fstarpu_init_internal () bind(C)
                                end subroutine fstarpu_init_internal
                        end interface

                        ! Initialize Fortran integer constants from C peers
                        FSTARPU_R = fstarpu_get_integer_constant(C_CHAR_"FSTARPU_R"//C_NULL_CHAR)
                        FSTARPU_W = fstarpu_get_integer_constant(C_CHAR_"FSTARPU_W"//C_NULL_CHAR)
                        FSTARPU_RW = fstarpu_get_integer_constant(C_CHAR_"FSTARPU_RW"//C_NULL_CHAR)
                        FSTARPU_SCRATCH = fstarpu_get_integer_constant(C_CHAR_"FSTARPU_SCRATCH"//C_NULL_CHAR)
                        FSTARPU_REDUX = fstarpu_get_integer_constant(C_CHAR_"FSTARPU_REDUX"//C_NULL_CHAR)
                        ! Initialize Fortran 'pointer' constants from C peers
                        FSTARPU_DATA = fstarpu_get_pointer_constant(C_CHAR_"FSTARPU_DATA"//C_NULL_CHAR)
                        ! Initialize StarPU
                        call fstarpu_init_internal()
                end subroutine fstarpu_init
end module fstarpu_mod
