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
module nf_varbuf_cl
contains
recursive subroutine cl_cpu_func_varbuf (buffers, cl_args) bind(C)
        use iso_c_binding       ! C interfacing module
        use fstarpu_mod         ! StarPU interfacing module
        implicit none

        type(c_ptr), value, intent(in) :: buffers, cl_args ! cl_args is unused
        integer(c_int),target :: nb_data
        integer(c_int),pointer :: val
        integer(c_int) :: i

        call fstarpu_unpack_arg(cl_args,(/ c_loc(nb_data) /))
        write(*,*) "number of data:", nb_data
        do i=0,nb_data-1
                call c_f_pointer(fstarpu_variable_get_ptr(buffers, i), val)
                write(*,*) "i:", i, ", val:", val
                if (val /= 42) then
                        stop 1
                end if
        end do
end subroutine cl_cpu_func_varbuf
end module nf_varbuf_cl
