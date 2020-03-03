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
module nf_partition_cl
contains
        ! 'cl_partition' codelet routine
recursive subroutine cl_partition_func (buffers, cl_args) bind(C)
        use iso_c_binding       ! C interfacing module
        use fstarpu_mod         ! StarPU interfacing module
        implicit none

        type(c_ptr), value, intent(in) :: buffers, cl_args ! cl_args is unused
        real(8), dimension(:,:), pointer :: ma
        integer :: ld_ma,nx_ma,ny_ma
        integer :: i,j

        ld_ma = fstarpu_matrix_get_ld(buffers, 0)
        nx_ma = fstarpu_matrix_get_nx(buffers, 0)
        ny_ma = fstarpu_matrix_get_ny(buffers, 0)
        write(*,*) "ld_ma = ", ld_ma, ", nx_ma = ", nx_ma, ", ny_ma = ", ny_ma

        call c_f_pointer(fstarpu_matrix_get_ptr(buffers, 0), ma, shape=[ld_ma,ny_ma])
        write(*,*) "ma"
        do i=1,nx_ma
        do j=1,ny_ma
                write(*,*) i,j,ma(i,j)
        end do
        write(*,*) '-'
        end do

end subroutine cl_partition_func
end module nf_partition_cl
