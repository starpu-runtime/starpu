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
module nf_codelets
contains
        ! 'cl_vec' codelet routine
        !
        ! Note: codelet routines must:
        ! . be declared recursive (~ 'reentrant routine')
        ! . be declared with the 'bind(C)' attribute for proper C interfacing
recursive subroutine cl_cpu_func_vec (buffers, cl_args) bind(C)
        use iso_c_binding       ! C interfacing module
        use fstarpu_mod         ! StarPU interfacing module
        implicit none

        type(c_ptr), value, intent(in) :: buffers, cl_args ! cl_args is unused
        real(8), dimension(:), pointer :: va
        integer, dimension(:), pointer :: vb
        integer :: nx_va,nx_vb,i

        write(*,*) "task -->"
        ! get 'va' number of elements
        nx_va = fstarpu_vector_get_nx(buffers, 0)
        write(*,*) "nx_va"
        write(*,*) nx_va

        ! get 'vb' number of elements
        nx_vb = fstarpu_vector_get_nx(buffers, 1)
        write(*,*) "nx_vb"
        write(*,*) nx_vb

        ! get 'va' converted Fortran pointer
        call c_f_pointer(fstarpu_vector_get_ptr(buffers, 0), va, shape=[nx_va])
        write(*,*) "va"
        do i=1,nx_va
                write(*,*) i,va(i)
        end do

        ! get 'vb' converted Fortran pointer
        call c_f_pointer(fstarpu_vector_get_ptr(buffers, 1), vb, shape=[nx_vb])
        write(*,*) "vb"
        do i=1,nx_vb
                write(*,*) i,vb(i)
        end do
        write(*,*) "task <--"

end subroutine cl_cpu_func_vec

        ! 'cl_mat' codelet routine
recursive subroutine cl_cpu_func_mat (buffers, cl_args) bind(C)
        use iso_c_binding       ! C interfacing module
        use fstarpu_mod         ! StarPU interfacing module
        implicit none

        type(c_ptr), value, intent(in) :: buffers, cl_args ! cl_args is unused
        real(8), dimension(:,:), pointer :: ma
        integer, dimension(:,:), pointer :: mb
        integer :: ld_ma,nx_ma,ny_ma
        integer :: ld_mb,nx_mb,ny_mb
        integer :: i,j

        write(*,*) "task -->"
        ld_ma = fstarpu_matrix_get_ld(buffers, 0)
        nx_ma = fstarpu_matrix_get_nx(buffers, 0)
        ny_ma = fstarpu_matrix_get_ny(buffers, 0)
        write(*,*) "ld_ma"
        write(*,*) ld_ma
        write(*,*) "nx_ma"
        write(*,*) nx_ma
        write(*,*) "ny_ma"
        write(*,*) ny_ma

        ld_mb = fstarpu_matrix_get_ld(buffers, 1)
        nx_mb = fstarpu_matrix_get_nx(buffers, 1)
        ny_mb = fstarpu_matrix_get_ny(buffers, 1)
        write(*,*) "ld_mb"
        write(*,*) ld_mb
        write(*,*) "nx_mb"
        write(*,*) nx_mb
        write(*,*) "ny_mb"
        write(*,*) ny_mb

        call c_f_pointer(fstarpu_matrix_get_ptr(buffers, 0), ma, shape=[ld_ma,ny_ma])
        write(*,*) "ma"
        do i=1,nx_ma
        do j=1,ny_ma
                write(*,*) i,j,ma(i,j)
        end do
        write(*,*) '-'
        end do

        call c_f_pointer(fstarpu_matrix_get_ptr(buffers, 1), mb, shape=[ld_mb,ny_mb])
        write(*,*) "mb"
        do i=1,nx_mb
        do j=1,ny_mb
                write(*,*) i,j,mb(i,j)
        end do
        write(*,*) '-'
        end do
        write(*,*) "task <--"

end subroutine cl_cpu_func_mat
end module nf_codelets
