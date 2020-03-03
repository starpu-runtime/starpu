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
module nf_mm_cl
contains
subroutine mat_disp (m)
        ! declared here so it can be used both for the
        ! program and for debugging codelet routines

        use iso_c_binding       ! C interfacing module
        implicit none
        real(kind=c_double) :: m(:,:)
        integer i,j

        do i=lbound(m,1),ubound(m,1)
                write(*, fmt="(A2) ",advance="no") "| "
        do j=lbound(m,2),ubound(m,2)
                write(*, fmt="(F6.1,A1) ", advance="no") m(i,j)," "
        end do
                write(*,*) "|"
        end do
        write(*,*)

end subroutine

recursive subroutine cl_cpu_mult (buffers, cl_args) bind(C)
        use iso_c_binding       ! C interfacing module
        use fstarpu_mod         ! StarPU interfacing module
        implicit none

        type(c_ptr), value, intent(in) :: buffers, cl_args ! cl_args is unused
        real(kind=c_double),pointer :: A(:,:), B(:,:), C(:,:)
        integer :: ld_A,nx_A,ny_A
        integer :: ld_B,nx_B,ny_B
        integer :: ld_C,nx_C,ny_C
        integer :: i,j,k

        ld_A = fstarpu_matrix_get_ld(buffers, 0)
        ld_B = fstarpu_matrix_get_ld(buffers, 1)
        ld_C = fstarpu_matrix_get_ld(buffers, 2)

        nx_A = fstarpu_matrix_get_nx(buffers, 0)
        nx_B = fstarpu_matrix_get_nx(buffers, 1)
        nx_C = fstarpu_matrix_get_nx(buffers, 2)

        ny_A = fstarpu_matrix_get_ny(buffers, 0)
        ny_B = fstarpu_matrix_get_ny(buffers, 1)
        ny_C = fstarpu_matrix_get_ny(buffers, 2)

        if (ny_C /= ny_B) then
                write(*,*) "C -- B column mismatch"
                stop 1
        end if

        if (nx_C /= nx_A) then
                write(*,*) "C -- A row mismatch"
                stop 1
        end if

        if (ny_A /= nx_B) then
                write(*,*) "A -- B col/row mismatch"
                stop 1
        end if

        call c_f_pointer(fstarpu_matrix_get_ptr(buffers, 0), A, shape=[ld_A,ny_A])
        call c_f_pointer(fstarpu_matrix_get_ptr(buffers, 1), B, shape=[ld_B,ny_B])
        call c_f_pointer(fstarpu_matrix_get_ptr(buffers, 2), C, shape=[ld_C,ny_C])

        do k = 1, ny_C
        do j = 1, nx_C
        do i = 1, nx_B
                C(j,k) = C(j,k) + A(j,i) * B(i,k)
        end do
        end do
        end do

end subroutine cl_cpu_mult
end module nf_mm_cl
