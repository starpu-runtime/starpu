! StarPU --- Runtime system for heterogeneous multicore architectures.
!
! Copyright (C) 2016-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
module nf_mm_cl_blas
contains

recursive subroutine cl_cpu_gemm (buffers, cl_args) bind(C)
        use iso_c_binding       ! C interfacing module
        use fstarpu_mod         ! StarPU interfacing module
        implicit none

        type(c_ptr), value, intent(in) :: buffers, cl_args ! cl_args is unused
        real, target                   :: alpha, beta
        real(kind=c_double),pointer    :: A(:,:), B(:,:), C(:,:)
        integer :: ld_A,nx_A,ny_A
        integer :: ld_B,nx_B,ny_B
        integer :: ld_C,nx_C,ny_C
        integer :: i,j,k

        write(*,*) "gemm task"
        call fstarpu_unpack_arg( cl_args, (/ c_loc(alpha), c_loc(beta) /))

        ld_A = fstarpu_matrix_get_ld(buffers, 0)
        ld_B = fstarpu_matrix_get_ld(buffers, 1)
        ld_C = fstarpu_matrix_get_ld(buffers, 2)

        nx_A = fstarpu_matrix_get_nx(buffers, 0)
        nx_B = fstarpu_matrix_get_nx(buffers, 1)
        nx_C = fstarpu_matrix_get_nx(buffers, 2)

        ny_A = fstarpu_matrix_get_ny(buffers, 0)
        ny_B = fstarpu_matrix_get_ny(buffers, 1)
        ny_C = fstarpu_matrix_get_ny(buffers, 2)

        call c_f_pointer(fstarpu_matrix_get_ptr(buffers, 0), A, shape=[ld_A,ny_A])
        call c_f_pointer(fstarpu_matrix_get_ptr(buffers, 1), B, shape=[ld_B,ny_B])
        call c_f_pointer(fstarpu_matrix_get_ptr(buffers, 2), C, shape=[ld_C,ny_C])
        call dgemm('n','n',nx_C,ny_C,nx_B, alpha, A(1,1), ld_A, B(1,1), ld_B, &
                beta, C(1,1), ld_C)
        write(*,*) "end gemm task"
        return

end subroutine cl_cpu_gemm

recursive subroutine cl_cpu_fill (buffers, cl_args) bind(C)
        use iso_c_binding       ! C interfacing module
        use fstarpu_mod         ! StarPU interfacing module
        use fstarpu_mpi_mod
        implicit none

        type(c_ptr), value, intent(in)             :: cl_args
        type(c_ptr), value, intent(in)             :: buffers

        real(kind=c_double), pointer               :: x(:,:)
        integer                        :: m, n, ld
        integer                        :: j
        integer                        :: iseed(4) = (/1,1,1,1/)

        integer                        :: comm_rank

        comm_rank = fstarpu_mpi_world_rank()

        m   = fstarpu_matrix_get_nx(buffers, 0)
        n   = fstarpu_matrix_get_ny(buffers, 0)
        ld  = fstarpu_matrix_get_ld(buffers, 0)
        write(*,*) comm_rank,"] fill", m, n, ld

        call c_f_pointer(fstarpu_matrix_get_ptr(buffers, 0), x, shape=(/ld,n/))

        ! copied from qrm_dsmat_fill_task a few lines up
        do j=1,n
          call dlarnv(2, iseed(1), m, x(1, j))
        end do
        write(*,*) comm_rank,"]end fill task"
        return

end subroutine cl_cpu_fill

end module nf_mm_cl_blas
