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

program nf_mm
        use iso_c_binding       ! C interfacing module
        use fstarpu_mod         ! StarPU interfacing module
        use fstarpu_mpi_mod     ! StarPU-MPI interfacing module
        use nf_mm_cl
        implicit none

        integer(c_int) :: ncpu
        integer(c_int) :: ret

        ret = fstarpu_mpi_init(1)
        print *,"fstarpu_mpi_init status:", ret
        if (ret /= 0) then
                stop 1
        end if

        ret = fstarpu_init(C_NULL_PTR)
        if (ret == -19) then
                stop 77
        else if (ret /= 0) then
                stop 1
        end if

        ! stop there if no CPU worker available
        ncpu = fstarpu_cpu_worker_get_count()
        if (ncpu == 0) then
                call fstarpu_shutdown()
                stop 77
        end if

        call fstarpu_shutdown()

        ret = fstarpu_mpi_shutdown()
        print *,"fstarpu_mpi_shutdown status:", ret
        if (ret /= 0) then
                stop 1
        end if
end program nf_mm
