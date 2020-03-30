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
program nf_basic_ring
        use iso_c_binding       ! C interfacing module
        use fstarpu_mod         ! StarPU interfacing module
        use fstarpu_mpi_mod     ! StarPU-MPI interfacing module
        implicit none

        integer(c_int) :: ncpu
        integer(c_int) :: ret
        integer(c_int) :: rank,sz
        integer(c_int),target :: token = 42
        integer(c_int) :: nloops = 32
        integer(c_int) :: loop
        integer(c_int64_t) :: tag
        integer(c_int) :: world
        integer(c_int) :: src,dst
        type(c_ptr) :: token_dh, st

        ret = fstarpu_init(C_NULL_PTR)
        if (ret == -19) then
                stop 77
        else if (ret /= 0) then
                stop 1
        end if

        ret = fstarpu_mpi_init(1)
        print *,"fstarpu_mpi_init status:", ret
        if (ret /= 0) then
                stop 1
        end if

        ! stop there if no CPU worker available
        ncpu = fstarpu_cpu_worker_get_count()
        if (ncpu == 0) then
                call fstarpu_shutdown()
                ret = fstarpu_mpi_shutdown()
                stop 77
        end if

        world = fstarpu_mpi_world_comm()
        rank = fstarpu_mpi_world_rank()
        sz = fstarpu_mpi_world_size()
        write(*,*) "rank=", rank,"size=",sz,"world=",world
        if (sz < 2) then
                call fstarpu_shutdown()
                ret = fstarpu_mpi_shutdown()
                stop 77
        end if

        call fstarpu_variable_data_register(token_dh, 0, c_loc(token), c_sizeof(token))

        st = fstarpu_mpi_status_alloc()
        do loop=1,nloops
                tag = loop*sz+rank
                token = 0
                if (loop == 1.and.rank == 0) then
                        write(*,*) "rank=", rank,"token=",token
                else
                        src = modulo((rank+sz-1),sz)
                        write(*,*) "rank=", rank,"recv--> src =", src, "tag =", tag
                        ret = fstarpu_mpi_recv(token_dh, src, tag, world, st)
                        if (ret /= 0) then
                                write(*,*) "fstarpu_mpi_recv failed"
                                stop 1
                        end if
                        write(*,*) "rank=", rank,"recv<--","token=",token
                        token = token+1
                end if
                if (loop == nloops.and.rank == (sz-1)) then
                        call fstarpu_data_acquire(token_dh, FSTARPU_R)
                        write(*,*) "finished: rank=", rank,"token=",token
                        call fstarpu_data_release(token_dh)
                else
                        dst = modulo((rank+1),sz)
                        write(*,*) "rank=", rank,"send--> dst =", dst, "tag =", tag+1
                        ret = fstarpu_mpi_send(token_dh, dst, tag+1, world)
                        if (ret /= 0) then
                                write(*,*) "fstarpu_mpi_recv failed"
                                stop 1
                        end if
                        write(*,*) "rank=", rank,"send<--"
                end if
        end do
        call fstarpu_mpi_status_free(st)
        call fstarpu_data_unregister(token_dh)
        call fstarpu_shutdown()

        ret = fstarpu_mpi_shutdown()
        print *,"fstarpu_mpi_shutdown status:", ret
        if (ret /= 0) then
                stop 1
        end if
end program nf_basic_ring

