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
module nf_sched_ctx_cl
contains
recursive subroutine cl_cpu_func_sched_ctx (buffers, cl_args) bind(C)
        use iso_c_binding       ! C interfacing module
        use fstarpu_mod         ! StarPU interfacing module
        implicit none

        interface
                function sleep(s) bind(C)
                        use iso_c_binding
                        integer(c_int) :: sleep
                        integer(c_int), value, intent(in) :: s
                end function
        end interface

        type(c_ptr), value, intent(in) :: buffers, cl_args ! cl_args is unused
        integer(c_int),target :: id
        integer(c_int) :: worker_id
        integer(c_int) :: ret

        call fstarpu_unpack_arg(cl_args,(/ c_loc(id) /))
        ! ret = sleep(1)
        worker_id = fstarpu_worker_get_id()
        write(*,*) "task:", id, ", worker_id:", worker_id
end subroutine cl_cpu_func_sched_ctx
end module nf_sched_ctx_cl
