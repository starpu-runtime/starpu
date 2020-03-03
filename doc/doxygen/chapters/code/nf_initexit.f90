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
! [To be included. You should update doxygen if you see this text.]
program nf_initexit
        use iso_c_binding       ! C interfacing module
        use fstarpu_mod         ! StarPU interfacing module
        implicit none           ! Fortran recommended best practice

        integer(c_int) :: err   ! return status for fstarpu_init

        ! initialize StarPU with default settings
        err = fstarpu_init(C_NULL_PTR)
        if (err /= 0) then
                stop 1          ! StarPU initialization failure
        end if

        ! - add StarPU Native Fortran API calls here

        ! shut StarPU down
        call fstarpu_shutdown()
end program nf_initexit
! [To be included. You should update doxygen if you see this text.]
