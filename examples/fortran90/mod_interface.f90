! StarPU --- Runtime system for heterogeneous multicore architectures.
!
! Copyright (C) 2015-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
! Copyright (C) 2015       ONERA
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
! Fortran module interface for StarPU initialization and element registration

MODULE mod_interface

  INTERFACE
     FUNCTION starpu_my_init_c() BIND(C)
       USE iso_c_binding
       INTEGER(KIND=C_INT)                   :: starpu_my_init_c
     END FUNCTION starpu_my_init_c
  END INTERFACE

  INTERFACE
     SUBROUTINE starpu_register_element_c(Neq,Np,Ng,ro,dro,basis,ro_h,dro_h,basis_h) BIND(C)
       USE iso_c_binding
       INTEGER(KIND=C_INT),VALUE             :: Neq,Np,Ng
       REAL(KIND=C_DOUBLE),DIMENSION(Neq,Np) :: ro,dro
       REAL(KIND=C_DOUBLE),DIMENSION(Np,Ng)  :: basis
       TYPE(C_PTR), INTENT(OUT)              :: ro_h, dro_h, basis_h
     END SUBROUTINE starpu_register_element_c
  END INTERFACE

  INTERFACE
     SUBROUTINE starpu_unregister_element_c( &
               ro_h,dro_h,basis_h) BIND(C)
       USE iso_c_binding
       TYPE(C_PTR), INTENT(IN)               :: ro_h, dro_h, basis_h
     END SUBROUTINE starpu_unregister_element_c
  END INTERFACE

  INTERFACE
     SUBROUTINE starpu_loop_element_task_c(coeff, &
               ro_h,dro_h,basis_h) BIND(C)
       USE iso_c_binding
       REAL(KIND=C_DOUBLE),VALUE             :: coeff
       TYPE(C_PTR), INTENT(IN)               :: ro_h, dro_h, basis_h
     END SUBROUTINE starpu_loop_element_task_c
  END INTERFACE

  INTERFACE
     SUBROUTINE starpu_copy_element_task_c( &
               ro_h,dro_h) BIND(C)
       USE iso_c_binding
       TYPE(C_PTR), INTENT(IN)               :: ro_h, dro_h
     END SUBROUTINE starpu_copy_element_task_c
  END INTERFACE

END MODULE mod_interface
