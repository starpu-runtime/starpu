/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010  Université de Bordeaux 1
 * Copyright (C) 2010  Centre National de la Recherche Scientifique
 *
 * StarPU is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * StarPU is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */
C
C StarPU
C Copyright (C) Université Bordeaux 1, CNRS 2008-2010 (see AUTHORS file)
C
C This program is free software; you can redistribute it and/or modify
C it under the terms of the GNU Lesser General Public License as published by
C the Free Software Foundation; either version 2.1 of the License, or (at
C your option) any later version.
C
C This program is distributed in the hope that it will be useful, but
C WITHOUT ANY WARRANTY; without even the implied warranty of
C MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
C
C See the GNU Lesser General Public License in COPYING.LGPL for more details.
C

      MODULE STARPU_FORTRAN
        USE ISO_C_BINDING

          TYPE codelet
              REAL :: A,B,C
          END TYPE codelet

      CONTAINS
      
          SUBROUTINE PRINT_INT(X)
              INTEGER :: X
              WRITE(*,*) 'X =', X
          END SUBROUTINE

          SUBROUTINE STARPU_SUBMIT_CODELET(CPUFUNC, ARG)
              INTEGER :: ARG

              INTERFACE
                  SUBROUTINE CPUFUNC(ARG)
                      INTEGER :: ARG
                  END SUBROUTINE
              END INTERFACE

              CALL CPUFUNC(ARG)
          END SUBROUTINE

      END MODULE STARPU_FORTRAN

      MODULE STARPU_FORTRAN2
        USE ISO_C_BINDING

      CONTAINS
          SUBROUTINE PRINT_INT2(X)
              INTEGER :: X
              WRITE(*,*) 'X =', X
          END SUBROUTINE

      END MODULE STARPU_FORTRAN2
