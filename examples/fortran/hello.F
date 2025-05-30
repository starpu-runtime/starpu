C StarPU --- Runtime system for heterogeneous multicore architectures.
C
C Copyright (C) 2009-2025   University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
C
C StarPU is free software; you can redistribute it and/or modify
C it under the terms of the GNU Lesser General Public License as published by
C the Free Software Foundation; either version 2.1 of the License, or (at
C your option) any later version.
C
C StarPU is distributed in the hope that it will be useful, but
C WITHOUT ANY WARRANTY; without even the implied warranty of
C MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
C
C See the GNU Lesser General Public License in COPYING.LGPL for more details.
C

#include "starpu_fortran.h"

      SUBROUTINE HELLOSUB(X)
          INTEGER :: X
          WRITE(*,*) 'X =', X
      END SUBROUTINE

      PROGRAM HELLO
          USE STARPU_FORTRAN
          USE ISO_C_BINDING

          INTEGER :: TOTO
          TOTO = 42

          CALL STARPU_SUBMIT_CODELET(PRINT_INT, TOTO)
C           CALL STARPU_SUBMIT_CODELET(HELLOSUB, TOTO)

C          CALL DUMMY_C_FUNC(TOTO)
      END PROGRAM
