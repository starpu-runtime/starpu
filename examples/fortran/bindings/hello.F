#include "StarPU-fortran.h"
      
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
