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
