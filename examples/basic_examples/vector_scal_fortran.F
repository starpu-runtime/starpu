      PROGRAM VECTOR_SCAL
      INTEGER,PARAMETER :: F_NX=2048
      REAL,DIMENSION(F_NX) :: VECTOR

      INTEGER :: I
      DO I=1,F_NX,1
         VECTOR(I)=1.0
      ENDDO

      WRITE (*,*) ' BEFORE : First element was ', VECTOR(1)
      WRITE (*,*) ' BEFORE : Last element was ', VECTOR(F_NX)

      CALL COMPUTE(F_NX, VECTOR)

      WRITE (*,*) ' AFTER : First element is ', VECTOR(1)
      WRITE (*,*) ' AFTER : Last element is ', VECTOR(F_NX)

      END PROGRAM
