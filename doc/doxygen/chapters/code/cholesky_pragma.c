extern void cholesky(unsigned nblocks, unsigned size,
                    float mat[nblocks][nblocks][size])
  __attribute__ ((task));

int
main (int argc, char *argv[])
{
#pragma starpu initialize

  /* ... */

  int nblocks, size;
  parse_args (&nblocks, &size);

  /* Allocate an array of the required size on the heap,
     and register it.  */

  {
    float matrix[nblocks][nblocks][size]
      __attribute__ ((heap_allocated, registered));

    cholesky (nblocks, size, matrix);

#pragma starpu wait

  }   /* MATRIX is automatically unregistered & freed here.  */

#pragma starpu shutdown

  return EXIT_SUCCESS;
}
