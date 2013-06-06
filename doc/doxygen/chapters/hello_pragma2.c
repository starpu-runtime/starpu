int
main (void)
{
#pragma starpu initialize

#define NX     0x100000
#define FACTOR 3.14

  {
    float vector[NX]
       __attribute__ ((heap_allocated, registered));

    size_t i;
    for (i = 0; i < NX; i++)
      vector[i] = (float) i;

    vector_scal (NX, vector, FACTOR);

#pragma starpu wait
  } /* VECTOR is automatically freed here. */

#pragma starpu shutdown

  return valid ? EXIT_SUCCESS : EXIT_FAILURE;
}
