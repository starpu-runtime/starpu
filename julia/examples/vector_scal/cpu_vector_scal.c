#include <stdio.h>
#include <stdint.h>
#include <starpu.h>
#include <math.h>

struct params {
  int32_t m;
  float k;
  float l;
};

float cpu_vector_scal(void *buffers[], void *cl_arg)
{
  /* get scalar parameters from cl_arg */
  struct params *scalars = (struct params *) cl_arg;
  int m = scalars->m;
  float k = scalars->k;
  float l = scalars->l;

  struct starpu_vector_interface *vector = (struct starpu_vector_interface *) buffers[0];

  /* length of the vector */
  unsigned n = STARPU_VECTOR_GET_NX(vector);

  /* get a pointer to the local copy of the vector : note that we have to
   * cast it in (float *) since a vector could contain any type of
   * elements so that the .ptr field is actually a uintptr_t */
  float *val = (float *)STARPU_VECTOR_GET_PTR(vector);

  /* scale the vector */
  for (unsigned i = 0; i < n; i++)
    val[i] = val[i] * k + l + m;

  return 0.0;
}

char* CPU = "cpu_vector_scal";
char* GPU = "gpu_vector_scal";
extern char *starpu_find_function(char *name, char *device) {
	if (!strcmp(device,"gpu")) return GPU;
	return CPU;
}
