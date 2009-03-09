#ifndef __STARPU_H__
#define __STARPU_H__

#include <stdlib.h>
#include <stdint.h>

/*
 * A codelet describes the various function 
 * that may be called from a worker
 */
typedef struct codelet_t {
	/* where can it be performed ? */
	uint32_t where;

	/* the different implementations of the codelet */
	void *cuda_func;
	void *cublas_func;
	void *core_func;
	void *spu_func;
	uint8_t gordon_func;

	/* the arguments are given as a buffer */
	void *cl_arg;
	/* in case the argument buffer has to be uploaded explicitely */
	size_t cl_arg_size;
	
	struct perfmodel_t *model;
} codelet;

#endif // __STARPU_H__
