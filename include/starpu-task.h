#ifndef __STARPU_TASK_H__
#define __STARPU_TASK_H__

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

	struct perfmodel_t *model;
} codelet;



#endif // __STARPU_TASK_H__
