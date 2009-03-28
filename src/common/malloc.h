#ifndef __MALLOC_H__
#define __MALLOC_H__

#include <errno.h>
#include <core/jobs.h>
#include <core/policies/sched_policy.h>
#ifdef USE_CUDA
#include <cuda.h>
#endif


/* This method is not optimal at all, but it makes life much easier in many codes */

#ifdef USE_CUDA
struct data_interface_s;

struct malloc_pinned_codelet_struct {
	float **ptr;
	size_t dim;
};

static void malloc_pinned_codelet(struct data_interface_s *buffers __attribute__((unused)), void *arg)
{
	struct malloc_pinned_codelet_struct *s = arg;

	cuMemAllocHost((void **)(s->ptr), s->dim);
}
#endif

static inline void malloc_pinned_if_possible(float **A, size_t dim)
{
	if (may_submit_cuda_task())
	{
#ifdef USE_CUDA
		int push_res;
	
		struct malloc_pinned_codelet_struct s = {
			.ptr = A,
			.dim = dim
		};	
	
		starpu_codelet *cl = malloc(sizeof(starpu_codelet));
			cl->cublas_func = malloc_pinned_codelet; 
			cl->where = CUBLAS;
			cl->model = NULL;
			cl->nbuffers = 0;
	
		struct starpu_task *task = starpu_task_create();
			task->callback_func = NULL; 
			task->cl = cl;
			task->cl_arg = &s;

		task->synchronous = 1;
	
		push_res = starpu_submit_task(task);
		STARPU_ASSERT(push_res != -ENODEV);

		free(cl);
		free(task);
#endif
	}
	else {
		*A = malloc(dim);
	}
}




#endif // __MALLOC_H__
