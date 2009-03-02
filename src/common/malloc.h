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
	
		codelet *cl = malloc(sizeof(codelet));
		cl->cublas_func = malloc_pinned_codelet; 
		cl->cl_arg = &s;
	
		job_t j = job_create();
		j->where = CUBLAS;
		j->cb = NULL; 
		j->cl = cl;
	
		push_res = submit_job_sync(j);
		STARPU_ASSERT(push_res != -ENODEV);
#endif
	}
	else {
		*A = malloc(dim);
	}
}




#endif // __MALLOC_H__
