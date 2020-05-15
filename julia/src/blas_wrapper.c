#include <starpu.h>
#include <blas.h>

#if defined(STARPU_ATLAS) || defined(STARPU_OPENBLAS) || defined(STARPU_MKL)
void julia_saxpy_cpu_codelet(void *descr[], void *arg)
{
	float alpha = *((float *)arg);

	unsigned n = STARPU_VECTOR_GET_NX(descr[0]);

	float *block_x = (float *)STARPU_VECTOR_GET_PTR(descr[0]);
	float *block_y = (float *)STARPU_VECTOR_GET_PTR(descr[1]);

	STARPU_SAXPY((int)n, alpha, block_x, 1, block_y, 1);
}
#endif

#ifdef STARPU_USE_CUDA

#include <starpu_cublas_v2.h>

void julia_saxpy_cuda_codelet(void *descr[], void *arg)
{
	float alpha = *((float *)arg);

	unsigned n = STARPU_VECTOR_GET_NX(descr[0]);

	float *block_x = (float *)STARPU_VECTOR_GET_PTR(descr[0]);
	float *block_y = (float *)STARPU_VECTOR_GET_PTR(descr[1]);

	cublasStatus_t status = cublasSaxpy(starpu_cublas_get_local_handle(), (int)n, &alpha, block_x, 1, block_y, 1);
	if (status != CUBLAS_STATUS_SUCCESS)
		STARPU_CUBLAS_REPORT_ERROR(status);
}
#endif
