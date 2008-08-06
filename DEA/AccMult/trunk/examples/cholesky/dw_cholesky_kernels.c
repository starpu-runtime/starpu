#include "dw_cholesky.h"

/*
 *   U22 
 */

static inline void chol_common_core_codelet_update_u22(data_interface_t *buffers, int s, __attribute__((unused)) void *_args)
{
	//printf("22\n");
	float *left 	= (float *)buffers[0].blas.ptr;
	float *right 	= (float *)buffers[1].blas.ptr;
	float *center 	= (float *)buffers[2].blas.ptr;

	unsigned dx = buffers[2].blas.nx;
	unsigned dy = buffers[2].blas.ny;
	unsigned dz = buffers[0].blas.nx;

	unsigned ld21 = buffers[0].blas.ld;
	unsigned ld12 = buffers[1].blas.ld;
	unsigned ld22 = buffers[2].blas.ld;

	switch (s) {
		case 0:
			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 
				dy, dx, dz, -1.0f, left, ld21, right, ld12,
					     1.0f, center, ld22);
			break;
#if defined (USE_CUBLAS) || defined (USE_CUDA)
		case 1:
			cublasSgemm('t', 'n', dx, dy, dz, 
					-1.0f, right, ld12, left, ld21, 
					 1.0f, center, ld22);
			break;
#endif
		default:
			ASSERT(0);
			break;
	}
}

void chol_core_codelet_update_u22(data_interface_t *descr, void *_args)
{
	chol_common_core_codelet_update_u22(descr, 0, _args);
}

#if defined (USE_CUBLAS) || defined (USE_CUDA)
void chol_cublas_codelet_update_u22(data_interface_t *descr, void *_args)
{
	chol_common_core_codelet_update_u22(descr, 1, _args);
}
#endif// USE_CUBLAS

/* 
 * U21
 */

static inline void chol_common_codelet_update_u21(data_interface_t *buffers, int s, __attribute__((unused)) void *_args)
{
//	printf("21\n");
	float *sub11;
	float *sub21;

	sub11 = (float *)buffers[0].blas.ptr;
	sub21 = (float *)buffers[1].blas.ptr;

	unsigned ld11 = buffers[0].blas.ld;
	unsigned ld21 = buffers[1].blas.ld;

	unsigned nx21 = buffers[1].blas.nx;
	unsigned ny21 = buffers[1].blas.ny;

	switch (s) {
		case 0:
			cblas_strsm(CblasRowMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit,
					nx21, ny21, 1.0f, sub11, ld11, sub21, ld21);
			break;
#if defined (USE_CUBLAS) || defined (USE_CUDA)
		case 1:
			cublasStrsm('L', 'U', 'T', 'N', ny21, nx21, 1.0f, sub11, ld11, sub21, ld21);
			break;
#endif
		default:
			ASSERT(0);
			break;
	}
}

void chol_core_codelet_update_u21(data_interface_t *descr, void *_args)
{
	 chol_common_codelet_update_u21(descr, 0, _args);
}

#if defined (USE_CUBLAS) || defined (USE_CUDA)
void chol_cublas_codelet_update_u21(data_interface_t *descr, void *_args)
{
	chol_common_codelet_update_u21(descr, 1, _args);
}
#endif 

/*
 *	U11
 */

static inline void chol_common_codelet_update_u11(data_interface_t *descr, int s, __attribute__((unused)) void *_args) 
{
//	printf("11\n");
	float *sub11;

	sub11 = (float *)descr[0].blas.ptr; 

	unsigned nx = descr[0].blas.nx;
	unsigned ld = descr[0].blas.ld;

	unsigned z;

	switch (s) {
		case 0:

			/*
			 *	- alpha 11 <- lambda 11 = sqrt(alpha11)
			 *	- alpha 21 <- l 21	= alpha 21 / lambda 11
			 *	- A22 <- A22 - l21 trans(l21)
			 */

			for (z = 0; z < nx; z++)
			{
				float lambda11;
				lambda11 = sqrt(sub11[z+z*ld]);
				sub11[z+z*ld] = lambda11;

				ASSERT(lambda11 != 0.0f);
		
				cblas_sscal(nx - z - 1, 1.0f/lambda11, &sub11[(z)+(z+1)*ld], ld);
		
				cblas_ssyr(CblasRowMajor, CblasLower, nx - z - 1, -1.0f, 
							&sub11[(z)+(z+1)*ld], ld,
							&sub11[(z+1)+(z+1)*ld], ld);
			}
			break;
#if defined (USE_CUBLAS) || defined (USE_CUDA)
		case 1:
			for (z = 0; z < nx; z++)
			{
				float lambda11;
				/* ok that's dirty and ridiculous ... */
				cublasGetVector(1, sizeof(float), &sub11[z+z*ld], sizeof(float), &lambda11, sizeof(float));

				lambda11 = sqrt(lambda11);

				cublasSetVector(1, sizeof(float), &lambda11, sizeof(float), &sub11[z+z*ld], sizeof(float));

				ASSERT(lambda11 != 0.0f);
				
				cublasSscal(nx - z - 1, 1.0f/lambda11, &sub11[(z)+(z+1)*ld], ld);

				cublasSsyr('U', nx - z - 1, -1.0f,
							&sub11[(z)+(z+1)*ld], ld,
							&sub11[(z+1)+(z+1)*ld], ld);
			}
			break;
#endif
		default:
			ASSERT(0);
			break;
	}

}


void chol_core_codelet_update_u11(data_interface_t *descr, void *_args)
{
	chol_common_codelet_update_u11(descr, 0, _args);
}

#if defined (USE_CUBLAS) || defined (USE_CUDA)
void chol_cublas_codelet_update_u11(data_interface_t *descr, void *_args)
{
	chol_common_codelet_update_u11(descr, 1, _args);
}
#endif// USE_CUBLAS
