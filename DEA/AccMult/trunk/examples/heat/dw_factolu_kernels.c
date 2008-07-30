#include "dw_factolu.h"

/*
 *   U22 
 */

static inline void dw_common_core_codelet_update_u22(data_interface_t *buffers, int s, __attribute__((unused)) void *_args)
{
	float *left 	= (float *)buffers[0].blas.ptr;
	float *right 	= (float *)buffers[1].blas.ptr;
	float *center 	= (float *)buffers[2].blas.ptr;

	unsigned dx = buffers[2].blas.nx;
	unsigned dy = buffers[2].blas.ny;
	unsigned dz = buffers[0].blas.ny;

	unsigned ld12 = buffers[0].blas.ld;
	unsigned ld21 = buffers[1].blas.ld;
	unsigned ld22 = buffers[2].blas.ld;

	switch (s) {
		case 0:
			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
				dy, dx, dz, -1.0f, left, ld21, right, ld12,
					     1.0f, center, ld22);
			break;
#if defined (USE_CUBLAS) || defined (USE_CUDA)

		case 1:
			cublasSgemm('n', 'n', dx, dy, dz, -1.0f, left, ld21,
					right, ld12, 1.0f, center, ld22);
			break;
#endif
		default:
			ASSERT(0);
			break;
	}
}

void dw_core_codelet_update_u22(data_interface_t *descr, void *_args)
{
	dw_common_core_codelet_update_u22(descr, 0, _args);
}

#if defined (USE_CUBLAS) || defined (USE_CUDA)
void dw_cublas_codelet_update_u22(data_interface_t *descr, void *_args)
{
	dw_common_core_codelet_update_u22(descr, 1, _args);
}
#endif// USE_CUBLAS

/*
 * U12
 */

static inline void dw_common_codelet_update_u12(data_interface_t *buffers, int s, __attribute__((unused)) void *_args) {
	float *sub11;
	float *sub12;

	sub11 = (float *)buffers[0].blas.ptr;	
	sub12 = (float *)buffers[1].blas.ptr;

	unsigned ld11 = buffers[0].blas.ld;
	unsigned ld12 = buffers[1].blas.ld;

	unsigned nx12 = buffers[1].blas.nx;
	unsigned ny12 = buffers[1].blas.ny;

	/* solve L11 U12 = A12 (find U12) */
	switch (s) {
		case 0:
			cblas_strsm(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit,
					 nx12, ny12, 1.0f, sub11, ld11, sub12, ld12);
			break;
#if defined (USE_CUBLAS) || defined (USE_CUDA)
		case 1:
			cublasStrsm('R', 'U', 'N', 'N', ny12, nx12,
					1.0f, sub11, ld11, sub12, ld12);
			break;
#endif
		default:
			ASSERT(0);
			break;
	}
}

void dw_core_codelet_update_u12(data_interface_t *descr, void *_args)
{
	 dw_common_codelet_update_u12(descr, 0, _args);
}

#if defined (USE_CUBLAS) || defined (USE_CUDA)
void dw_cublas_codelet_update_u12(data_interface_t *descr, void *_args)
{
	 dw_common_codelet_update_u12(descr, 1, _args);
}
#endif // USE_CUBLAS

/* 
 * U21
 */

static inline void dw_common_codelet_update_u21(data_interface_t *buffers, int s, __attribute__((unused)) void *_args) {
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
			cblas_strsm(CblasRowMajor, CblasRight, CblasUpper, CblasNoTrans, 
				CblasUnit, nx21, ny21, 1.0f, sub11, ld11, sub21, ld21);
			break;
#if defined (USE_CUBLAS) || defined (USE_CUDA)
		case 1:
			cublasStrsm('L', 'L', 'N', 'U', ny21, nx21, 1.0f, sub11, ld11, sub21, ld21);
			break;
#endif
		default:
			ASSERT(0);
			break;
	}
}

void dw_core_codelet_update_u21(data_interface_t *descr, void *_args)
{
	 dw_common_codelet_update_u21(descr, 0, _args);
}

#if defined (USE_CUBLAS) || defined (USE_CUDA)
void dw_cublas_codelet_update_u21(data_interface_t *descr, void *_args)
{
	dw_common_codelet_update_u21(descr, 1, _args);
}
#endif 

/*
 *	U11
 */

static inline void dw_common_codelet_update_u11(data_interface_t *descr, int s, __attribute__((unused)) void *_args) 
{
	float *sub11;

	sub11 = (float *)descr[0].blas.ptr; 

	unsigned nx = descr[0].blas.nx;
	unsigned ld = descr[0].blas.ld;

	unsigned z;

	switch (s) {
		case 0:
			for (z = 0; z < nx; z++)
			{
				float pivot;
				pivot = sub11[z+z*ld];
				ASSERT(pivot != 0.0f);
		
				cblas_sscal(nx - z - 1, 1.0f/pivot, &sub11[(z+1)+z*ld], 1);
		
				cblas_sger(CblasRowMajor, nx - z - 1, nx - z - 1, -1.0f,
								&sub11[(z+1)+z*ld], 1,
								&sub11[z+(z+1)*ld], ld,
								&sub11[(z+1) + (z+1)*ld],ld);
		
			}
			break;
#if defined (USE_CUBLAS) || defined (USE_CUDA)
		case 1:
			for (z = 0; z < nx; z++)
			{
				float pivot;
				/* ok that's dirty and ridiculous ... */
				cublasGetVector(1, sizeof(float), &sub11[z+z*ld], sizeof(float), &pivot, sizeof(float));

				ASSERT(pivot != 0.0f);
				
				cublasSscal(nx - z - 1, 1.0f/pivot, &sub11[(z+1)+z*ld], 1);
				
				cublasSger(nx - z - 1, nx - z - 1, -1.0f,
								&sub11[(z+1)+z*ld], 1,
								&sub11[z+(z+1)*ld], ld,
								&sub11[(z+1) + (z+1)*ld],ld);
			}
			break;
#endif
		default:
			ASSERT(0);
			break;
	}

}


void dw_core_codelet_update_u11(data_interface_t *descr, void *_args)
{
	dw_common_codelet_update_u11(descr, 0, _args);
}

#if defined (USE_CUBLAS) || defined (USE_CUDA)
void dw_cublas_codelet_update_u11(data_interface_t *descr, void *_args)
{
	dw_common_codelet_update_u11(descr, 1, _args);
}
#endif// USE_CUBLAS
