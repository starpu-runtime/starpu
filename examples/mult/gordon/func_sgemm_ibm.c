#include "func_sgemm_ibm.h"

#include <blas_s.h>

void func_sgemm_ibm(__attribute__ ((unused)) void **alloc,
		__attribute__ ((unused)) void **in,
		__attribute__ ((unused)) void **inout,
		__attribute__ ((unused)) void **out)
{
	/* we assume data will be in A:R,B:R,C:RW mode
 	 *  -> in[0] : describe problem
 	 *  -> in[1] : A
 	 *  -> in[2] : B
 	 *  -> inout[0] : C
 	 *
 	 *   C = AB + C
 	 *   but, being in fortran ordering, we compute
 	 *   t(C) = t(B)t(A) + t(C) instead
 	 */
	struct ibm_sgemm_block_conf *conf = in[0];
	float *A = in[1];
	float *B = in[2];
	float *C = inout[0];

	sgemm_spu(conf->m, conf->n, conf->k, B, A, C);
}
