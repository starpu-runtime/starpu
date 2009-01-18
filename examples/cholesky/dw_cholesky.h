#ifndef __DW_CHOLESKY_H__
#define __DW_CHOLESKY_H__

#include <semaphore.h>
#include <core/jobs.h>
#include <core/workers.h>
#include <common/timing.h>
#include <common/util.h>
#include <common/blas.h>
#include <string.h>
#include <math.h>
#ifdef USE_CUDA
#include <cuda.h>
#endif


#include <datawizard/datawizard.h>
#include <core/dependencies/tags.h>
#include <common/malloc.h>
//#include "lu_kernels_model.h"

#define NMAXBLOCKS	32

#define TAG11(k)	( (1ULL<<60) | (unsigned long long)(k))
#define TAG21(k,j)	(((3ULL<<60) | (((unsigned long long)(k))<<32)	\
					| (unsigned long long)(j)))
#define TAG22(k,i,j)	(((4ULL<<60) | ((unsigned long long)(k)<<32) 	\
					| ((unsigned long long)(i)<<16)	\
					| (unsigned long long)(j)))

#define BLOCKSIZE	(size/nblocks)


#define BLAS3_FLOP(n1,n2,n3)    \
        (2*((uint64_t)n1)*((uint64_t)n2)*((uint64_t)n3))

typedef struct {
	data_state *dataA;
	unsigned i;
	unsigned j;
	unsigned k;
	unsigned nblocks;
	unsigned *remaining;
	sem_t *sem;
} cl_args;

static unsigned size = 4*1024;
static unsigned nblocks = 4;
static unsigned pinned = 0;

void chol_core_codelet_update_u11(data_interface_t *, void *);
void chol_core_codelet_update_u21(data_interface_t *, void *);
void chol_core_codelet_update_u22(data_interface_t *, void *);

#ifdef USE_CUDA
void chol_cublas_codelet_update_u11(data_interface_t *descr, void *_args);
void chol_cublas_codelet_update_u21(data_interface_t *descr, void *_args);
void chol_cublas_codelet_update_u22(data_interface_t *descr, void *_args);
#endif

void initialize_system(float **A, unsigned dim, unsigned pinned);
void dw_cholesky(float *matA, unsigned size, unsigned ld, unsigned nblocks);

extern struct perfmodel_t chol_model_11;
extern struct perfmodel_t chol_model_21;
extern struct perfmodel_t chol_model_22;

static void parse_args(int argc, char **argv)
{
	int i;
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-size") == 0) {
		        char *argptr;
			size = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-nblocks") == 0) {
		        char *argptr;
			nblocks = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-pin") == 0) {
			pinned = 1;
		}

		if (strcmp(argv[i], "-h") == 0) {
			printf("usage : %s [-pin] [-size size] [-nblocks nblocks]\n", argv[0]);
		}
	}
}

#endif // __DW_CHOLESKY_H__
