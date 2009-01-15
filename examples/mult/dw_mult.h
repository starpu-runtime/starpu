#ifndef __MULT_H__
#define __MULT_H__

#include <semaphore.h>
#include <core/jobs.h>
#include <core/workers.h>
#include <core/dependencies/tags.h>
#include <common/timing.h>
#include <common/util.h>
#include <common/malloc.h>
#include <string.h>
#include <math.h>
#include <sys/types.h>
#include <sys/time.h>
#include <pthread.h>
#include <signal.h>
#include <common/blas.h>
#include <common/timing.h>

#include <datawizard/datawizard.h>

#include <task-models/blas_model.h>

#include <common/fxt.h>

#ifdef USE_CUDA
#include <cuda.h>
#endif

#define MAXSLICESX	32
#define MAXSLICESY	32
#define MAXSLICESZ	32

#define BLAS3_FLOP(n1,n2,n3)	\
	(2*((uint64_t)n1)*((uint64_t)n2)*((uint64_t)n3))

#define BLAS3_LS(n1,n2,n3)    \
	((2*(n1)*(n3) + (n1)*(n2) + (n2)*(n3))*sizeof(float))

extern struct perfmodel_t sgemm_model;

struct block_conf {
	uint32_t m;
	uint32_t n;
	uint32_t k;
	uint32_t pad;
};

#define NITER	100

unsigned niter = NITER;
unsigned nslicesx = 4;
unsigned nslicesy = 4;
unsigned nslicesz = 4;
unsigned xdim = 256;
unsigned ydim = 256;
unsigned zdim = 64;
unsigned norandom = 0;
unsigned pin = 0;

/* to compute MFlop/s */
uint64_t flop_cublas = 0;
uint64_t flop_atlas = 0;

/* to compute MB/s (load/store) */
uint64_t ls_cublas = 0;
uint64_t ls_atlas = 0;


struct timeval start;
struct timeval end;
sem_t sem;

static int jobcounter __attribute__ ((unused));
static struct block_conf conf __attribute__ ((aligned (128)));

#define BLOCKSIZEX	(xdim / nslicesx)
#define BLOCKSIZEY	(ydim / nslicesy)
#define BLOCKSIZEZ	(zdim / nslicesz)

static void parse_args(int argc, char **argv)
{
	int i;
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-nblocks") == 0) {
			char *argptr;
			nslicesx = strtol(argv[++i], &argptr, 10);
			nslicesy = nslicesx;
			nslicesz = nslicesx;
		}

		if (strcmp(argv[i], "-nblocksx") == 0) {
			char *argptr;
			nslicesx = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-nblocksy") == 0) {
			char *argptr;
			nslicesy = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-nblocksz") == 0) {
			char *argptr;
			nslicesz = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-x") == 0) {
			char *argptr;
			xdim = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-y") == 0) {
			char *argptr;
			ydim = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-z") == 0) {
			char *argptr;
			zdim = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-iter") == 0) {
			char *argptr;
			niter = strtol(argv[++i], &argptr, 10);
		}


		if (strcmp(argv[i], "-no-random") == 0) {
			norandom = 1;
		}

		if (strcmp(argv[i], "-pin") == 0) {
			pin = 1;
		}
	}

	assert(nslicesx <= MAXSLICESX); 
	assert(nslicesy <= MAXSLICESY); 
	assert(nslicesz <= MAXSLICESZ); 
}

static void display_memory_consumption(void)
{
//	fprintf(stderr, "Memory consumption : A -> %ld KB\n", (MAXSLICESY*MAXSLICESZ*sizeof(float *))/(1024));
//	fprintf(stderr, "Memory consumption : B -> %ld KB\n", (MAXSLICESZ*MAXSLICESX*sizeof(float *))/(1024));
//	fprintf(stderr, "Memory consumption : C -> %ld KB\n", (MAXSLICESY*MAXSLICESX*sizeof(float *))/(1024));
//	fprintf(stderr, "Memory consumption : A_state -> %ld KB\n", (MAXSLICESY*MAXSLICESZ*sizeof(data_state)/(1024)));
//	fprintf(stderr, "Memory consumption : B_state -> %ld KB\n", (MAXSLICESZ*MAXSLICESX*sizeof(data_state)/(1024)));
//	fprintf(stderr, "Memory consumption : C_state -> %ld KB\n", (MAXSLICESY*MAXSLICESX*sizeof(data_state)/(1024)));
//	fprintf(stderr, "Memory consumption : data A -> %ld MB\n", (ydim*zdim*sizeof(float)/(1024*1024)));
//	fprintf(stderr, "Memory consumption : data B -> %ld MB\n", (zdim*xdim*sizeof(float)/(1024*1024)));
//	fprintf(stderr, "Memory consumption : data C -> %ld MB\n", (ydim*xdim*sizeof(float)/(1024*1024)));

	fprintf(stderr, "Total memory : %ld MB\n", (MAXSLICESY*MAXSLICESZ*sizeof(float *) + MAXSLICESZ*MAXSLICESX*sizeof(float *) + MAXSLICESY*MAXSLICESX*sizeof(float *) + MAXSLICESY*MAXSLICESZ*sizeof(data_state) + MAXSLICESZ*MAXSLICESX*sizeof(data_state) + MAXSLICESY*MAXSLICESX*sizeof(data_state) + ydim*zdim*sizeof(float) +  zdim*xdim*sizeof(float) +  ydim*xdim*sizeof(float))/(1024*1024) );
}


#endif // __MULT_H__
