#include <pthread.h>
#include "mult.h"
#include "comp.h"
#include "timing.h"

#define NMAXCORES	0
//#define COMPARE_SEQ	1

/* number of actual CPU cores */
unsigned ncores;
pthread_t corethreads[NMAXCORES];

#ifdef USE_CUDA
pthread_t cudathreads[MAXCUDADEVS];
int cudacounters[MAXCUDADEVS];
cuda_worker_arg cudaargs[MAXCUDADEVS];
#endif

int corecounters[NMAXCORES];

tick_t start, stop;
tick_t refstart, refstop;

extern int ncudagpus;

char *execpath;


void execute_job_on_core(job_t j)
{
	switch (j->type) {
		case MUL:
			dummy_mult(&j->input.matA, &j->input.matB, &j->output.matC_sub);
			break;
		case ABORT:
			pthread_exit(NULL);
			break;
		default:
			break;
	}
}

void *core_worker(void *arg)
{
	int core = (uintptr_t)arg;

	job_t j;

	do {
		j = pop_task();
		if (j == NULL) continue;

		execute_job_on_core(j);

		if (j->cb)
			j->cb(j->argcb);

		corecounters[core]++;
	} while(1);

	return NULL;
}

void init_machine()
{
	srand(2008);

	ncores = MIN(sysconf(_SC_NPROCESSORS_ONLN), NMAXCORES);

#ifdef USE_CUDA
	init_cuda();
#endif
	timing_init();
}

void init_workers(matrix *A, matrix *B, matrix *C)
{
	/* initialize the queue containing the jobs */
	init_work_queue();

	/* launch one thread per CPU */
#ifdef USE_CPUS
	int core;
	for (core = 0; core < ncores; core++)
	{
		corecounters[core] = 0;

		pthread_create(&corethreads[core], NULL, core_worker, (void *)core);
	}
#endif

#ifdef USE_CUDA
	/* initialize CUDA with the proper number of threads */
	int cudadev;
	for (cudadev = 0; cudadev < ncudagpus; cudadev++)
	{
		cudaargs[cudadev].deviceid = cudadev;
		cudaargs[cudadev].A = A;
		cudaargs[cudadev].B = B;
		cudaargs[cudadev].C = C;

		cudacounters[cudadev] = 0;

		pthread_create(&cudathreads[cudadev], NULL, cuda_worker, (void*)&cudaargs[cudadev]);
	}
#endif
}

void terminate_workers()
{
#ifdef USE_CPUS
	int core;
	for (core = 0; core < ncores; core++)
	{
		pthread_join(corethreads[core], NULL);
	}
#endif

#ifdef USE_CUDA
	/* initialize CUDA with the proper number of threads */
	int cudadev;
	for (cudadev = 0; cudadev < ncudagpus; cudadev++)
	{
		pthread_join(cudathreads[cudadev], NULL);
	}
#endif

}

void matrix_fill_rand(matrix *m)
{
	int i,j;
	for (i=0; i < m->width; i++) {
		for (j=0; j < m->heigth; j++) {
			/* we don't want to deal with integer overflow ... */
			m->data[i+j*m->width] = (uint32_t)(rand()%2);
		}
	}
}

void matrix_fill_zero(matrix *m)
{
	memset(m->data, 0, m->width*m->heigth*sizeof(uint32_t));
}

void alloc_matrix(matrix *m, unsigned width, unsigned heigth)
{
	m->width = width;
	m->heigth = heigth;
	m->data = malloc(width*heigth*sizeof(uint32_t));
}

void free_matrix(matrix *m)
{
	free(m->data);
}

void display_stats(void)
{
	int i;
	int total = 0;
#ifdef USE_CPUS
	for (i = 0; i < ncores ; i++)
	{
		total += corecounters[i];
	}
#endif

#ifdef USE_CUDA
	for (i = 0; i < ncudagpus ; i++)
	{
		total += cudacounters[i];
	}
#endif


#ifdef USE_CPUS
	printf("CORES :\n");
	for (i = 0; i < ncores ; i++)
	{
		printf("\tcore %d\t %d tasks\t%f %%\n", i, corecounters[i], (100.0*corecounters[i])/total);
	}
#endif

#ifdef USE_CUDA
	printf("CUDA :\n");
	for (i = 0; i < ncudagpus ; i++)
	{
		printf("\tdev %d\t %d tasks\t%f %%\n", i, cudacounters[i], (100.0*cudacounters[i])/total);
	}
#endif

	float chrono 	=  (float)(TIMING_DELAY(start, stop));
	printf("Computation time : %f ms\n", chrono/1000);

#ifdef COMPARE_SEQ
	float refchrono	=  SEQFACTOR*((float)(TIMING_DELAY(refstart, refstop)));
	printf("Ref time : %f ms\n", refchrono/1000);
	printf("Speedup\t=\t%f\n", refchrono/chrono); 
#endif
	

}

void compare_matrix(matrix *A, matrix *B, int seqfactor)
{
	int isdiff = 0;
	int ndiff = 0;
	int ntotal = 0;

	int x,y;
	for (x = 0; x < A->width; x++) 
	{
		for (y = 0; y < A->heigth/seqfactor ; y++) 
		{
			if (A->data[x+y*A->width] != B->data[x+y*A->width]) {
				isdiff = 1;
				ndiff++;
				printf("(%d,%d) expecting %d got %d\n", x, y,  B->data[x+y*A->width],  A->data[x+y*A->width]);
			}
			ntotal++;
		}
	}

	if (isdiff) {
		printf("Matrix are DIFFERENT (%d on %d differs ...)!\n", ndiff, ntotal);
	} else {
		printf("Matrix are IDENTICAL (warning : only checked %d lines out of %d)\n", A->heigth/seqfactor , A->heigth);
	}
}

int main(int argc, char **argv)
{
	execpath = argv[0];

	matrix matA;
	matrix matB;
	matrix matC;
	matrix matD;

	/* for simplicity, use N = power of 2 ! */
	alloc_matrix(&matA, N, N);
	alloc_matrix(&matB, N, N);

	alloc_matrix(&matC, N, N);
	alloc_matrix(&matD, N, N);

	matrix_fill_rand(&matA);
	matrix_fill_rand(&matB);

	matrix_fill_zero(&matC);
	matrix_fill_zero(&matD);

	init_machine();
	init_workers(&matA, &matB, &matC);

	GET_TICK(start);
	mult(&matA, &matB, &matC);

	terminate_workers();
	GET_TICK(stop);


#ifdef COMPARE_SEQ
	printf("running the sequential comparision ... \n");
	GET_TICK(refstart);
	/* only compare with 1/SEQFACTOR of the initial prob ... */
	ref_mult(&matA, &matB, &matD, SEQFACTOR);	
	GET_TICK(refstop);

	/* only compare the 1/SEQFACTOR part ... */
	compare_matrix(&matC, &matD, SEQFACTOR);
#endif

	display_stats();

	free_matrix(&matA);
	free_matrix(&matB);
	free_matrix(&matC);

	return 0;
}
