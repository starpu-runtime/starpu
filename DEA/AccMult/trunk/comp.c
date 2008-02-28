#include "comp.h"
#include "mult.h"
#include "jobs.h"

extern int ncudagpus;
extern int ncublasgpus;
extern unsigned ncores;

/*
 *  A x B = C
 */
void mult(matrix *A, matrix *B, matrix *C)
{
	unsigned nx,ny;
	unsigned x,y;

	nx = (B->width)/GRAIN;
	ny = (A->heigth)/GRAIN;

	for (x = 0; x < nx; x++)
	{
		for (y = 0; y < ny; y++)
		{
			job_t j = job_new();

			j->input.matA.mat = A;
			j->input.matA.xa = 0;
			j->input.matA.xb = A->width;
			j->input.matA.ya = y*GRAIN;
			j->input.matA.yb = MIN( (y+1)*GRAIN, A->heigth);

			j->input.matB.mat = B;
			j->input.matB.xa = x*GRAIN;
			j->input.matB.xb = MIN( (x+1)*GRAIN, B->width);
			j->input.matB.ya = 0;
			j->input.matB.yb = B->heigth;

			j->output.matC_sub.mat  = C;
			j->output.matC_sub.xa = x*GRAIN;
			j->output.matC_sub.xb = MIN( (x+1)*GRAIN, B->width);
			j->output.matC_sub.ya = y*GRAIN;
			j->output.matC_sub.yb = MIN( (y+1)*GRAIN, A->heigth);

			j->type = MUL;

			j->cb = NULL;

			push_task(j);
		}
	}

	/* terminate all threads */
	unsigned nworkers = 0;

#ifdef USE_CPUS
	nworkers += ncores;
#endif
#ifdef USE_CUDA
	nworkers += ncudagpus;
#endif
#ifdef USE_CUBLAS
	nworkers += ncublasgpus;
#endif

	int worker;
	for (worker = 0; worker < nworkers ; worker++) {
		job_t j = job_new();
		j->output.matC_existing = C;
		j->type = ABORT;
		push_task(j);
	}

	if (nworkers == 0) {
		fprintf(stderr, "Warning there is no worker ... \n");
	}

	return;
}
