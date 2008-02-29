#include "comp.h"
#include "mult.h"
#include "jobs.h"
#include "timing.h"

extern int ncudagpus;
extern int ncublasgpus;
extern unsigned ncores;

void kill_all_workers(void)
{
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

	unsigned worker;
	for (worker = 0; worker < nworkers ; worker++) {
		job_t j = job_new();
	//	j->output.matC_existing = C;
		j->type = ABORT;
		j->where = ANY;
		push_task(j);
	}

	if (nworkers == 0) {
		fprintf(stderr, "Warning there is no worker ... \n");
	}

}

void cleanup_problem(void *cbarg)
{
	job_descr *jd = (job_descr *)cbarg;

#ifdef USE_CUBLAS
	job_t j = job_new();

	j->where = CUBLAS;

	j->type = CLEAN;

	j->cb = jd->f;//kill_all_workers;
	j->argcb = jd->argf;//cbarg;

	push_task(j);
#else
	//kill_all_workers();
	jd->f(jd->argf);	
#endif
}

void mult_callback(void *cbarg)
{
	job_descr *jd = (job_descr *)cbarg;

	int cnt = ATOMIC_ADD(&jd->counter, -1);
	
	if (cnt == 0) { 
		printf("callback %d !\n", cnt);
		printf("DONE !!\n");
		cleanup_problem(cbarg);
	}

}

//void step1(matrix *A, matrix *B, matrix *C)
//{
//#ifdef USE_CUBLAS
//	precondition_cublas(A, B, C);
//#elif USE_CUDA
//	precondition_cuda(A, B, C);
//#else // no need to precondition ... 
//
//#endif
//}

/*
 *  A x B = C
 *
 * There are 3 steps : 
 *   - if CUDA / CUBLAS is used precondition the matrices
 *   - launch the tasks on ANY cores / gpu ...
 *   - terminate 
 *
 */

void step2(void *arg)
{

	/* now the matrices are preconditionned : create the actual jobs */

	job_descr * jd = (job_descr *)arg;

	matrix *A = jd->matA;
	matrix *B = jd->matB;
	matrix *C = jd->matC;

	GET_TICK(jd->job_preconditionned);

        unsigned nx,ny;
        unsigned x,y;

        nx = (B->width)/GRAIN;
        ny = (A->heigth)/GRAIN;


	jd->counter = nx*ny - 1;

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

			j->where = ANY;

			j->cb = mult_callback;
			j->argcb = arg;

			push_task(j);
		}
	}
}

void mult(matrix *A, matrix *B, matrix *C, callback f, void *argf)
{
	job_descr *jd = malloc(sizeof(job_descr));

	jd->matA = A; 
	jd->matB = B; 
	jd->matC = C; 

	jd->f = f;
	jd->argf = argf;

	GET_TICK(jd->job_submission);

#ifdef USE_CUBLAS
	job_t j = job_new();

	j->type = PRECOND;
	j->where = CUBLAS;

	j->cb = step2;
	j->argcb = jd;

	push_task(j);
#elif USE_CUDA
	job_t j = job_new();

	j->type = PRECOND;
	j->where = CUDA;

	j->cb = step2;
	j->argcb = jd;

	push_task(j);
#else
	/* directly partition work */
	step2(jd);
#endif
}
