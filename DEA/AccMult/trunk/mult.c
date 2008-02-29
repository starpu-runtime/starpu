#include "mult.h"
#include "timing.h"

/* number of actual CPU cores */

#ifdef USE_CPUS
unsigned ncores;
thread_t corethreads[NMAXCORES];
core_worker_arg coreargs[NMAXCORES]; 
#endif

#ifdef USE_CUDA
thread_t cudathreads[MAXCUDADEVS];
int cudacounters[MAXCUDADEVS];
cuda_worker_arg cudaargs[MAXCUDADEVS];
extern int ncudagpus;
#endif

#ifdef USE_CUBLAS
thread_t cublasthreads[MAXCUBLASDEVS];
int cublascounters[MAXCUBLASDEVS];
cublas_worker_arg cublasargs[MAXCUBLASDEVS];
extern int ncublasgpus;
#endif

#ifdef USE_CPUS
int corecounters[NMAXCORES];
#endif

static int current_bindid = 0;

void init_machine(void)
{
	srand(2008);

#ifdef USE_CPUS
	ncores = MIN(sysconf(_SC_NPROCESSORS_ONLN), NMAXCORES);
#endif

#ifdef USE_CUDA
	init_cuda();
#endif
	timing_init();
}

void init_workers(void) 
{
	/* initialize the queue containing the jobs */
	init_work_queue();

	/* launch one thread per CPU */
#ifdef USE_CPUS
	unsigned core;
	for (core = 0; core < ncores; core++)
	{
		corecounters[core] = 0;

		coreargs[core].bindid = (current_bindid++) % (sysconf(_SC_NPROCESSORS_ONLN));
		
		coreargs[core].coreid = core;
		coreargs[core].ready_flag = 0;

		thread_create(&corethreads[core], NULL, core_worker, &coreargs[core]);
		/* wait until the thread is actually launched ... */
		while (coreargs[core].ready_flag == 0) {}
	}
#endif

#ifdef USE_CUDA
	/* initialize CUDA with the proper number of threads */
	int cudadev;
	for (cudadev = 0; cudadev < ncudagpus; cudadev++)
	{
		cudaargs[cudadev].deviceid = cudadev;
		cudaargs[cudadev].ready_flag = 0;

		cudaargs[cudadev].bindid = (current_bindid++) % (sysconf(_SC_NPROCESSORS_ONLN));

		cudacounters[cudadev] = 0;

		thread_create(&cudathreads[cudadev], NULL, cuda_worker, (void*)&cudaargs[cudadev]);

		/* wait until the thread is actually launched ... */
		while (cudaargs[cudadev].ready_flag == 0) {}
	}
#endif


#ifdef USE_CUBLAS
	/* initialize CUBLAS with the proper number of threads */
	int cublasdev;
	for (cublasdev = 0; cublasdev < ncublasgpus; cublasdev++)
	{
		cublasargs[cublasdev].deviceid = cublasdev;
		cublasargs[cublasdev].ready_flag = 0;

		cublasargs[cublasdev].bindid = (current_bindid++) % (sysconf(_SC_NPROCESSORS_ONLN));

		cublascounters[cublasdev] = 0;

		thread_create(&cublasthreads[cublasdev], NULL, cublas_worker, (void*)&cublasargs[cublasdev]);

		/* wait until the thread is actually launched ... */
		while (cublasargs[cublasdev].ready_flag == 0) {}
	}
#endif

}

void terminate_workers(void)
{
	printf("terminate workers \n");
#ifdef USE_CPUS
	unsigned core;
	for (core = 0; core < ncores; core++)
	{
		thread_join(corethreads[core], NULL);
	}
	printf("core terminated ... \n");
#endif



#ifdef USE_CUDA
	int cudadev;
	for (cudadev = 0; cudadev < ncudagpus; cudadev++)
	{
		thread_join(cudathreads[cudadev], NULL);
	}
	printf("cuda terminated\n");
#endif

#ifdef USE_CUBLAS
	int cublasdev;
	for (cublasdev = 0; cublasdev < ncublasgpus; cublasdev++)
	{
		thread_join(cublasthreads[cublasdev], NULL);
	}
	printf("cublas terminated\n");
#endif

}

void matrix_fill_rand(matrix *m)
{
	unsigned i,j;
	for (i=0; i < m->width; i++) {
		for (j=0; j < m->heigth; j++) {
			m->data[i+j*m->width] = (float)(drand48());
			//m->data[i+j*m->width] = (float)(i==j?1.0:0.0);
		}
	}
}

void matrix_fill_zero(matrix *m)
{
	memset(m->data, 0, m->width*m->heigth*sizeof(float));
}

void alloc_matrix(matrix *m, unsigned width, unsigned heigth)
{
	m->width = width;
	m->heigth = heigth;
	m->data = malloc(width*heigth*sizeof(float));
}

void free_matrix(matrix *m)
{
	free(m->data);
}

void display_matrix(matrix *m)
{
	unsigned x,y;

	fprintf(stderr, "****************************\n");
	for (y = 0; y < m->heigth; y++) {
	for (x = 0; x < m->width; x++) {
		fprintf(stderr, "%f\t", m->data[x+y*m->width]);
	}
	fprintf(stderr, "\n");
	}
	fprintf(stderr, "****************************\n");
}

int count_tasks(void)
{
	int total = 0;
	unsigned i __attribute__ ((unused));

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

#ifdef USE_CUBLAS
	for (i = 0; i < ncublasgpus ; i++)
	{
		total += cublascounters[i];
	}
#endif

	return total;
}

void display_stats(job_descr *jd)
{
	unsigned i __attribute__ ((unused));
	int total __attribute__ ((unused));
	
	total = count_tasks();

#ifdef USE_CPUS
	printf("CORES :\n");
	for (i = 0; i < ncores ; i++)
	{
		printf("\tcore %d\t %d tasks\t%f %%\n", i, corecounters[i],
							(100.0*corecounters[i])/total);
	}
#endif

#ifdef USE_CUDA
	printf("CUDA :\n");
	for (i = 0; i < ncudagpus ; i++)
	{
		printf("\tdev %d\t %d tasks\t%f %%\n", i, cudacounters[i],
							(100.0*cudacounters[i])/total);
	}
#endif

#ifdef USE_CUBLAS
	printf("CUBLAS :\n");
	for (i = 0; i < ncublasgpus ; i++)
	{
		printf("\tblas %d\t %d tasks\t%f %%\n", i, cublascounters[i],
							(100.0*cublascounters[i])/total);
	}
#endif

	float chrono 	=  (float)(TIMING_DELAY(jd->job_submission, jd->job_finished));
	printf("Computation time : %f ms\n", chrono/1000);

#ifdef COMPARE_SEQ
	float refchrono	=  ((float)(TIMING_DELAY(jd->job_refstart, jd->job_refstop)));
	printf("Ref time : %f ms\n", refchrono/1000);
	printf("Speedup\t=\t%f\n", refchrono/chrono); 
#endif
	

}

void compare_matrix(matrix *A, matrix *B, float eps)
{
	int isdiff = 0;
	int ndiff = 0;
	int ntotal = 0;

	unsigned x,y;
	for (x = 0; x < A->width; x++) 
	{
		for (y = 0; y < A->heigth ; y++) 
		{
			if (fabs(A->data[x+y*A->width] - B->data[x+y*A->width]) > eps) {
				isdiff = 1;
				ndiff++;
				fprintf(stderr, "(%d,%d) expecting %f got %f\n", x, y,  B->data[x+y*A->width],  A->data[x+y*A->width]);
			}
			ntotal++;
		}
	}

	if (isdiff) {
		printf("Matrix are DIFFERENT (%d on %d differs ...)!\n", ndiff, ntotal);
	} else {
		printf("Matrix are IDENTICAL\n");
	}
}

#define NSAMPLE	10
int counter = NSAMPLE;

void terminate_mult(void *arg)
{
	printf("FOOOOOO\n");

	job_descr *jd = (job_descr *)arg;

	GET_TICK(jd->job_finished);

//	if (ATOMIC_ADD(&counter, -1) == 1) {
	if (--counter == 0)
	{
		printf("kill all workers ... \n");
		kill_all_workers();
	}
	
	printf("counter = %d \n", counter);

#ifdef COMPARE_SEQ
	printf("running the sequential comparision ... \n");
	GET_TICK(jd->job_refstart);
	/* only compare with 1/SEQFACTOR of the initial prob ... */
	ref_mult(jd->matA, jd->matB, jd->matD);	
	GET_TICK(jd->job_refstop);

#ifdef CHECK_OUTPUT
	compare_matrix(jd->matC, jd->matD, SIZE*0.001);
#endif
#endif

	display_stats(jd);

//	printf("matrix A :\n");
//	display_matrix(&matA);
//	printf("matrix B :\n");
//	display_matrix(&matB);
//	printf("matrix C :\n");
//	display_matrix(&matC);
//	printf("matrix D :\n");
//	display_matrix(&matD);

	free_matrix(jd->matA);
	free_matrix(jd->matB);
	free_matrix(jd->matC);
}

void mult_example(void)
{
	job_descr *jd = malloc(sizeof(job_descr));
	matrix *ABCD = malloc(4*sizeof(matrix));

	jd->matA = &ABCD[0];
	jd->matB = &ABCD[1];
	jd->matC = &ABCD[2];
	jd->matD = &ABCD[3];

	/* for simplicity, use SIZE = power of 2 ! */
	alloc_matrix(jd->matA, SIZE, SIZE);
	alloc_matrix(jd->matB, SIZE, SIZE);

	alloc_matrix(jd->matC, SIZE, SIZE);
	alloc_matrix(jd->matD, SIZE, SIZE);

	matrix_fill_rand(jd->matA);
	matrix_fill_rand(jd->matB);

	matrix_fill_zero(jd->matC);
	matrix_fill_zero(jd->matD);

	GET_TICK(jd->job_submission);

	mult(jd->matA, jd->matB, jd->matC, terminate_mult, jd);
}

int main(int argc __attribute__ ((unused)), char **argv __attribute__ ((unused)) )
{
#ifdef USE_MARCEL
	marcel_init(&argc, argv);
#endif

	init_machine();
	init_workers();

	int i;
	for (i = 0; i < NSAMPLE; i++)
		mult_example();

	terminate_workers();
	return 0;
}
