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

tick_t start, stop;
tick_t refstart, refstop;


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

void init_workers(matrix *A, matrix *B, matrix *C)
{
	/* initialize the queue containing the jobs */
	init_work_queue();

	/* launch one thread per CPU */
#ifdef USE_CPUS
	unsigned core;
	for (core = 0; core < ncores; core++)
	{
		corecounters[core] = 0;
		
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
		cudaargs[cudadev].A = A;
		cudaargs[cudadev].B = B;
		cudaargs[cudadev].C = C;

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
		cublasargs[cublasdev].A = A;
		cublasargs[cublasdev].B = B;
		cublasargs[cublasdev].C = C;

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

void display_stats(void)
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

	float chrono 	=  (float)(TIMING_DELAY(start, stop));
	printf("Computation time : %f ms\n", chrono/1000);

#ifdef COMPARE_SEQ
	float refchrono	=  ((float)(TIMING_DELAY(refstart, refstop)));
	printf("Ref time : %f ms\n", refchrono/1000);
	printf("Speedup\t=\t%f\n", refchrono/chrono); 
#endif
	

}

void compare_matrix(matrix *A, matrix *B, float eps)
{
	int isdiff = 0;
	int ndiff = 0;
	int ntotal = 0;

	int x,y;
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

int main(int argc, char **argv)
{
#ifdef USE_MARCEL
	marcel_init(&argc, argv);
#endif

	matrix matA;
	matrix matB;
	matrix matC;
	matrix matD;

	/* for simplicity, use SIZE = power of 2 ! */
	alloc_matrix(&matA, SIZE, SIZE);
	alloc_matrix(&matB, SIZE, SIZE);

	alloc_matrix(&matC, SIZE, SIZE);
	alloc_matrix(&matD, SIZE, SIZE);

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
	ref_mult(&matA, &matB, &matD);	
	GET_TICK(refstop);

#ifdef CHECK_OUTPUT
	compare_matrix(&matC, &matD, SIZE*0.001);
#endif
#endif

	display_stats();

//	printf("matrix A :\n");
//	display_matrix(&matA);
//	printf("matrix B :\n");
//	display_matrix(&matB);
//	printf("matrix C :\n");
//	display_matrix(&matC);
//	printf("matrix D :\n");
//	display_matrix(&matD);

	free_matrix(&matA);
	free_matrix(&matB);
	free_matrix(&matC);

	return 0;
}
