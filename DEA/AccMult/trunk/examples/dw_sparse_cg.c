/*
 * Conjugate gradients for Sparse matrices
 */

#include "dw_sparse_cg.h"

tick_t start,end;

#ifdef USE_CUDA
/* CUDA spmv codelet */
static struct cuda_module_s cuda_module;
static struct cuda_function_s cuda_function;
static cuda_codelet_t cuda_codelet;

void initialize_cuda(void)
{
	char *module_path = 
		"/home/gonnet/DEA/AccMult/examples/cuda/spmv_cuda.cubin";
	char *function_symbol = "spmv_kernel_3";

	init_cuda_module(&cuda_module, module_path);
	init_cuda_function(&cuda_function, &cuda_module, function_symbol);

	cuda_codelet.func = &cuda_function;
	cuda_codelet.stack = NULL;
	cuda_codelet.stack_size = 0; 

	cuda_codelet.gridx = grids;
	cuda_codelet.gridy = 1;

	cuda_codelet.blockx = blocks;
	cuda_codelet.blocky = 1;

	cuda_codelet.shmemsize = 128;
}




#endif // USE_CUDA

void init_problem(void)
{
	/* create the sparse input matrix */
	float *nzval;
	float *vecb;
	float *vecx;
	uint32_t nnz;
	uint32_t nrow;
	uint32_t *colind;
	uint32_t *rowptr;

	create_data(&nzval, &vecb, &vecx, &nnz, &nrow, &colind, &rowptr);

	GET_TICK(start);
	conjugate_gradient(nzval, vecb, vecx, nnz, nrow, colind, rowptr);
	GET_TICK(end);
}

/*
 *	cg initialization phase 
 */

void init_cg(struct cg_problem *problem) 
{
	problem->i = 0;

	/* r = b  - A x */
	job_t job1 = create_job(1UL);
	job1->where = CORE;
	job1->cl->core_func = core_codelet_func_1;
	job1->nbuffers = 4;
		job1->buffers[0].state = problem->ds_matrixA;
		job1->buffers[0].mode = R;
		job1->buffers[1].state = problem->ds_vecx;
		job1->buffers[1].mode = R;
		job1->buffers[2].state = problem->ds_vecr;
		job1->buffers[2].mode = W;
		job1->buffers[3].state = problem->ds_vecb;
		job1->buffers[3].mode = R;

	/* d = r */
	job_t job2 = create_job(2UL);
	job2->where = CORE;
	job2->cl->core_func = core_codelet_func_2;
	job2->nbuffers = 2;
		job2->buffers[0].state = problem->ds_vecd;
		job2->buffers[0].mode = W;
		job2->buffers[1].state = problem->ds_vecr;
		job2->buffers[1].mode = R;
	
	tag_declare_deps(2UL, 1, 1UL);

	/* delta_new = trans(r) r */
	job_t job3 = create_job(3UL);
	job3->where = CUBLAS;
#if defined (USE_CUBLAS) || defined (USE_CUDA)
	job3->cl->cublas_func = cublas_codelet_func_3;
#endif
	job3->cl->core_func = core_codelet_func_3;
	job3->cl->cl_arg = problem;
	job3->nbuffers = 1;
		job3->buffers[0].state = problem->ds_vecr;
		job3->buffers[0].mode = R;

	job3->cb = iteration_cg;
	job3->argcb = problem;
	
	/* XXX 3 should only depend on 1 ... */
	tag_declare_deps(3UL, 1, 2UL);

	/* launch the computation now */
	push_task(job1);
}

/*
 *	the inner iteration of the cg algorithm 
 *		the codelet code launcher is its own callback !
 */

void launch_new_cg_iteration(struct cg_problem *problem)
{
	unsigned iter = problem->i;

	unsigned long long maskiter = (iter*1024);

	/* q = A d */
	job_t job4 = create_job(maskiter | 4UL);
	job4->where = CORE;
	job4->cl->core_func = core_codelet_func_4;
	job4->nbuffers = 3;
		job4->buffers[0].state = problem->ds_matrixA;
		job4->buffers[0].mode = R;
		job4->buffers[1].state = problem->ds_vecd;
		job4->buffers[1].mode = R;
		job4->buffers[2].state = problem->ds_vecq;
		job4->buffers[2].mode = W;

	/* alpha = delta_new / ( trans(d) q )*/
	job_t job5 = create_job(maskiter | 5UL);
	job5->where = CUBLAS|CORE;
#if defined (USE_CUBLAS) || defined (USE_CUDA) 
	job5->cl->cublas_func = cublas_codelet_func_5;
#endif
	job5->cl->core_func = core_codelet_func_5;
	job5->cl->cl_arg = problem;
	job5->nbuffers = 2;
		job5->buffers[0].state = problem->ds_vecd;
		job5->buffers[0].mode = R;
		job5->buffers[1].state = problem->ds_vecq;
		job5->buffers[1].mode = R;

	tag_declare_deps(maskiter | 5UL, 1, maskiter | 4UL);

	/* x = x + alpha d */
	job_t job6 = create_job(maskiter | 6UL);
	job6->where = CUBLAS|CORE;
#if defined (USE_CUBLAS) || defined (USE_CUDA)
	job6->cl->cublas_func = cublas_codelet_func_6;
#endif
	job6->cl->core_func = core_codelet_func_6;
	job6->cl->cl_arg = problem;
	job6->nbuffers = 2;
		job6->buffers[0].state = problem->ds_vecx;
		job6->buffers[0].mode = RW;
		job6->buffers[1].state = problem->ds_vecd;
		job6->buffers[1].mode = R;

	tag_declare_deps(maskiter | 6UL, 1, maskiter | 5UL);

	/* r = r - alpha q */
	job_t job7 = create_job(maskiter | 7UL);
	job7->where = CUBLAS|CORE;
#if defined (USE_CUBLAS) || defined (USE_CUDA)
	job7->cl->cublas_func = cublas_codelet_func_7;
#endif
	job7->cl->core_func = core_codelet_func_7;
	job7->cl->cl_arg = problem;
	job7->nbuffers = 2;
		job7->buffers[0].state = problem->ds_vecr;
		job7->buffers[0].mode = RW;
		job7->buffers[1].state = problem->ds_vecq;
		job7->buffers[1].mode = R;

	tag_declare_deps(maskiter | 7UL, 1, maskiter | 6UL);

	/* update delta_* and compute beta */
	job_t job8 = create_job(maskiter | 8UL);
	job8->where = CUBLAS|CORE;
#if defined (USE_CUBLAS) || defined (USE_CUDA)
	job8->cl->cublas_func = cublas_codelet_func_8;
#endif
	job8->cl->core_func = core_codelet_func_8;
	job8->cl->cl_arg = problem;
	job8->nbuffers = 1;
		job8->buffers[0].state = problem->ds_vecr;
		job8->buffers[0].mode = R;

	tag_declare_deps(maskiter | 8UL, 1, maskiter | 7UL);

	/* d = r + beta d */
	job_t job9 = create_job(maskiter | 9UL);
	job9->where = CUBLAS;
#if defined (USE_CUBLAS) || defined (USE_CUDA)
	job9->cl->cublas_func = cublas_codelet_func_9;
#endif
	job9->cl->core_func = core_codelet_func_9;
	job9->cl->cl_arg = problem;
	job9->nbuffers = 2;
		job9->buffers[0].state = problem->ds_vecd;
		job9->buffers[0].mode = RW;
		job9->buffers[1].state = problem->ds_vecr;
		job9->buffers[1].mode = R;

	tag_declare_deps(maskiter | 9UL, 1, maskiter | 8UL);

	job9->cb = iteration_cg;
	job9->argcb = problem;
	
	/* launch the computation now */
	push_task(job4);
}

void iteration_cg(void *problem)
{
	struct cg_problem *pb = problem;

	printf("i : %d\n\tdelta_new %2.5f\n", pb->i, pb->epsilon);

	if ((pb->i++ < MAXITER) && 
		(pb->delta_new > pb->epsilon) )
	{
		/* we did not reach the stop condition yet */
		launch_new_cg_iteration(problem);
	}
	else {
		/* we may stop */
		printf("We are done ... after %d iterations \n", pb->i - 1);
		sem_post(pb->sem);
	}
}

/*
 *	initializing the problem 
 */

void conjugate_gradient(float *nzvalA, float *vecb, float *vecx, uint32_t nnz,
			unsigned nrow, uint32_t *colind, uint32_t *rowptr)
{
	/* first declare all the data structures to the runtime */

	struct data_state_t ds_matrixA;
	struct data_state_t ds_vecx, ds_vecb;
	struct data_state_t ds_vecr, ds_vecd, ds_vecq; 

	/* first the user-allocated data */
	monitor_csr_data(&ds_matrixA, 0, nnz, nrow, 
			(uintptr_t)nzvalA, colind, rowptr, 0, sizeof(float));
	monitor_blas_data(&ds_vecx, 0, (uintptr_t)vecx,
			nrow, nrow, 1, sizeof(float));
	monitor_blas_data(&ds_vecb, 0, (uintptr_t)vecb,
			nrow, nrow, 1, sizeof(float));

	/* then allocate the algorithm intern data */
	float *ptr_vecr, *ptr_vecd, *ptr_vecq;

	unsigned i;
	ptr_vecr = malloc(nrow*sizeof(float));
	ptr_vecd = malloc(nrow*sizeof(float));
	ptr_vecq = malloc(nrow*sizeof(float));

	for (i = 0; i < nrow; i++)
	{
		ptr_vecr[i] = 0.0f;
		ptr_vecd[i] = 0.0f;
		ptr_vecq[i] = 0.0f;
	}

	/* and declare them as well */
	monitor_blas_data(&ds_vecr, 0, (uintptr_t)ptr_vecr, 
			nrow, nrow, 1, sizeof(float));
	monitor_blas_data(&ds_vecd, 0, (uintptr_t)ptr_vecd, 
			nrow, nrow, 1, sizeof(float));
	monitor_blas_data(&ds_vecq, 0, (uintptr_t)ptr_vecq, 
			nrow, nrow, 1, sizeof(float));

	/* we now have the complete problem */
	struct cg_problem problem;

	problem.ds_matrixA = &ds_matrixA;
	problem.ds_vecx    = &ds_vecx;
	problem.ds_vecb    = &ds_vecb;
	problem.ds_vecr    = &ds_vecr;
	problem.ds_vecd    = &ds_vecd;
	problem.ds_vecq    = &ds_vecq;

	problem.epsilon = EPSILON;

	/* we need a semaphore to synchronize with callbacks */
	sem_t sem;
	sem_init(&sem, 0, 0U);
	problem.sem  = &sem;

	init_cg(&problem);

	sem_wait(&sem);
	sem_destroy(&sem);

	print_results(vecx, nrow);
}

int main(__attribute__ ((unused)) int argc,
	__attribute__ ((unused)) char **argv)
{
	parse_args(argc, argv);

	timing_init();

	/* start the runtime */
	init_machine();
	init_workers();


#ifdef USE_CUDA
	initialize_cuda();
#endif

	init_problem();

	double timing = timing_delay(&start, &end);
	fprintf(stderr, "Computation took (in ms)\n");
	printf("%2.2f\n", timing/1000);


	return 0;
}
