/*
 * Conjugate gradients for Sparse matrices
 */

#include "dw_spmv.h"

tick_t start,end;

unsigned nblocks = 1;
unsigned remainingjobs = -1;

/* First a Matrix-Vector product (SpMV) */

unsigned blocks = 512;
unsigned grids  = 8;

#ifdef USE_CUDA
/* CUDA spmv codelet */
static struct cuda_module_s cuda_module;
static struct cuda_function_s cuda_function;
static cuda_codelet_t cuda_spmv;

void initialize_cuda(void)
{
	char module_path[1024];
	sprintf(module_path,
		"%s/examples/cuda/spmv_cuda.cubin", STARPUDIR);
	char *function_symbol = "spmv_kernel_3";

	init_cuda_module(&cuda_module, module_path);
	init_cuda_function(&cuda_function, &cuda_module, function_symbol);

	cuda_spmv.func = &cuda_function;
	cuda_spmv.stack = NULL;
	cuda_spmv.stack_size = 0; 

	cuda_spmv.gridx = grids;
	cuda_spmv.gridy = 1;

	cuda_spmv.blockx = blocks;
	cuda_spmv.blocky = 1;

	cuda_spmv.shmemsize = 60;
}




#endif // USE_CUDA


sem_t sem;
uint32_t size = 4194304;

data_state sparse_matrix;
data_state vector_in, vector_out;

float *sparse_matrix_nzval;
uint32_t *sparse_matrix_colind;
uint32_t *sparse_matrix_rowptr;

float *vector_in_ptr;
float *vector_out_ptr;

unsigned usecpu = 0;


void parse_args(int argc, char **argv)
{
	int i;
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-size") == 0) {
			char *argptr;
			size = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-block") == 0) {
			char *argptr;
			blocks = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-grid") == 0) {
			char *argptr;
			grids = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-nblocks") == 0) {
			char *argptr;
			nblocks = strtol(argv[++i], &argptr, 10);
		}


		if (strcmp(argv[i], "-cpu") == 0) {
			usecpu = 1;
		}
	}
}

void core_spmv(data_interface_t *descr, __attribute__((unused))  void *arg)
{
	float *nzval = (float *)descr[0].csr.nzval;
	uint32_t *colind = descr[0].csr.colind;
	uint32_t *rowptr = descr[0].csr.rowptr;

	float *vecin = (float *)descr[1].vector.ptr;
	float *vecout = (float *)descr[2].vector.ptr;

	uint32_t firstelem = descr[0].csr.firstentry;

	uint32_t nnz;
	uint32_t nrow;

	nnz = descr[0].csr.nnz;
	nrow = descr[0].csr.nrow;

	//STARPU_ASSERT(nrow == descr[1].vector.nx);
	STARPU_ASSERT(nrow == descr[2].vector.nx);

	unsigned row;
	for (row = 0; row < nrow; row++)
	{
		float tmp = 0.0f;
		unsigned index;

		unsigned firstindex = rowptr[row] - firstelem;
		unsigned lastindex = rowptr[row+1] - firstelem;

		for (index = firstindex; index < lastindex; index++)
		{
			unsigned col;

			col = colind[index];
			tmp += nzval[index]*vecin[col];
		}

		vecout[row] = tmp;
	}

}

void create_data(void)
{
	/* we need a sparse symetric (definite positive ?) matrix and a "dense" vector */
	
	/* example of 3-band matrix */
	float *nzval;
	uint32_t nnz;
	uint32_t *colind;
	uint32_t *rowptr;

	nnz = 3*size-2;

	nzval = malloc(nnz*sizeof(float));
	colind = malloc(nnz*sizeof(uint32_t));
	rowptr = malloc((size+1)*sizeof(uint32_t));

	assert(nzval);
	assert(colind);
	assert(rowptr);

	/* fill the matrix */
	unsigned row;
	unsigned pos = 0;
	for (row = 0; row < size; row++)
	{
		rowptr[row] = pos;

		if (row > 0) {
			nzval[pos] = 1.0f;
			colind[pos] = row-1;
			pos++;
		}
		
		nzval[pos] = 5.0f;
		colind[pos] = row;
		pos++;

		if (row < size - 1) {
			nzval[pos] = 1.0f;
			colind[pos] = row+1;
			pos++;
		}
	}

	STARPU_ASSERT(pos == nnz);

	rowptr[size] = nnz;
	
	monitor_csr_data(&sparse_matrix, 0, nnz, size, (uintptr_t)nzval, colind, rowptr, 0, sizeof(float));

	sparse_matrix_nzval = nzval;
	sparse_matrix_colind = colind;
	sparse_matrix_rowptr = rowptr;

	/* initiate the 2 vectors */
	float *invec, *outvec;
	invec = malloc(size*sizeof(float));
	assert(invec);

	outvec = malloc(size*sizeof(float));
	assert(outvec);

	/* fill those */
	unsigned ind;
	for (ind = 0; ind < size; ind++)
	{
		invec[ind] = 2.0f;
		outvec[ind] = 0.0f;
	}

	monitor_vector_data(&vector_in, 0, (uintptr_t)invec, size, sizeof(float));
	monitor_vector_data(&vector_out, 0, (uintptr_t)outvec, size, sizeof(float));

	vector_in_ptr = invec;
	vector_out_ptr = outvec;

}

void init_problem_callback(void *arg)
{
	unsigned *remaining = arg;


	unsigned val = ATOMIC_ADD(remaining, -1);

	printf("callback %d remaining \n", val);
	if ( val == 0 )
	{
		printf("DONE ...\n");
		GET_TICK(end);

		unpartition_data(&sparse_matrix, 0);
		unpartition_data(&vector_out, 0);

		sem_post(&sem);
	}
}


void call_spmv_codelet_filters(void)
{

	remainingjobs = nblocks;

	codelet *cl = malloc(sizeof(codelet));

	/* partition the data along a block distribution */
	filter csr_f, vector_f;
	csr_f.filter_func    = vertical_block_filter_func_csr;
	csr_f.filter_arg     = nblocks;
	vector_f.filter_func = block_filter_func_vector;
	vector_f.filter_arg  = nblocks;

	partition_data(&sparse_matrix, &csr_f);
	partition_data(&vector_out, &vector_f);

	cl->cl_arg = NULL;
	cl->core_func =  core_spmv;
#ifdef USE_CUDA
	cl->cuda_func = &cuda_spmv;
#endif


	GET_TICK(start);
	unsigned part;
	for (part = 0; part < nblocks; part++)
	{
		job_t job;
		job = job_create();
//#ifdef USE_CUDA
//		job->where = usecpu?CORE:CUDA;
//#else
//		job->where = CORE;
//#endif
		job->where = CORE|CUDA;
		job->cb = init_problem_callback;
		job->argcb = &remainingjobs;
		job->cl = cl;
	
		job->nbuffers = 3;
		job->buffers[0].state = get_sub_data(&sparse_matrix, 1, part);
		job->buffers[0].mode  = R;
		job->buffers[1].state = &vector_in;
		job->buffers[1].mode = R;
		job->buffers[2].state = get_sub_data(&vector_out, 1, part);
		job->buffers[2].mode = W;
	
		push_task(job);
	}
}



void call_spmv_codelet(void)
{

	remainingjobs = 1;

	job_t job;
	codelet *cl = malloc(sizeof(codelet));

	cl->cl_arg = NULL;
	cl->core_func =  core_spmv;
#ifdef USE_CUDA
	cl->cuda_func = &cuda_spmv;
#endif

	job = job_create();
#ifdef USE_CUDA
	job->where = usecpu?CORE:CUDA;
#else
	job->where = CORE;
#endif
	job->cb = init_problem_callback;
	job->argcb = &remainingjobs;
	job->cl = cl;

	job->nbuffers = 3;
	job->buffers[0].state = &sparse_matrix;
	job->buffers[0].mode  = R;
	job->buffers[1].state = &vector_in;
	job->buffers[1].mode = R;
	job->buffers[2].state = &vector_out;
	job->buffers[2].mode = W;




	GET_TICK(start);
}

void init_problem(void)
{
	/* create the sparse input matrix */
	create_data();

	/* create a new codelet that will perform a SpMV on it */
//	call_spmv_codelet();
	call_spmv_codelet_filters();
}

void print_results(void)
{
	unsigned row;

	for (row = 0; row < MIN(size, 16); row++)
	{
		printf("%2.2f\t%2.2f\n", vector_in_ptr[row], vector_out_ptr[row]);
	}
}

int main(__attribute__ ((unused)) int argc,
	__attribute__ ((unused)) char **argv)
{
	parse_args(argc, argv);


	timing_init();

	/* start the runtime */
	init_machine();

	sem_init(&sem, 0, 0U);

#ifdef USE_CUDA
	initialize_cuda();
#endif

	init_problem();

	sem_wait(&sem);
	sem_destroy(&sem);

	print_results();

	double timing = timing_delay(&start, &end);
	fprintf(stderr, "Computation took (in ms)\n");
	printf("%2.2f\n", timing/1000);

	return 0;
}
