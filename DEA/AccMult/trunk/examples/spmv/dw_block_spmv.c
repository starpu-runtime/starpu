#include "dw_block_spmv.h"
#include "matrix-market/mm_to_bcsr.h"

tick_t start,end;
sem_t sem;

unsigned c = 128;
unsigned r = 128;

unsigned nblocks = 1;
unsigned remainingjobs = -1;

data_state sparse_matrix;
data_state vector_in, vector_out;

char *inputfile;
bcsr_t *bcsr_matrix;

float *vector_in_ptr;
float *vector_out_ptr;

unsigned usecpu = 0;

void create_data(void)
{
	/* read the input file */
	bcsr_matrix = mm_file_to_bcsr(inputfile, c, r);

	/* declare the corresponding block CSR to the runtime */
	monitor_bcsr_data(&sparse_matrix, 0, bcsr_matrix->nnz_blocks, bcsr_matrix->nrows_blocks,
	                (uintptr_t)bcsr_matrix->val, bcsr_matrix->colind, bcsr_matrix->rowptr, 
			0, bcsr_matrix->r, bcsr_matrix->c, sizeof(float));

	///* initiate the 2 vectors */
	//float *invec, *outvec;
	//invec = malloc(size*sizeof(float));
	//assert(invec);

	//outvec = malloc(size*sizeof(float));
	//assert(outvec);

	///* fill those */
	//unsigned ind;
	//for (ind = 0; ind < size; ind++)
	//{
	//	invec[ind] = 2.0f;
	//	outvec[ind] = 0.0f;
	//}

	//monitor_vector_data(&vector_in, 0, (uintptr_t)invec, size, sizeof(float));
	//monitor_vector_data(&vector_out, 0, (uintptr_t)outvec, size, sizeof(float));

	//vector_in_ptr = invec;
	//vector_out_ptr = outvec;

}
//
//void init_problem_callback(void *arg)
//{
//	unsigned *remaining = arg;
//
//
//	unsigned val = ATOMIC_ADD(remaining, -1);
//
//	printf("callback %d remaining \n", val);
//	if ( val == 0 )
//	{
//		printf("DONE ...\n");
//		GET_TICK(end);
//
//		unpartition_data(&sparse_matrix, 0);
//		unpartition_data(&vector_out, 0);
//
//		sem_post(&sem);
//	}
//}
//

void call_spmv_codelet_filters(void)
{

	remainingjobs = nblocks;

	codelet *cl = malloc(sizeof(codelet));

	filter bcsr_f;
	filter vector_in_f, vector_out_f;

	bcsr_f.filter_func    = canonical_block_filter_bcsr;

//	vector_f.filter_func = block_filter_func_vector;
//	vector_f.filter_arg  = nblocks;

	partition_data(&sparse_matrix, &bcsr_f);
//	partition_data(&vector_out, &vector_f);

//	cl->cl_arg = NULL;
//	cl->core_func =  core_spmv;
//#ifdef USE_CUDA
//	cl->cuda_func = &cuda_spmv;
//#endif
//
//
//	GET_TICK(start);
//	unsigned part;
//	for (part = 0; part < nblocks; part++)
//	{
//		job_t job;
//		job = job_create();
////#ifdef USE_CUDA
////		job->where = usecpu?CORE:CUDA;
////#else
////		job->where = CORE;
////#endif
//		job->where = CORE|CUDA;
//		job->cb = init_problem_callback;
//		job->argcb = &remainingjobs;
//		job->cl = cl;
//	
//		job->nbuffers = 3;
//		job->buffers[0].state = get_sub_data(&sparse_matrix, 1, part);
//		job->buffers[0].mode  = R;
//		job->buffers[1].state = &vector_in;
//		job->buffers[1].mode = R;
//		job->buffers[2].state = get_sub_data(&vector_out, 1, part);
//		job->buffers[2].mode = W;
//	
//		push_task(job);
//	}
}


//
//void call_spmv_codelet(void)
//{
//
//	remainingjobs = 1;
//
//	job_t job;
//	codelet *cl = malloc(sizeof(codelet));
//
//	cl->cl_arg = NULL;
//	cl->core_func =  core_spmv;
//#ifdef USE_CUDA
//	cl->cuda_func = &cuda_spmv;
//#endif
//
//	job = job_create();
//#ifdef USE_CUDA
//	job->where = usecpu?CORE:CUDA;
//#else
//	job->where = CORE;
//#endif
//	job->cb = init_problem_callback;
//	job->argcb = &remainingjobs;
//	job->cl = cl;
//
//	job->nbuffers = 3;
//	job->buffers[0].state = &sparse_matrix;
//	job->buffers[0].mode  = R;
//	job->buffers[1].state = &vector_in;
//	job->buffers[1].mode = R;
//	job->buffers[2].state = &vector_out;
//	job->buffers[2].mode = W;
//
//
//
//
//	GET_TICK(start);
//}
//
void init_problem(void)
{
	/* create the sparse input matrix */
	create_data();

	/* create a new codelet that will perform a SpMV on it */
	call_spmv_codelet_filters();
}

//void print_results(void)
//{
//	unsigned row;
//
//	for (row = 0; row < MIN(size, 16); row++)
//	{
//		printf("%2.2f\t%2.2f\n", vector_in_ptr[row], vector_out_ptr[row]);
//	}
//}

int main(__attribute__ ((unused)) int argc,
	__attribute__ ((unused)) char **argv)
{
//	parse_args(argc, argv);
	
	if (argc < 2)
	{
		fprintf(stderr, "usage : %s filename\n", argv[0]);
		exit(-1);
	}

	inputfile = argv[1];

	timing_init();

	/* start the runtime */
	init_machine();

	sem_init(&sem, 0, 0U);

//#ifdef USE_CUDA
//	initialize_cuda();
//#endif

	init_problem();

//	sem_wait(&sem);
//	sem_destroy(&sem);
//
//	//print_results();
//
	double timing = timing_delay(&start, &end);
	fprintf(stderr, "Computation took (in ms)\n");
	printf("%2.2f\n", timing/1000);

	return 0;
}
