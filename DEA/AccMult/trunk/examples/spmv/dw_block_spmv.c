#include "dw_block_spmv.h"
#include "matrix-market/mm_to_bcsr.h"

tick_t start,end;
sem_t sem;

unsigned c = 256;
unsigned r = 256;


unsigned remainingjobs = -1;

data_state sparse_matrix;
data_state vector_in, vector_out;

uint32_t size;
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

	size = c*r*get_bcsr_nnz(&sparse_matrix);
//	printf("size = %d \n ", size);

	/* initiate the 2 vectors */
	vector_in_ptr = malloc(size*sizeof(float));
	assert(vector_in_ptr);

	vector_out_ptr = malloc(size*sizeof(float));
	assert(vector_out_ptr);

	/* fill those */
	unsigned ind;
	for (ind = 0; ind < size; ind++)
	{
		vector_in_ptr[ind] = 2.0f;
		vector_out_ptr[ind] = 0.0f;
	}

	monitor_vector_data(&vector_in, 0, (uintptr_t)vector_in_ptr, size, sizeof(float));
	monitor_vector_data(&vector_out, 0, (uintptr_t)vector_out_ptr, size, sizeof(float));
}

void init_problem_callback(void *arg)
{
	unsigned *remaining = arg;

	unsigned val = ATOMIC_ADD(remaining, -1);

//	if (val < 10)
//		printf("callback %d remaining \n", val);

	if ( val == 0 )
	{
		printf("DONE ...\n");
		GET_TICK(end);

//		unpartition_data(&sparse_matrix, 0);
		unpartition_data(&vector_out, 0);

		sem_post(&sem);
	}
}


void call_filters(void)
{

	filter bcsr_f;
	filter vector_in_f, vector_out_f;

	bcsr_f.filter_func    = canonical_block_filter_bcsr;

	vector_in_f.filter_func = block_filter_func_vector;
	vector_in_f.filter_arg  = size/c;
	
	vector_out_f.filter_func = block_filter_func_vector;
	vector_out_f.filter_arg  = size/r;

	partition_data(&sparse_matrix, &bcsr_f);

	partition_data(&vector_in, &vector_in_f);
	partition_data(&vector_out, &vector_out_f);
}

#define NSPMV	1000
unsigned totaljobs;

void launch_spmv_codelets(void)
{
	codelet *cl = malloc(sizeof(codelet));

	/* we call one codelet per block */
	unsigned nblocks = get_bcsr_nnz(&sparse_matrix); 
	unsigned nrows = get_bcsr_nrow(&sparse_matrix); 

	remainingjobs = NSPMV*nblocks;
	totaljobs = remainingjobs;

	printf("there will be %d codelets\n", remainingjobs);

	uint32_t *rowptr = get_bcsr_local_rowptr(&sparse_matrix);
	uint32_t *colind = get_bcsr_local_colind(&sparse_matrix);

	cl->cl_arg = NULL;
	cl->core_func =  core_block_spmv;
#if defined (USE_CUBLAS) || defined (USE_CUDA)
	cl->cublas_func = cublas_block_spmv;
#endif

	GET_TICK(start);

	unsigned loop;
	for (loop = 0; loop < NSPMV; loop++)
	{

	unsigned row;
	unsigned part = 0;

	for (row = 0; row < nrows; row++)
	{
		unsigned index;

		if (rowptr[row] == rowptr[row+1])
		{
			continue;
		}

		for (index = rowptr[row]; index < rowptr[row+1]; index++, part++)
		{
			job_t job;
			job = job_create();

			job->where = CORE|CUBLAS;
			job->cb = init_problem_callback;
			job->argcb = &remainingjobs;
			job->cl = cl;

			unsigned i = colind[index];
			unsigned j = row;
	
			job->nbuffers = 3;
			job->buffers[0].state = get_sub_data(&sparse_matrix, 1, part);
			job->buffers[0].mode  = R;
			job->buffers[1].state = get_sub_data(&vector_in, 1, i);
			job->buffers[1].mode = R;
			job->buffers[2].state = get_sub_data(&vector_out, 1, j);
			job->buffers[2].mode = W;

//			printf("submit task %d (i=%d ,j=%d)\n", part, i, j);
	
			push_task(job);
		}
	}
	}
}

void init_problem(void)
{
	/* create the sparse input matrix */
	create_data();

	/* create a new codelet that will perform a SpMV on it */
	call_filters();
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

	init_problem();

	launch_spmv_codelets();

	sem_wait(&sem);
	sem_destroy(&sem);

	print_results();

	double totalflop = 2.0*c*r*totaljobs;

	double timing = timing_delay(&start, &end);
	fprintf(stderr, "Computation took (in ms)\n");
	printf("%2.2f\n", timing/1000);
	fprintf(stderr, "Flop %e\n", totalflop);
	fprintf(stderr, "GFlops : %2.2f\n", totalflop/timing/1000);

	return 0;
}
