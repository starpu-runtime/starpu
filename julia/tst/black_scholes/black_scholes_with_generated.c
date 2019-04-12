#include <stdio.h>
#include <stdlib.h>
#include <starpu.h>
#include "../includes/sorting.h"

void black_scholes(void **, void *);
void CUDA_black_scholes(void **, void*);

static struct starpu_perfmodel model =
{
	.type = STARPU_HISTORY_BASED,
	.symbol = "history_perf"
};

struct starpu_codelet cl =
{
	.cpu_funcs = {black_scholes},
	.cuda_funcs = {CUDA_black_scholes},
	.nbuffers = 2,
	.modes = {STARPU_R, STARPU_RW},
	.model = &model
};

void black_scholes_with_starpu(double *data, double *res, unsigned nslices, unsigned nbr_data)
{

	starpu_data_handle_t D_handle, RES_handle;

	starpu_matrix_data_register(&D_handle, STARPU_MAIN_RAM, (uintptr_t)data, 5, 5, nbr_data, sizeof(double));
	starpu_matrix_data_register(&RES_handle, STARPU_MAIN_RAM, (uintptr_t)res, 2, 2, nbr_data, sizeof(double));

	struct starpu_data_filter vert =
	{
		.filter_func = starpu_matrix_filter_vertical_block,
		.nchildren = nslices
	};

	starpu_data_partition(D_handle, &vert);
	starpu_data_partition(RES_handle, &vert);

	unsigned taskx;
	
	for (taskx = 0; taskx < nslices; taskx++){
		struct starpu_task *task = starpu_task_create();
		
		task->cl = &cl;
		task->handles[0] = starpu_data_get_sub_data(D_handle, 1, taskx);
		task->handles[1] = starpu_data_get_sub_data(RES_handle, 1, taskx);
		
		starpu_task_submit(task);
	}

	starpu_task_wait_for_all();


	starpu_data_unpartition(D_handle, STARPU_MAIN_RAM);
	starpu_data_unpartition(RES_handle, STARPU_MAIN_RAM);

	starpu_data_unregister(D_handle);
	starpu_data_unregister(RES_handle);
}
	

void init_data(double *data, unsigned nbr_data)
{
	unsigned i;
	for (i = 0; i < nbr_data; i++){

		data[5*i] = 100. * rand() / (double) RAND_MAX;
		data[5*i + 1] = 100. * rand() / (double) RAND_MAX;
		data[5*i + 2] = rand() / (double) RAND_MAX;
		data[5*i + 3] = 10. * rand() / (double) RAND_MAX;
		data[5*i + 4] = 10. * rand() / (double) RAND_MAX;
		
	}
}

double median_time(unsigned nbr_data, unsigned nslices, unsigned nbr_tests)
{
	double *data = malloc(5 * nbr_data * sizeof(double));
	double *res = calloc(2 * nbr_data, sizeof(double));
	double exec_times[nbr_tests];
	
	/* printf("nbr_data: %u\n", nbr_data); */
	unsigned i;
	for (i = 0; i < nbr_tests; i++){
		
		init_data(data, nbr_data);
		/* data[0] = 100.0; */
		/* data[1] = 100.0; */
		/* data[2] = 0.05; */
		/* data[3] = 1.0; */
		/* data[4] = 0.2; */

		double start = starpu_timing_now();
		black_scholes_with_starpu(data, res, nslices, nbr_data);
		double stop = starpu_timing_now();
		
		exec_times[i] = (stop-start)/1.e6;
		
		
	}

	/* printf("RES:\n%f\n%f\n", res[0], res[1]); */

	free(data);
	free(res);

	quicksort(exec_times, 0, nbr_tests - 1);
	return exec_times[nbr_tests/2];
}
	


void display_times(unsigned start_nbr, unsigned step_nbr, unsigned stop_nbr, unsigned nslices, unsigned nbr_tests){
	
	double t;
	unsigned nbr_data;

	FILE *myfile;
	myfile = fopen("DAT/black_scholes_c_generated_times.dat", "w");

	for (nbr_data = start_nbr; nbr_data <= stop_nbr; nbr_data+=step_nbr){
		t = median_time(nbr_data, nslices, nbr_tests);
		printf("Number of data: %u\nTime: %f\n", nbr_data, t);
		fprintf(myfile, "%f\n", t);
	}
	fclose(myfile);
}

int main(int argc, char *argv[])
{
	if (argc != 6){
		printf("Usage: %s start_nbr step_nbr stop_nbr nslices nbr_tests\n", argv[0]);
		return 1;
	}
	
	if (starpu_init(NULL) != EXIT_SUCCESS){
		fprintf(stderr, "ERROR\n");
		return 77;
	}

	unsigned start_nbr = (unsigned) atoi(argv[1]);
	unsigned step_nbr = (unsigned) atoi(argv[2]);
	unsigned stop_nbr = (unsigned) atoi(argv[3]);
	unsigned nslices = (unsigned) atoi(argv[4]);
	unsigned nbr_tests = (unsigned) atoi(argv[5]);


	display_times(start_nbr, step_nbr, stop_nbr, nslices, nbr_tests);
		
	starpu_shutdown();

	return 0;
}
