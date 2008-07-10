/*
 * Conjugate gradients for Sparse matrices
 */

#include <semaphore.h>
#include <core/jobs.h>
#include <core/workers.h>
#include <core/tags.h>
#include <common/timing.h>
#include <common/util.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <sys/types.h>
#include <pthread.h>
#include <signal.h>
#include <cblas.h>
#include <common/timing.h>

#include <datawizard/coherency.h>
#include <datawizard/hierarchy.h>
#include <datawizard/interfaces/blas_interface.h>
#include <datawizard/interfaces/blas_filters.h>


/* First a Matrix-Vector product (SpMV) */

sem_t sem;

uint32_t size;

void parse_args(int argc, char **argv)
{
	int i;
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-size") == 0) {
			char *argptr;
			size = strtol(argv[++i], &argptr, 10);
		}
	}
}

void create_data(void)
{

}

void call_spmv_codelet()
{

}

void init_problem_callback(void *arg __attribute__((unused)))
{
	sem_post(&sem);
}

void init_problem(void)
{
	/* create the sparse input matrix */
	create_data();

	/* create a new codelet that will perform a SpMV on it */
	call_spmv_codelet();
}

int main(__attribute__ ((unused)) int argc,
	__attribute__ ((unused)) char **argv)
{
	parse_args(argc, argv);

	/* start the runtime */
	init_machine();
	init_workers();

	sem_init(&sem, 0, 0U);

	init_problem();

	sem_wait(&sem);
	sem_destroy(&sem);

	return 0;
}
