#include <semaphore.h>
#include <core/jobs.h>
#include <core/workers.h>
#include <common/timing.h>
#include <common/util.h>
#include <string.h>
#include <math.h>
#include <sys/types.h>
#include <pthread.h>
#include <signal.h>
#include <core/dependencies/tags.h>

#define TAG(i, iter)	((uint64_t)  ((iter)*(16777216) | (i)) )

sem_t sem;
codelet cl;

#define Ni	64
#define Nk	2

static unsigned ni, nk;
static unsigned callback_cnt;
static unsigned iter = 0;

static void parse_args(int argc, char **argv)
{
	int i;
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-iter") == 0) {
		        char *argptr;
			nk = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-i") == 0) {
		        char *argptr;
			ni = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-h") == 0) {
			printf("usage : %s [-iter iter] [-i i]\n", argv[0]);
		}
	}
}

void callback_core(void *argcb);
static void express_deps(unsigned i, unsigned iter);

static void tag_cleanup_grid(unsigned ni, unsigned iter)
{
	unsigned i;

	for (i = 0; i < ni; i++)
	{
		tag_remove(TAG(i,iter));
	}


} 

static void create_task_grid(unsigned iter)
{
	int i;

	fprintf(stderr, "start iter %d ni %d...\n", iter, ni);

	callback_cnt = (ni);

//	for (i = ni-1; i >= 0; i--)

	for (i = 0; i < ni; i++)
	{
		/* create a new task */
		job_t jb = job_create();
		jb->where = CORE;
		jb->cb = callback_core;
		//jb->argcb = &coords[i][j];
		jb->cl = &cl;

		tag_declare(TAG(i, iter), jb);

		if (i == 0)
		{
			push_task(jb);
		}
		else
		{
			tag_declare_deps(TAG(i,iter), 1, TAG(i-1,iter));
		}
	}
}


void callback_core(void *argcb __attribute__ ((unused)))
{
	fprintf(stderr, "callback\n");

	unsigned newcnt = ATOMIC_ADD(&callback_cnt, -1);	

	if (newcnt == 0)
	{
		
		iter++;
		if (iter < nk)
		{
			/* cleanup old grids ... */
			if (iter > 2)
				tag_cleanup_grid(ni, iter-2);

			/* create a new iteration */
			create_task_grid(iter);
		}
		else {
			sem_post(&sem);
		}
	}
}

void core_codelet(void *_args __attribute__ ((unused)))
{
	fprintf(stderr, "core_codelet\n");
}

int main(int argc __attribute__((unused)) , char **argv __attribute__((unused)))
{
	init_machine();

	parse_args(argc, argv);

	cl.cl_arg = NULL;
	cl.core_func = core_codelet;
	cl.cublas_func = core_codelet;

	sem_init(&sem, 0, 0);

	create_task_grid(0);

	sem_wait(&sem);

	fprintf(stderr, "TEST DONE ...\n");

	return 0;
}
