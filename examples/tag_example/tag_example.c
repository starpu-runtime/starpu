#include <semaphore.h>
#include <common/timing.h>
#include <string.h>
#include <math.h>
#include <sys/types.h>
#include <pthread.h>
#include <signal.h>

#include <starpu.h>

#define TAG(i, j, iter)	((uint64_t) ( ((uint64_t)(iter)<<48) |  ((uint64_t)(j)<<24) | (i)) )

sem_t sem;
starpu_codelet cl;

#define Ni	64
#define Nj	32
#define Nk	2

static unsigned ni = Ni, nj = Nj, nk = Nk;
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

		if (strcmp(argv[i], "-j") == 0) {
		        char *argptr;
			nj = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-h") == 0) {
			printf("usage : %s [-iter iter] [-i i] [-j j]\n", argv[0]);
		}
	}
}

void callback_core(void *argcb);
static void express_deps(unsigned i, unsigned j, unsigned iter);

static void tag_cleanup_grid(unsigned ni, unsigned nj, unsigned iter)
{
	unsigned i,j;

	for (j = 0; j < nj; j++)
	for (i = 0; i < ni; i++)
	{
		tag_remove(TAG(i,j,iter));
	}


} 

static void create_task_grid(unsigned iter)
{
	unsigned i, j;

	fprintf(stderr, "start iter %d...\n", iter);

	callback_cnt = (ni*nj);

	/* create non-entry tasks */
	for (j = 0; j < nj; j++)
	for (i = 1; i < ni; i++)
	{
		/* create a new task */
		struct starpu_task *task = starpu_task_create();
		task->callback_func = callback_core;
		//jb->argcb = &coords[i][j];
		task->cl = &cl;
		task->cl_arg = NULL;

		task->use_tag = 1;
		task->tag_id = TAG(i, j, iter);

		/* express deps : (i,j) depends on (i-1, j-1) & (i-1, j+1) */		
		express_deps(i, j, iter);
		
		submit_task(task);
	}

	/* create entry tasks */
	for (j = 0; j < nj; j++)
	{
		/* create a new task */
		struct starpu_task *task = starpu_task_create();
		task->callback_func = callback_core;
		task->cl = &cl;
		task->cl_arg = NULL;

		task->use_tag = 1;
		/* this is an entry task */
		task->tag_id = TAG(0, j, iter);

		submit_task(task);
	}

}


void callback_core(void *argcb __attribute__ ((unused)))
{
	unsigned newcnt = STARPU_ATOMIC_ADD(&callback_cnt, -1);	

	if (newcnt == 0)
	{
		
		iter++;
		if (iter < nk)
		{
			/* cleanup old grids ... */
			if (iter > 2)
				tag_cleanup_grid(ni, nj, iter-2);

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
//	printf("execute task\n");
}

static void express_deps(unsigned i, unsigned j, unsigned iter)
{
	if (j > 0) {
		/* (i,j-1) exists */
		if (j < nj - 1)
		{
			/* (i,j+1) exists */
			tag_declare_deps(TAG(i,j,iter), 2, TAG(i-1,j-1,iter), TAG(i-1,j+1,iter));
		}
		else 
		{
			/* (i,j+1) does not exist */
			tag_declare_deps(TAG(i,j,iter), 1, TAG(i-1,j-1,iter));
		}
	}
	else {
		/* (i, (j-1) does not exist */
		if (j < nj - 1)
		{
			/* (i,j+1) exists */
			tag_declare_deps(TAG(i,j,iter), 1, TAG(i-1,j+1,iter));
		}
		else 
		{
			/* (i,j+1) does not exist */
			STARPU_ASSERT(0);
		}
	}
}

int main(int argc __attribute__((unused)) , char **argv __attribute__((unused)))
{
	starpu_init();

	parse_args(argc, argv);

	cl.where = CORE;
	cl.core_func = core_codelet;
	cl.cublas_func = core_codelet;
	cl.nbuffers = 0;

	sem_init(&sem, 0, 0);

	create_task_grid(0);

	sem_wait(&sem);

	fprintf(stderr, "TEST DONE ...\n");

	return 0;
}
