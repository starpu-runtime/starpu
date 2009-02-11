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

#define TAG(i, j, iter)	((uint64_t) ( ((iter)*262144) |  ((j)*8192) | (i)) )

sem_t sem;
codelet cl;

#define Ni	1024
#define Nj	32
#define Nk	512

typedef struct coord_s {
	int i;
	int j;
} coord_t;

//coord_t coords[Ni][Nj];

static unsigned callback_cnt;
static unsigned iter = 0;

void callback_core(void *argcb);
static void express_deps(unsigned nj, unsigned i, unsigned j, unsigned iter);

static void tag_cleanup_grid(unsigned ni, unsigned nj, unsigned iter)
{
	unsigned i,j;

	for (j = 0; j < nj; j++)
	for (i = 0; i < ni; i++)
	{
		tag_remove(TAG(i,j,iter));
	}


} 

static void create_task_grid(unsigned ni, unsigned nj, unsigned iter)
{
	unsigned i, j;

	fprintf(stderr, "start iter %d...\n", iter);

	callback_cnt = (Ni*Nj);

	/* create non-entry tasks */
	for (j = 0; j < nj; j++)
//	for (i = ni -1; i > 0; i--)
	for (i = 1; i < ni; i++)
	{
		//coords[i][j].i = i;
		//coords[i][j].j = j;

		/* create a new task */
		job_t jb = job_create();
		jb->where = CORE;
		jb->cb = callback_core;
		//jb->argcb = &coords[i][j];
		jb->cl = &cl;

		tag_declare(TAG(i,j, iter), jb);
//		fprintf(stderr, "job %p TAG %lx\n", jb, TAG(i,j));

		/* express deps : (i,j) depends on (i-1, j-1) & (i-1, j+1) */		
		if (i == 0) {
			/* this is an entry task */
			push_task(jb);
		}
		else {
			express_deps(nj, i, j, iter);
		}
	}

	/* create entry tasks */
	for (j = 0; j < nj; j++)
	{
		//coords[0][j].i = 0;
		//coords[0][j].j = j;

		/* create a new task */
		job_t jb = job_create();
		jb->where = CORE;
		jb->cb = callback_core;
		//jb->argcb = &coords[0][j];
		jb->cl = &cl;

		tag_declare(TAG(0,j, iter), jb);
//		fprintf(stderr, "job %p TAG %lx\n", jb, TAG(i,j));

		/* this is an entry task */
		push_task(jb);
	}

}


void callback_core(void *argcb)
{
//	struct coord_s *c = argcb;
	unsigned newcnt = ATOMIC_ADD(&callback_cnt, -1);	

//	printf("callback core %d (i = %d, j = %d)\n", newcnt, c->i, c->j);
	if (newcnt == 0)
	{
		
		iter++;
		if (iter < Nk)
		{
			/* cleanup old grids ... */
			if (iter > 2)
				tag_cleanup_grid(Ni, Nj, iter-2);

			/* create a new iteration */
			create_task_grid(Ni, Nj, iter);
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

static void express_deps(unsigned nj, unsigned i, unsigned j, unsigned iter)
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
	init_machine();

	cl.cl_arg = NULL;
	cl.core_func = core_codelet;
	cl.cublas_func = core_codelet;

	sem_init(&sem, 0, 0);

	create_task_grid(Ni, Nj, 0);

	sem_wait(&sem);

	fprintf(stderr, "TEST DONE ...\n");

	return 0;
}
