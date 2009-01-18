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

void callback_core(void *argcb __attribute__ ((unused)))
{
	printf("callback core\n");
}

void last_callback_core(void *argcb __attribute__ ((unused)))
{
	printf("callback core\n");
}

void core_codelet(void *_args __attribute__ ((unused)))
{
	printf("codelet core\n");
}



int main(int argc __attribute__((unused)) , char **argv __attribute__((unused)))
{
	init_machine();

	codelet cl;
	codelet cl2;
	codelet cl3;

	job_t j = job_create();
	j->where = ANY;
	j->cb = callback_core;
	j->cl = &cl;

	job_t j2 = job_create();
	j2->where = CORE;
	j2->cb = callback_core;
	j2->cl = &cl2;

	job_t j3 = job_create();
	j3->type = CODELET;
	j3->where = CORE;
	j3->cb = last_callback_core;
	j3->cl = &cl3;



	cl.cl_arg = NULL;
	cl.core_func = core_codelet;

	cl2.cl_arg = NULL;
	cl2.core_func = core_codelet;

	cl3.cl_arg = NULL;
	cl3.core_func = core_codelet;

	tag_declare(42, j);
	tag_declare(1664, j2);
	tag_declare(10000000, j3);

	tag_declare_deps(1664, 1, 42);
	tag_declare_deps(10000000, 2, 42, 1664);

	push_task_sync(j);

	tag_remove(10000000);
	tag_remove(1664);
	tag_remove(42);

	return 0;
}
