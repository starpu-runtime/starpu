#include <sys/time.h>
#include <stdio.h>
#include <unistd.h>
#include <semaphore.h>

#include <starpu.h>

static sem_t sem;

static unsigned ntasks = 1024;
static unsigned cnt;

static void *dummy_func(void *arg __attribute__ ((unused)))
{
	return NULL;
}

static starpu_codelet dummy_codelet = 
{
	.where = ANY,
	.core_func = dummy_func,
	.cublas_func = dummy_func,
	.model = NULL,
	.nbuffers = 0
};

void callback(void *arg)
{
	unsigned res = STARPU_ATOMIC_ADD(&cnt, -1);

	if (res == 0)
		sem_post(&sem);
}

void inject_one_task(void)
{
	struct starpu_task *task = starpu_task_create();

	task->cl = &dummy_codelet;
	task->cl_arg = NULL;
	task->callback_func = callback;
	task->callback_arg = NULL;

	starpu_submit_task(task);
}

static void parse_args(int argc, char **argv)
{
	int c;
	while ((c = getopt(argc, argv, "i:")) != -1)
	switch(c) {
		case 'i':
			ntasks = atoi(optarg);
			break;
	}

	
}

int main(int argc, char **argv)
{
	unsigned i;
	double timing;
	struct timeval start;
	struct timeval end;

	sem_init(&sem, 0, 0);

	parse_args(argc, argv);

	cnt = ntasks;

	starpu_init();

	fprintf(stderr, "#tasks : %d\n", ntasks);

	gettimeofday(&start, NULL);
	for (i = 0; i < ntasks; i++)
	{
		inject_one_task();
	}

	sem_wait(&sem);
	gettimeofday(&end, NULL);

	timing = (double)((end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec));

	fprintf(stderr, "Total: %le usecs\n", timing);
	fprintf(stderr, "Per task: %le usecs\n", timing/ntasks);

	starpu_shutdown();

	return 0;
}
