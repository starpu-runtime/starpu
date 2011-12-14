#include <starpu.h>

static int can_always_execute(unsigned workerid,
			      struct starpu_task *task,
			      unsigned nimpl)
{
	(void) workerid;
	(void) task;
	(void) nimpl;

	return 1;
}

static int can_never_execute(unsigned workerid,
			     struct starpu_task *task,
			     unsigned nimpl)
{
	(void) workerid;
	(void) task;
	(void) nimpl;

	return 0;
}

static void fake(void *buffers[], void *args)
{
	(void) buffers;
	(void) args;
}

static struct starpu_codelet cl =
{
	.where        = STARPU_CPU | STARPU_CUDA | STARPU_OPENCL,
	.cpu_funcs    = { fake, NULL},
	.cuda_funcs   = { fake, NULL},
	.opencl_funcs = { fake, NULL},
	.nbuffers     = 0
};

int
main(void)
{
	int ret;
	struct starpu_task *task;

	starpu_init(NULL);

	task = starpu_task_create();
	task->cl = &cl;

	cl.can_execute = NULL;
	ret = _starpu_worker_exists(task);
	if (!ret)
		return EXIT_FAILURE;

	cl.can_execute = can_always_execute;
	ret = _starpu_worker_exists(task);
	if (!ret)
		return EXIT_FAILURE;

	cl.can_execute = can_never_execute;
	ret = _starpu_worker_exists(task);
	if (ret)
		return EXIT_FAILURE;

	starpu_shutdown();
	
	return EXIT_SUCCESS;
}
