#include <signal.h>
#include <stdlib.h>

#include "../helper.h"

/*
 * It is possible to depend on a task that is over, but not on a task that has
 * already been destroyed. In this test, we make sure things go wrong if taskB
 * depends upon the destroyed taskA. It should trigger STARPU_ASSERT or
 * STARPU_ABORT somewhere in StarPU, so we can try and cath SIGABRT. Note that
 * the error might be weirder, leading this test to fail. In this case, it is
 * probably OK to disable it for a while :-) Maybe we could also detect
 * destroyed tasks in starpu_task_declare_deps_array.
 */
static void abort_catcher(int sig)
{
	(void) sig;
	starpu_shutdown();

	/* Here, failure is success. */
	exit(EXIT_SUCCESS);
} 

int
main(void)
{
#ifdef STARPU_HAVE_VALGRIND_H
	if (RUNNING_ON_VALGRIND)
		return STARPU_TEST_SKIPPED;
#endif	

	int ret;
	struct starpu_task *taskA, *taskB;

	ret = starpu_init(NULL);
	if (ret == -ENODEV)
	{
		return STARPU_TEST_SKIPPED;
	}

	taskA = starpu_task_create();
	taskA->cl = NULL;
	taskA->detach = 0;

	taskB = starpu_task_create();
	taskB->cl = NULL;


	ret = starpu_task_submit(taskA);
	if (ret == -ENODEV)
	{
		starpu_shutdown();
		return STARPU_TEST_SKIPPED;
	}

	ret = starpu_task_wait(taskA);
	if (ret != 0)
	{
		starpu_shutdown();
		return EXIT_FAILURE;
	}

	/* taskA should have been destroyed by now. */
	struct sigaction sa;
	memset(&sa, 0, sizeof(sa));
	sa.sa_handler = abort_catcher;
	sigaction(SIGABRT, &sa, NULL);
	sigaction(SIGSEGV, &sa, NULL);

	starpu_task_declare_deps_array(taskB, 1, &taskA);

	ret = starpu_task_submit(taskB);
	if (ret == -ENODEV)
	{
		starpu_shutdown();
		return EXIT_FAILURE;
	}

	starpu_task_wait_for_all();
	starpu_shutdown();

	return EXIT_FAILURE;
}
