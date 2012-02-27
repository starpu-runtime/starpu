static void
f(void)
{
	int err;
	err = starpu_task_submit(task1);

	starpu_task_submit(task2);
	return;
}

static void
g(void)
{
	starpu_task_submit(task);
}
