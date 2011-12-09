static void
foo(void)
{
	abort();
}

static void
bar(struct starpu_task *task)
{
	assert(task && task->cl);
}
