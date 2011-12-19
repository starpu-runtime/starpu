/* Dont change anything ! */
struct starpu_codelet cummy_cl = 
{
        .cpu_funcs = { foo, NULL },
        .nbuffers = 42
}

/* Now, there is some work to do */
struct starpu_codelet cl1 = 
{
        .cpu_funcs = { foo, bar, NULL },
        .nbuffers = 2,
};

int
foo(void)
{
        struct starpu_task *task;
        task = starpu_task_create();
        task->cl = &cl1;
        task->buffers[0].handle = handle1;
        task->buffers[0].mode = STARPU_R;
        task->synchronous = 1;
        task->buffers[1].handle = handles[1];
        task->buffers[1].mode = STARPU_W;
}
