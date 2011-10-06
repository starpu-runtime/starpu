/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010, 2011  Centre National de la Recherche Scientifique
 * Copyright (C) 2010, 2011  Université de Bordeaux 1
 *
 * Permission is granted to copy, distribute and/or modify this document
 * under the terms of the GNU Free Documentation License, Version 1.3
 * or any later version published by the Free Software Foundation;
 * with no Invariant Sections, no Front-Cover Texts, and no Back-Cover Texts.
 * See the GNU Free Documentation License in COPYING.GFDL for more details.
 */

#include <starpu.h>

struct params {
    int i;
    float f;
};

void cpu_func(void *buffers[], void *cl_arg)
{
    struct params *params = cl_arg;

    printf("Hello world (params = {%i, %f} )\n", params->i, params->f);
}

starpu_codelet cl =
{
    .where = STARPU_CPU,
    .cpu_func = cpu_func,
    .nbuffers = 0
};

void callback_func(void *callback_arg)
{
    printf("Callback function (arg %x)\n", callback_arg);
}

int main(int argc, char **argv)
{
    /* initialize StarPU */
    starpu_init(NULL);

    struct starpu_task *task = starpu_task_create();

    task->cl = &cl; /* Pointer to the codelet defined above */

    struct params params = { 1, 2.0f };
    task->cl_arg = &params;
    task->cl_arg_size = sizeof(params);

    task->callback_func = callback_func;
    task->callback_arg = 0x42;

    /* starpu_task_submit will be a blocking call */
    task->synchronous = 1;

    /* submit the task to StarPU */
    starpu_task_submit(task);

    /* terminate StarPU */
    starpu_shutdown();

    return 0;
}
