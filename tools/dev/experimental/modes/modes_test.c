/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012-2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 *
 * StarPU is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * StarPU is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */
/* Dont change anything ! */
struct starpu_codelet cummy_cl = 
{
        .cpu_funcs = { foo },
        .nbuffers = 42
};

/* Now, there is some work to do */
struct starpu_codelet cl1 = 
{
        .cpu_funcs = { foo, bar },
        .nbuffers = 2,
};


int
foo(void)
{
        struct starpu_task *task = starpu_task_create();
        task->cl = &cl1;
        task->buffers[0].handle = handle1;
        task->buffers[0].mode = STARPU_R;
        task->synchronous = 1;
        task->buffers[1].handle = handles[1];
        task->buffers[1].mode = STARPU_W;
}

struct starpu_codelet cl2 = 
{
	.cpu_funcs = {foo},
	.nbuffers = 1
};


static void
bar(void)
{
	struct starpu_task *task = starpu_task_create();
	task->cl = &cl2;
	task->buffers[0].handle = h;
	task->buffers[0].mode = STARPU_RW;

	task->buffers[1].handle = h;
	task->buffers[1].mode = random_mode();
}

struct starpu_codelet cl3 = 
{
	.cpu_funcs = { bar, baz },
	.nbuffers = 1
};

static void
baz(void)
{
	struct starpu_task *blah;
	blah->cl = &cl3;
	blah->buffers[0].handle = some_handle;
	blah->buffers[0].mode = STARPU_RW;
}
