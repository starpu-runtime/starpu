/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <starpu.h>
#include <common/thread.h>

#include "increment_codelet.h"
#include "../helper.h"

/*
 * Test that one can let a whole task graph repeatedly regenerate itself, using
 * tag dependencies, with a complete cycle.
 */

#ifdef STARPU_QUICK_CHECK
static unsigned niter = 64;
#else
static unsigned niter = 16384;
#endif

#define TAG_A 1
#define TAG_B 2
#define TAG_C 3
#define TAG_D 4

/*
 *
 *		    /-->B--\
 *		    |      |
 *	     -----> A      D---\--->
 *		^   |      |   |
 *		|   \-->C--/   |
 *		|              |
 *		\--------------/
 *
 *	- {B, C} depend on A
 *	- D depends on {B, C}
 *	- A, B, C and D are resubmitted at the end of the loop (or not)
 */

static struct starpu_task taskA, taskB, taskC, taskD;

static unsigned loop_cnt = 0;
static unsigned loop_cnt_A = 0;
static unsigned loop_cnt_B = 0;
static unsigned loop_cnt_C = 0;
static unsigned *check_cnt;
static starpu_pthread_cond_t cond = STARPU_PTHREAD_COND_INITIALIZER;
static starpu_pthread_mutex_t mutex = STARPU_PTHREAD_MUTEX_INITIALIZER;

static void callback_task_A(void *arg)
{
	(void)arg;

	loop_cnt_A++;

	if (loop_cnt_A == niter)
	{
		/* We are done */
		taskA.regenerate = 0;
	}
}

static void callback_task_B(void *arg)
{
	(void)arg;

	loop_cnt_B++;

	if (loop_cnt_B == niter)
	{
		/* We are done */
		taskB.regenerate = 0;
	}
}

static void callback_task_C(void *arg)
{
	(void)arg;

	loop_cnt_C++;

	if (loop_cnt_C == niter)
	{
		/* We are done */
		taskC.regenerate = 0;
	}
}

static void callback_task_D(void *arg)
{
	(void)arg;

	STARPU_PTHREAD_MUTEX_LOCK(&mutex);
	loop_cnt++;

	if (loop_cnt == niter)
	{
		/* We are done */
		taskD.regenerate = 0;
		STARPU_PTHREAD_COND_SIGNAL(&cond);
		STARPU_PTHREAD_MUTEX_UNLOCK(&mutex);
	}
	else
	{
		STARPU_PTHREAD_MUTEX_UNLOCK(&mutex);
		/* Let's go for another iteration */
	}
}

int main(int argc, char **argv)
{
//	unsigned i;
//	double timing;
//	double start;
//	double end;
	int ret;

	ret = starpu_initialize(NULL, &argc, &argv);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	/* Implicit data dependencies and regeneratable tasks are not compatible */
	starpu_data_set_default_sequential_consistency_flag(0);

	starpu_malloc((void**)&check_cnt, sizeof(*check_cnt));
	*check_cnt = 0;

	starpu_data_handle_t check_data;
	starpu_variable_data_register(&check_data, STARPU_MAIN_RAM, (uintptr_t)check_cnt, sizeof(*check_cnt));

	starpu_task_init(&taskA);
	taskA.cl = &increment_codelet;
	taskA.regenerate = 1; /* this task will be explicitely resubmitted if needed */
	taskA.use_tag = 1;
	taskA.tag_id = TAG_A;
	taskA.callback_func = callback_task_A;
	taskA.handles[0] = check_data;

	starpu_task_init(&taskB);
	taskB.cl = &increment_codelet;
	taskB.regenerate = 1;
	taskB.use_tag = 1;
	taskB.tag_id = TAG_B;
	taskB.callback_func = callback_task_B;
	taskB.handles[0] = check_data;

	starpu_task_init(&taskC);
	taskC.cl = &increment_codelet;
	taskC.regenerate = 1;
	taskC.use_tag = 1;
	taskC.tag_id = TAG_C;
	taskC.callback_func = callback_task_C;
	taskC.handles[0] = check_data;

	starpu_task_init(&taskD);
	taskD.cl = &increment_codelet;
	taskD.callback_func = callback_task_D;
	taskD.regenerate = 1;
	taskD.use_tag = 1;
	taskD.tag_id = TAG_D;
	taskD.handles[0] = check_data;

	starpu_tag_declare_deps((starpu_tag_t) TAG_B, 1, (starpu_tag_t) TAG_A);
	starpu_tag_declare_deps((starpu_tag_t) TAG_C, 1, (starpu_tag_t) TAG_A);

	starpu_tag_declare_deps((starpu_tag_t) TAG_D, 2, (starpu_tag_t) TAG_B, (starpu_tag_t) TAG_C);

	starpu_tag_declare_deps((starpu_tag_t) TAG_A, 1, (starpu_tag_t) TAG_D);

	ret = starpu_task_submit(&taskA); if (ret == -ENODEV) goto enodev; STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	ret = starpu_task_submit(&taskB); if (ret == -ENODEV) goto enodev; STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	ret = starpu_task_submit(&taskC); if (ret == -ENODEV) goto enodev; STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	ret = starpu_task_submit(&taskD); if (ret == -ENODEV) goto enodev; STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

	/* Break the loop */
	starpu_tag_notify_restart_from_apps((starpu_tag_t) TAG_D);

	starpu_do_schedule();
	/* Wait for the termination of all loops */
	STARPU_PTHREAD_MUTEX_LOCK(&mutex);
	if (loop_cnt < niter)
		STARPU_PTHREAD_COND_WAIT(&cond, &mutex);
	STARPU_PTHREAD_MUTEX_UNLOCK(&mutex);

	starpu_data_acquire(check_data, STARPU_R);
	starpu_data_release(check_data);

	STARPU_ASSERT(*check_cnt == (4*loop_cnt));

	starpu_free(check_cnt);

	starpu_data_unregister(check_data);

	starpu_task_wait_for_all();
	starpu_task_clean(&taskA);
	starpu_task_clean(&taskB);
	starpu_task_clean(&taskC);
	starpu_task_clean(&taskD);

	starpu_shutdown();

	return EXIT_SUCCESS;

enodev:
	fprintf(stderr, "WARNING: No one can execute this task\n");
	/* yes, we do not perform the computation but we did detect that no one
 	 * could perform the kernel, so this is not an error from StarPU */
	starpu_data_unregister(check_data);
	starpu_shutdown();
	return STARPU_TEST_SKIPPED;
}

