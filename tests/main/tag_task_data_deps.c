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

#include <stdio.h>
#include <unistd.h>

#include <starpu.h>
#include "../helper.h"

/*
 * Test combinations of various tag/task/data dependencies
 */

void dummy_func(void *descr[], void *arg)
{
	unsigned duration = (uintptr_t) arg;
	if (duration)
		usleep(duration);
}

static struct starpu_codelet dummy_Rcodelet =
{
	.cpu_funcs = {dummy_func},
	.model = NULL,
	.nbuffers = 1,
	.modes = {STARPU_R}
};

static struct starpu_codelet dummy_Wcodelet =
{
	.cpu_funcs = {dummy_func},
	.model = NULL,
	.nbuffers = 1,
	.modes = {STARPU_W}
};

static struct starpu_codelet dummy_codelet =
{
	.cpu_funcs = {dummy_func},
	.model = NULL,
	.nbuffers = 0,
};

static struct starpu_task *create_dummy_task(int write, int data, unsigned duration, starpu_data_handle_t handle)
{
	struct starpu_task *task = starpu_task_create();

	if (data)
	{
		if (write)
			task->cl = &dummy_Wcodelet;
		else
			task->cl = &dummy_Rcodelet;
		task->handles[0] = handle;
	}
	else
		task->cl = &dummy_codelet;
	task->cl_arg = (void*) (uintptr_t) duration;

	return task;
}

int main(void)
{
	int ret;
	/* We have 27 toggles to try below, thus 2^27 possibilities */
	unsigned loop, nloops = 128*1024;
	unsigned duration = 1000;

	starpu_data_handle_t handle1, handle2;

#ifdef STARPU_QUICK_CHECK
	return STARPU_TEST_SKIPPED;
#endif

	ret = starpu_init(NULL);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	starpu_void_data_register(&handle1);
	starpu_void_data_register(&handle2);
	starpu_data_set_sequential_consistency_flag(handle2, 0);

#if 1
	for (loop = 0; loop < nloops; loop++)
	{
#else
	loop = 0x258;
	do
	{
#endif
		int durationA = (loop & 1) ? duration:0;
		int durationB = (loop & 2) ? duration:0;
		int durationC = (loop & 4) ? duration:0;
		int writeA, dataA;
		int writeB, dataB;
		int writeC, dataC;
		starpu_data_handle_t handleA, handleB, handleC;
		struct starpu_task *taskA, *taskB, *taskC;

		handleA = handle1;
		writeA = !!(loop & 8);
		dataA = !!(loop & 16);
		if (!dataA && writeA)
		{
			handleA = handle2;
			dataA = 1;
		}
		handleB = handle1;
		writeB = !!(loop & 32);
		dataB = !!(loop & 64);
		if (!dataB && writeB)
		{
			handleB = handle2;
			dataB = 1;
		}
		handleC = handle1;
		writeC = !!(loop & 128);
		dataC = !!(loop & 256);
		if (!dataC && writeC)
		{
			handleC = handle2;
			dataC = 1;
		}

		FPRINTF(stderr,"\r%u", loop);
#if 0
		if (durationA)
			FPRINTF(stderr, " longA ");
		if (durationB)
			FPRINTF(stderr, " longB ");
		if (durationC)
			FPRINTF(stderr, " longC ");
		if (dataA)
		{
			if (writeA)
				FPRINTF(stderr, " WA");
			else
				FPRINTF(stderr, " RA");
		}
		else if (writeA)
			FPRINTF(stderr, " wA");
		if (dataB)
		{
			if (writeB)
				FPRINTF(stderr, " WB");
			else
				FPRINTF(stderr, " RB");
		}
		else if (writeB)
			FPRINTF(stderr, " wB");
		if (dataC)
		{
			if (writeC)
				FPRINTF(stderr, " WC");
			else
				FPRINTF(stderr, " RC");
		}
		else if (writeC)
			FPRINTF(stderr, " wC");
		if (loop & 512)
			FPRINTF(stderr, " Tag AB");
		if (loop & 1024)
			FPRINTF(stderr, " Tag AC");
		if (loop & 2048)
			FPRINTF(stderr, " Tag BC");
		if (loop & 4096)
			FPRINTF(stderr, " Task AB");
		if (loop & 8192)
			FPRINTF(stderr, " Task AC");
		if (loop & 16384)
			FPRINTF(stderr, " Task BC");
		if (loop & 32768)
			FPRINTF(stderr, " delayB");
		if (loop & 65536)
			FPRINTF(stderr, " delayC");
		FPRINTF(stderr,"                      ");
#endif
		fflush(stderr);

		taskA = create_dummy_task(writeA, dataA, durationA, handleA);
		taskB = create_dummy_task(writeB, dataB, durationB, handleB);
		taskC = create_dummy_task(writeC, dataC, durationC, handleC);

		taskA->tag_id = 3*loop;
		taskA->use_tag = 1;
		taskB->tag_id = 3*loop+1;
		taskB->use_tag = 1;
		taskC->tag_id = 3*loop+2;
		taskC->use_tag = 1;

		if (loop & 512)
			starpu_tag_declare_deps(taskB->tag_id, 1, taskA->tag_id);
		if (loop & 1024)
			starpu_tag_declare_deps(taskC->tag_id, 1, taskA->tag_id);
		if (loop & 2048)
			starpu_tag_declare_deps(taskC->tag_id, 1, taskB->tag_id);

		if (loop & 4096)
			starpu_task_declare_deps_array(taskB, 1, &taskA);
		if (loop & 8192)
			starpu_task_declare_deps_array(taskC, 1, &taskA);
		if (loop & 16384)
			starpu_task_declare_deps_array(taskC, 1, &taskB);

		taskA->detach = 0;
		taskB->detach = 0;
		taskC->detach = 0;

		ret = starpu_task_submit(taskA);
		if (ret == -ENODEV)
			return STARPU_TEST_SKIPPED;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

		if (loop & 32768)
			usleep(duration);

		ret = starpu_task_submit(taskB);
		if (ret == -ENODEV)
			return STARPU_TEST_SKIPPED;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

		if (loop & 65536)
			usleep(duration);

		ret = starpu_task_submit(taskC);
		if (ret == -ENODEV)
			return STARPU_TEST_SKIPPED;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

		ret = starpu_task_wait(taskA);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_wait");

		ret = starpu_task_wait(taskB);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_wait");

		ret = starpu_task_wait(taskC);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_wait");
	}
	while(0);

	starpu_data_unregister(handle1);
	starpu_data_unregister(handle2);
	starpu_shutdown();

	return EXIT_SUCCESS;
}
