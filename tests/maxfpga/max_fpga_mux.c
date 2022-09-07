/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2019-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <stdlib.h>
#include <stdio.h>
#include <starpu_scheduler.h>
#include "../helper.h"

/* This examples shows the case of letting the runtime determing whether data
 * should be in CPU memory or DFE memory, by integrating a multiplexer in a
 * design that can be toggled dynamically.  */

#include "MyTasksMux.h"
#include <MaxSLiCInterface.h>
#define SIZE (192/sizeof(int32_t))

static max_file_t *maxfile;

/*
 * Dynamically configure multiplexer and streaming from CPU or from LMem (ignoring the other)
 */
#define setupReadData(name) do { \
	if (kind##name == STARPU_CPU_RAM) { \
		max_route(acts, "joinIn"#name".inCPU", "joinIn"#name".join"); \
		max_queue_input(acts, "in"#name"CPU", ptr##name, size * sizeof(uint32_t)); \
		max_ignore_memctl(acts, "MemoryControllerPro0", "in"#name"LMem"); \
	} else { \
		max_route(acts, "joinIn"#name".inLMem", "joinIn"#name".join"); \
		max_ignore_stream(acts, "in"#name"CPU"); \
		max_memctl_linear(acts, "MemoryControllerPro0", "in"#name"LMem", (size_t) ptr##name, size * sizeof(int32_t)); \
	} \
} while (0)

/*
 * Ignore data from unused input
 */
#define ignoreReadData(name) do { \
	max_route(acts, "joinIn"#name".inLMem", "joinIn"#name".join"); \
	max_ignore_stream(acts, "in"#name"CPU"); \
	max_ignore_memctl(acts, "MemoryControllerPro0", "in"#name"LMem"); \
} while (0)

/*
 * Configure demultiplexer and streaming to CPU or to LMem (ignoring the other)
 */
#define setupWriteData(name) do { \
	if (kind##name == STARPU_CPU_RAM) { \
		max_route(acts, "forkOut"#name, "outCPU"); \
		max_queue_output(acts, "out"#name"CPU", ptr##name, size * sizeof(uint32_t)); \
		max_ignore_memctl(acts, "MemoryControllerPro0", "out"#name"LMem"); \
	} else { \
		max_route(acts, "forkOut"#name, "outLMem"); \
		max_ignore_stream(acts, "out"#name"CPU"); \
		max_memctl_linear(acts, "MemoryControllerPro0", "out"#name"LMem", (size_t) ptr##name, size * sizeof(uint32_t)); \
	} \
} while (0)

/*
 * Ignore data from unused output
 */
#define ignoreWriteData(name) do { \
		max_route(acts, "forkOut"#name, "outLMem"); \
		max_ignore_stream(acts, "out"#name"CPU"); \
		max_ignore_memctl(acts, "MemoryControllerPro0", "out"#name"LMem"); \
} while (0)

void fpga_impl1(void *buffers[], void *cl_arg)
{
	(void)cl_arg;

	int32_t *ptrAT1 = (int32_t*) STARPU_VECTOR_GET_PTR(buffers[0]);
	int32_t *ptrBT1 = (int32_t*) STARPU_VECTOR_GET_PTR(buffers[1]);
	int32_t *ptrCT1 = (int32_t*) STARPU_VECTOR_GET_PTR(buffers[2]);
	enum starpu_node_kind kindAT1 = starpu_node_get_kind(starpu_task_get_current_data_node(0));
	enum starpu_node_kind kindBT1 = starpu_node_get_kind(starpu_task_get_current_data_node(1));
	enum starpu_node_kind kindCT1 = starpu_node_get_kind(starpu_task_get_current_data_node(2));

	int size = STARPU_VECTOR_GET_NX(buffers[0]);

	max_engine_t *engine = starpu_max_fpga_get_local_engine();;

	printf("T1 with %p %p %p\n", ptrAT1, ptrBT1, ptrCT1);
	/* C = A+B */

	max_actions_t *acts = max_actions_init(maxfile, NULL);
	max_set_ticks(acts, "Task1", size);
	max_ignore_scalar(acts, "Task2", "run_cycle_count");
	max_ignore_scalar(acts, "Task3", "run_cycle_count");

	setupReadData(AT1);
	setupReadData(BT1);
	setupWriteData(CT1);

	ignoreReadData(AT2);
	ignoreReadData(BT2);
	ignoreWriteData(CT2);

	ignoreReadData(AT3);
	ignoreReadData(BT3);
	ignoreWriteData(CT3);

	max_run(engine, acts);
	max_actions_free(acts);

	printf("T1 finished\n");
}

static struct starpu_codelet cl1 =
{
	.max_fpga_funcs = {fpga_impl1},
	.nbuffers = 3,
	.modes = {STARPU_R, STARPU_R, STARPU_W},
	.specific_nodes = 1,
	.nodes = {STARPU_SPECIFIC_NODE_LOCAL_OR_CPU, STARPU_SPECIFIC_NODE_LOCAL_OR_CPU, STARPU_SPECIFIC_NODE_LOCAL},
};

void fpga_impl2(void *buffers[], void *cl_arg)
{
	(void)cl_arg;

	int32_t *ptrAT2 = (int32_t*) STARPU_VECTOR_GET_PTR(buffers[0]);
	int32_t *ptrBT2 = (int32_t*) STARPU_VECTOR_GET_PTR(buffers[1]);
	int32_t *ptrCT2 = (int32_t*) STARPU_VECTOR_GET_PTR(buffers[2]);
	enum starpu_node_kind kindAT2 = starpu_node_get_kind(starpu_task_get_current_data_node(0));
	enum starpu_node_kind kindBT2 = starpu_node_get_kind(starpu_task_get_current_data_node(1));
	enum starpu_node_kind kindCT2 = starpu_node_get_kind(starpu_task_get_current_data_node(2));

	int size = STARPU_VECTOR_GET_NX(buffers[0]);

	max_engine_t *engine = starpu_max_fpga_get_local_engine();;

	printf("T2 with %p %p %p\n", ptrAT2, ptrBT2, ptrCT2);
	/* C = A*B */

	max_actions_t *acts = max_actions_init(maxfile, NULL);
	max_ignore_scalar(acts, "Task1", "run_cycle_count");
	max_set_ticks(acts, "Task2", size);
	max_ignore_scalar(acts, "Task3", "run_cycle_count");

	setupReadData(AT2);
	setupReadData(BT2);
	setupWriteData(CT2);

	ignoreReadData(AT1);
	ignoreReadData(BT1);
	ignoreWriteData(CT1);

	ignoreReadData(AT3);
	ignoreReadData(BT3);
	ignoreWriteData(CT3);

	max_run(engine, acts);
	max_actions_free(acts);

	printf("T2 finished\n");
}

static struct starpu_codelet cl2 =
{
	.max_fpga_funcs = {fpga_impl2},
	.nbuffers = 3,
	.modes = {STARPU_R, STARPU_R, STARPU_W},
	.specific_nodes = 1,
	.nodes = {STARPU_SPECIFIC_NODE_LOCAL_OR_CPU, STARPU_SPECIFIC_NODE_LOCAL_OR_CPU, STARPU_SPECIFIC_NODE_LOCAL},
};

void fpga_impl3(void *buffers[], void *cl_arg)
{
	(void)cl_arg;

	int32_t *ptrAT3 = (int32_t*) STARPU_VECTOR_GET_PTR(buffers[0]);
	int32_t *ptrBT3 = (int32_t*) STARPU_VECTOR_GET_PTR(buffers[1]);
	int32_t *ptrCT3 = (int32_t*) STARPU_VECTOR_GET_PTR(buffers[2]);
	enum starpu_node_kind kindAT3 = starpu_node_get_kind(starpu_task_get_current_data_node(0));
	enum starpu_node_kind kindBT3 = starpu_node_get_kind(starpu_task_get_current_data_node(1));
	enum starpu_node_kind kindCT3 = starpu_node_get_kind(starpu_task_get_current_data_node(2));

	int size = STARPU_VECTOR_GET_NX(buffers[0]);

	max_engine_t *engine = starpu_max_fpga_get_local_engine();;

	printf("T3 with %p %p %p\n", ptrAT3, ptrBT3, ptrCT3);
	/* C = A+B */

	max_actions_t *acts = max_actions_init(maxfile, NULL);
	max_ignore_scalar(acts, "Task1", "run_cycle_count");
	max_ignore_scalar(acts, "Task2", "run_cycle_count");
	max_set_ticks(acts, "Task3", size);

	setupReadData(AT3);
	setupReadData(BT3);
	setupWriteData(CT3);

	ignoreReadData(AT1);
	ignoreReadData(BT1);
	ignoreWriteData(CT1);

	ignoreReadData(AT2);
	ignoreReadData(BT2);
	ignoreWriteData(CT2);

	max_run(engine, acts);
	max_actions_free(acts);

	printf("T3 finished\n");
}

static struct starpu_codelet cl3 =
{
	.max_fpga_funcs = {fpga_impl3},
	.nbuffers = 3,
	.modes = {STARPU_R, STARPU_R, STARPU_W},
	.specific_nodes = 1,
	.nodes = {STARPU_SPECIFIC_NODE_LOCAL_OR_CPU, STARPU_SPECIFIC_NODE_LOCAL_OR_CPU, STARPU_SPECIFIC_NODE_CPU},
};

int main(int argc, char **argv)
{
	struct starpu_conf conf;
	starpu_data_handle_t handle_a, handle_b, handle_ct1, handle_ct2, handle_c;
	int ret;

	maxfile = MyTasksMux_init();

	struct starpu_max_load load[2];
	load[0].file = maxfile;
	load[0].engine_id_pattern = "*";
	load[1].file = NULL;
	load[1].engine_id_pattern = NULL;

	starpu_conf_init(&conf);
	conf.sched_policy_name = "eager";
	conf.calibrate = 0;
	conf.max_fpga_load = load;

	ret = starpu_initialize(&conf, &argc, &argv);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	/* Enable profiling */
	starpu_profiling_status_set(STARPU_PROFILING_ENABLE);

	int32_t a[SIZE];
	int32_t b[SIZE];
	int32_t c[SIZE];

	int i;
	for(i = 0; i < SIZE; ++i)
	{
		a[i] = random() % 100;
		b[i] = random() % 100;
	}

	starpu_vector_data_register(&handle_a, STARPU_MAIN_RAM, (uintptr_t) &a, SIZE, sizeof(a[0]));
	starpu_vector_data_register(&handle_b, STARPU_MAIN_RAM, (uintptr_t) &b, SIZE, sizeof(b[0]));

	starpu_vector_data_register(&handle_ct1, -1, 0, SIZE, sizeof(c[0]));
	starpu_vector_data_register(&handle_ct2, -1, 0, SIZE, sizeof(c[0]));

	starpu_vector_data_register(&handle_c, STARPU_MAIN_RAM, (uintptr_t) &c, SIZE, sizeof(c[0]));

	ret = starpu_task_insert(&cl1, STARPU_R, handle_a, STARPU_R, handle_b, STARPU_W, handle_ct1, 0);
	fprintf(stderr,"task submitted %d\n", ret);
	ret = starpu_task_insert(&cl2, STARPU_R, handle_ct1, STARPU_R, handle_ct1, STARPU_W, handle_ct2, 0);
	fprintf(stderr,"task submitted %d\n", ret);
	ret = starpu_task_insert(&cl3, STARPU_R, handle_ct2, STARPU_R, handle_ct2, STARPU_W, handle_c, 0);
	fprintf(stderr,"task submitted %d\n", ret);

	starpu_data_unregister(handle_a);
	starpu_data_unregister(handle_b);
	starpu_data_unregister(handle_c);

	ret = EXIT_SUCCESS;

	for (i = 0; i < SIZE; ++i)
	{
		int ct1 = a[i] + b[i];
		int ct2 = ct1 * ct1;
		int ct3 = ct2 + ct2;

		if (c[i] != ct3)
			ret = EXIT_FAILURE;

		if (i < 10)
		{
			printf("%d == %d\n", c[i], ct3);
			if (c[i] != ct3)
				printf("OOOPS\n");
		}
	}

	starpu_shutdown();

	if (ret == EXIT_SUCCESS)
		printf("OK!\n");

	return ret;
}
