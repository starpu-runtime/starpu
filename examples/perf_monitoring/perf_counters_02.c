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
#include <assert.h>
#include <inttypes.h>

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)

/* global counters */
static int id_g_total_submitted;
static int id_g_peak_submitted;
static int id_g_peak_ready;

/* per worker counters */
static int id_w_total_executed;
static int id_w_cumul_execution_time;

/* per_codelet counters */
static int id_c_total_submitted;
static int id_c_peak_submitted;
static int id_c_peak_ready;
static int id_c_total_executed;
static int id_c_cumul_execution_time;

void g_listener_cb(struct starpu_perf_counter_listener *listener, struct starpu_perf_counter_sample *sample, void *context)
{
	(void) listener;
	(void) context;
	int64_t g_total_submitted = starpu_perf_counter_sample_get_int64_value(sample, id_g_total_submitted);
	int64_t g_peak_submitted = starpu_perf_counter_sample_get_int64_value(sample, id_g_peak_submitted);
	int64_t g_peak_ready = starpu_perf_counter_sample_get_int64_value(sample, id_g_peak_ready);
	printf("global: g_total_submitted = %"PRId64", g_peak_submitted = %"PRId64", g_peak_ready = %"PRId64"\n", g_total_submitted, g_peak_submitted, g_peak_ready);
}

void w_listener_cb(struct starpu_perf_counter_listener *listener, struct starpu_perf_counter_sample *sample, void *context)
{
	(void) listener;
	(void) context;
	int workerid = starpu_worker_get_id();
	int64_t w_total_executed = starpu_perf_counter_sample_get_int64_value(sample, id_w_total_executed);
	double w_cumul_execution_time = starpu_perf_counter_sample_get_double_value(sample, id_w_cumul_execution_time);

	printf("worker[%d]: w_total_executed = %"PRId64", w_cumul_execution_time = %lf\n", workerid, w_total_executed, w_cumul_execution_time);
}

void c_listener_cb(struct starpu_perf_counter_listener *listener, struct starpu_perf_counter_sample *sample, void *context)
{
	(void) listener;
	struct starpu_codelet *cl = context;
	int64_t c_total_submitted = starpu_perf_counter_sample_get_int64_value(sample, id_c_total_submitted);
	int64_t c_peak_submitted = starpu_perf_counter_sample_get_int64_value(sample, id_c_peak_submitted);
	int64_t c_peak_ready = starpu_perf_counter_sample_get_int64_value(sample, id_c_peak_ready);
	int64_t c_total_executed = starpu_perf_counter_sample_get_int64_value(sample, id_c_total_executed);
	double c_cumul_execution_time = starpu_perf_counter_sample_get_double_value(sample, id_c_cumul_execution_time);
	if (cl->name != NULL)
	{
		printf("codelet[%s]: c_total_submitted = %"PRId64", c_peak_submitted = %"PRId64", c_peak_ready = %"PRId64", c_total_executed = %"PRId64", c_cumul_execution_time = %lf\n", cl->name, c_total_submitted, c_peak_submitted, c_peak_ready, c_total_executed, c_cumul_execution_time);
	}
	else
	{
		printf("codelet[%p]: c_total_submitted = %"PRId64", c_peak_submitted = %"PRId64", c_peak_ready = %"PRId64", c_total_executed = %"PRId64", c_cumul_execution_time = %lf\n", cl, c_total_submitted, c_peak_submitted, c_peak_ready, c_total_executed, c_cumul_execution_time);
	}
}

void func(void *buffers[], void *cl_args)
{
	int *int_vector = (int*)STARPU_VECTOR_GET_PTR(buffers[0]);
	int NX = (int)STARPU_VECTOR_GET_NX(buffers[0]);
	const int niters;
	starpu_codelet_unpack_args(cl_args, &niters);
	int i;
	for (i=0; i<niters; i++)
	{
		int_vector[i % NX] += i;
	}
}

struct starpu_codelet cl =
{
	.cpu_funcs      = {func},
	.cpu_funcs_name = {"func"},
	.nbuffers       = 1,
	.name           = "perf_counter_f"
};

const enum starpu_perf_counter_scope g_scope = starpu_perf_counter_scope_global;
const enum starpu_perf_counter_scope w_scope = starpu_perf_counter_scope_per_worker;
const enum starpu_perf_counter_scope c_scope = starpu_perf_counter_scope_per_codelet;

#define NVECTORS 5
#define NTASKS 1000
#define NITER 1000
#define VECTOR_LEN 2

int main(int argc, char **argv)
{
	struct starpu_conf conf;
	starpu_conf_init(&conf);

	/* Start collecting perfomance counter right after initialization */
	conf.start_perf_counter_collection = 1;

	int ret;
	ret = starpu_init(&conf);
	if (ret == -ENODEV)
		return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	struct starpu_perf_counter_set *g_set = starpu_perf_counter_set_alloc(g_scope);
	STARPU_ASSERT(g_set != NULL);
	struct starpu_perf_counter_set *w_set = starpu_perf_counter_set_alloc(w_scope);
	STARPU_ASSERT(w_set != NULL);
	struct starpu_perf_counter_set *c_set = starpu_perf_counter_set_alloc(c_scope);
	STARPU_ASSERT(c_set != NULL);

	id_g_total_submitted = starpu_perf_counter_name_to_id(g_scope, "starpu.task.g_total_submitted");
	STARPU_ASSERT(id_g_total_submitted != -1);
	id_g_peak_submitted = starpu_perf_counter_name_to_id(g_scope, "starpu.task.g_peak_submitted");
	STARPU_ASSERT(id_g_peak_submitted != -1);
	id_g_peak_ready = starpu_perf_counter_name_to_id(g_scope, "starpu.task.g_peak_ready");
	STARPU_ASSERT(id_g_peak_ready != -1);


	id_w_total_executed = starpu_perf_counter_name_to_id(w_scope, "starpu.task.w_total_executed");
	STARPU_ASSERT(id_w_total_executed != -1);
	id_w_cumul_execution_time = starpu_perf_counter_name_to_id(w_scope, "starpu.task.w_cumul_execution_time");
	STARPU_ASSERT(id_w_cumul_execution_time != -1);

	id_c_total_submitted = starpu_perf_counter_name_to_id(c_scope, "starpu.task.c_total_submitted");
	STARPU_ASSERT(id_c_total_submitted != -1);
	id_c_peak_submitted = starpu_perf_counter_name_to_id(c_scope, "starpu.task.c_peak_submitted");
	STARPU_ASSERT(id_c_peak_submitted != -1);
	id_c_peak_ready = starpu_perf_counter_name_to_id(c_scope, "starpu.task.c_peak_ready");
	STARPU_ASSERT(id_c_peak_ready != -1);
	id_c_total_executed = starpu_perf_counter_name_to_id(c_scope, "starpu.task.c_total_executed");
	STARPU_ASSERT(id_c_total_executed != -1);
	id_c_cumul_execution_time = starpu_perf_counter_name_to_id(c_scope, "starpu.task.c_cumul_execution_time");
	STARPU_ASSERT(id_c_cumul_execution_time != -1);

	starpu_perf_counter_set_enable_id(g_set, id_g_total_submitted);
	starpu_perf_counter_set_enable_id(g_set, id_g_peak_submitted);
	starpu_perf_counter_set_enable_id(g_set, id_g_peak_ready);

	starpu_perf_counter_set_enable_id(w_set, id_w_total_executed);
	starpu_perf_counter_set_enable_id(w_set, id_w_cumul_execution_time);

	starpu_perf_counter_set_enable_id(c_set, id_c_total_submitted);
	starpu_perf_counter_set_enable_id(c_set, id_c_peak_submitted);
	starpu_perf_counter_set_enable_id(c_set, id_c_peak_ready);
	starpu_perf_counter_set_enable_id(c_set, id_c_total_executed);
	starpu_perf_counter_set_enable_id(c_set, id_c_cumul_execution_time);

	struct starpu_perf_counter_listener * g_listener = starpu_perf_counter_listener_init(g_set, g_listener_cb, (void *)(uintptr_t)42);
	struct starpu_perf_counter_listener * w_listener = starpu_perf_counter_listener_init(w_set, w_listener_cb, (void *)(uintptr_t)17);
	struct starpu_perf_counter_listener * c_listener = starpu_perf_counter_listener_init(c_set, c_listener_cb, (void *)(uintptr_t)76);

	starpu_perf_counter_set_global_listener(g_listener);
	starpu_perf_counter_set_all_per_worker_listeners(w_listener);

	starpu_perf_counter_set_per_codelet_listener(&cl, c_listener);

	int* vector[NVECTORS];
	starpu_data_handle_t vector_h[NVECTORS];
	int v;
	for (v=0; v<NVECTORS; v++)
	{
		vector[v] = calloc(VECTOR_LEN, sizeof(*(vector[v])));
		STARPU_ASSERT(vector[v] != NULL);

		{
			int i;
			for (i=0; i<VECTOR_LEN; i++)
			{
				vector[v][i] = i;
			}
		}

		starpu_vector_data_register(&vector_h[v], STARPU_MAIN_RAM, (uintptr_t)vector[v], VECTOR_LEN, sizeof(*vector[v]));
	}

	{
		int i;
		for (i=0; i<NTASKS; i++)
		{
			v = i % NVECTORS;
			const int niter = NITER;
			ret = starpu_task_insert(&cl,
						 STARPU_RW, vector_h[v],
						 STARPU_VALUE, &niter, sizeof(int),
						 0);
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
		}
	}

	for (v=0; v<NVECTORS; v++)
	{
		starpu_data_unregister(vector_h[v]);
		free(vector[v]);
	}

	starpu_perf_counter_unset_per_codelet_listener(&cl);
	starpu_perf_counter_unset_all_per_worker_listeners();
	starpu_perf_counter_unset_global_listener();

	starpu_perf_counter_listener_exit(c_listener);
	starpu_perf_counter_listener_exit(w_listener);
	starpu_perf_counter_listener_exit(g_listener);

	starpu_perf_counter_set_disable_id(c_set, id_c_cumul_execution_time);
	starpu_perf_counter_set_disable_id(c_set, id_c_total_executed);
	starpu_perf_counter_set_disable_id(c_set, id_c_peak_ready);
	starpu_perf_counter_set_disable_id(c_set, id_c_peak_submitted);
	starpu_perf_counter_set_disable_id(c_set, id_c_total_submitted);

	starpu_perf_counter_set_disable_id(w_set, id_w_cumul_execution_time);
	starpu_perf_counter_set_disable_id(w_set, id_w_total_executed);

	starpu_perf_counter_set_disable_id(g_set, id_g_peak_ready);
	starpu_perf_counter_set_disable_id(g_set, id_g_peak_submitted);
	starpu_perf_counter_set_disable_id(g_set, id_g_total_submitted);

	starpu_perf_counter_set_free(c_set);
	c_set = NULL;

	starpu_perf_counter_set_free(w_set);
	w_set = NULL;

	starpu_perf_counter_set_free(g_set);
	g_set = NULL;

	starpu_shutdown();

	return 0;
}
