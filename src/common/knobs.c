/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2019                                     Inria
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

/* Performance counters and configurable knobs */

#include <stdlib.h>
#include <stdint.h>
#include <starpu.h>
#include <common/config.h>
#include <common/starpu_spinlock.h>
#include <core/workers.h>
#include <common/knobs.h>

struct perf_counter_array
{
	int size;
	struct starpu_perf_counter *array;
	int updater_array_size;
	void (**updater_array)(struct starpu_perf_counter_sample *sample, void *context);
};

static struct perf_counter_array global_counters	= { .size = 0, .array = NULL, .updater_array_size = 0, .updater_array = NULL };
static struct perf_counter_array per_worker_counters	= { .size = 0, .array = NULL, .updater_array_size = 0, .updater_array = NULL };
static struct perf_counter_array per_codelet_counters	= { .size = 0, .array = NULL, .updater_array_size = 0, .updater_array = NULL };

static struct starpu_perf_counter_sample global_sample	= { .scope = starpu_perf_counter_scope_global, .listener = NULL, .value_array = NULL };

/* - */

void _starpu_perf_counter_sample_init(struct starpu_perf_counter_sample *sample, enum starpu_perf_counter_scope scope)
{
	STARPU_ASSERT_PERF_COUNTER_SCOPE_DEFINED(scope);
	sample->scope = scope;
	sample->listener = NULL;
	sample->value_array = NULL;
	_starpu_spin_init(&sample->lock);
}

void _starpu_perf_counter_sample_exit(struct starpu_perf_counter_sample *sample)
{
	STARPU_ASSERT(sample->listener == NULL);
	sample->listener = NULL;
	if (sample->value_array)
	{
		free(sample->value_array);
	}
	sample->value_array = NULL;
	sample->scope = starpu_perf_counter_scope_undefined;
	_starpu_spin_destroy(&sample->lock);
}

/* - */

void _starpu_perf_counter_init(void)
{
	STARPU_ASSERT(!_starpu_machine_is_running());
	_starpu_perf_counter_sample_init(&global_sample, starpu_perf_counter_scope_global);

	/* call counter registration routines in each modules */
	_starpu__task_c__register_counters();
}

void _starpu_perf_counter_exit(void)
{
	STARPU_ASSERT(!_starpu_machine_is_running());

	_starpu_perf_counter_unregister_all_scopes();
	_starpu_perf_counter_sample_exit(&global_sample);
}

/* - */

int starpu_perf_counter_scope_name_to_id(const char * const name)
{
	if (strcmp(name, "global") == 0)
		return starpu_perf_counter_scope_global;
	if (strcmp(name, "per_worker") == 0)
		return starpu_perf_counter_scope_per_worker;
	if (strcmp(name, "per_codelet") == 0)
		return starpu_perf_counter_scope_per_codelet;
	return -1;
}

const char *starpu_perf_counter_scope_id_to_name(const enum starpu_perf_counter_scope scope)
{
	switch (scope)
	{
		case starpu_perf_counter_scope_global:
			return "global";

		case starpu_perf_counter_scope_per_worker:
			return "per_worker";

		case starpu_perf_counter_scope_per_codelet:
			return "per_codelet";

		default:
			return NULL;
	};
}

/* - */

int starpu_perf_counter_type_name_to_id(const char * const name)
{
	if (strcmp(name, "int32") == 0)
		return starpu_perf_counter_type_int32;
	if (strcmp(name, "int64") == 0)
		return starpu_perf_counter_type_int64;
	if (strcmp(name, "float") == 0)
		return starpu_perf_counter_type_float;
	if (strcmp(name, "double") == 0)
		return starpu_perf_counter_type_double;
	return -1;
}

const char *starpu_perf_counter_type_id_to_name(const enum starpu_perf_counter_type type)
{
	switch (type)
	{
		case starpu_perf_counter_type_int32:
			return "int32";

		case starpu_perf_counter_type_int64:
			return "int64";

		case starpu_perf_counter_type_float:
			return "float";

		case starpu_perf_counter_type_double:
			return "double";

		default:
			return NULL;
	};
}

static struct perf_counter_array *_get_counters(const enum starpu_perf_counter_scope scope)
{
	STARPU_ASSERT_PERF_COUNTER_SCOPE_DEFINED(scope);
	switch (scope)
	{
		case starpu_perf_counter_scope_global:
			return &global_counters;

		case starpu_perf_counter_scope_per_worker:
			return &per_worker_counters;

		case starpu_perf_counter_scope_per_codelet:
			return &per_codelet_counters;

		default:
			STARPU_ABORT();
	};
};

/* - */

int _starpu_perf_counter_register(enum starpu_perf_counter_scope scope, const char *name, enum starpu_perf_counter_type type, const char *help)
{
	STARPU_ASSERT(!_starpu_machine_is_running());

	struct perf_counter_array * const counters = _get_counters(scope);
	STARPU_ASSERT_PERF_COUNTER_TYPE_DEFINED(type);

	const int index = counters->size++;
	_STARPU_REALLOC(counters->array, counters->size * sizeof(*counters->array));

	struct starpu_perf_counter * const new_counter = &counters->array[index];
	const int id = _starpu_perf_counter_id_build(scope, index);
	new_counter->id = id;
	new_counter->name = name;
	new_counter->help = help;
	new_counter->type = type;

	return id;
}

static void _unregister_scope(enum starpu_perf_counter_scope scope)
{
	STARPU_ASSERT(!_starpu_machine_is_running());

	struct perf_counter_array * const counters = _get_counters(scope);
	free(counters->array);
	counters->array = NULL;
	counters->size  = 0;
}

void _starpu_perf_counter_unregister_all_scopes(void)
{
	STARPU_ASSERT(!_starpu_machine_is_running());

	_unregister_scope(starpu_perf_counter_scope_global);
	_unregister_scope(starpu_perf_counter_scope_per_worker);
	_unregister_scope(starpu_perf_counter_scope_per_codelet);
}

/* - */

int starpu_perf_counter_nb(enum starpu_perf_counter_scope scope)
{
	const struct perf_counter_array * const counters = _get_counters(scope);
	return counters->size;
}

int starpu_perf_counter_nth_to_id(enum starpu_perf_counter_scope scope, int nth)
{
	return _starpu_perf_counter_id_build(scope, nth);
}

int starpu_perf_counter_name_to_id(enum starpu_perf_counter_scope scope, const char *name)
{
	const struct perf_counter_array * const counters = _get_counters(scope);
	int index;
	for (index = 0; index < counters->size; index++)
	{
		if (strcmp(name, counters->array[index].name) == 0)
		{
			return _starpu_perf_counter_id_build(scope, index);
		}
	}
	return -1;
}

const char *starpu_perf_counter_id_to_name(int id)
{
	const int scope = _starpu_perf_counter_id_get_scope(id);
	const int index = _starpu_perf_counter_id_get_index(id);
	const struct perf_counter_array * const counters = _get_counters(scope);
	if (index < 0 || index >= counters->size)
		return NULL;
	return counters->array[index].name;
}

const char *starpu_perf_counter_get_help_string(int id)
{
	const int scope = _starpu_perf_counter_id_get_scope(id);
	const int index = _starpu_perf_counter_id_get_index(id);
	const struct perf_counter_array * const counters = _get_counters(scope);
	STARPU_ASSERT(index >= 0 && index < counters->size);
	return counters->array[index].help;
}

int starpu_perf_counter_get_type_id(int id)
{
	const int scope = _starpu_perf_counter_id_get_scope(id);
	const int index = _starpu_perf_counter_id_get_index(id);
	const struct perf_counter_array * const counters = _get_counters(scope);
	STARPU_ASSERT(index >= 0 && index < counters->size);
	return counters->array[index].type;
}

/* - */

void starpu_perf_counter_list_avail(enum starpu_perf_counter_scope scope)
{
	const struct perf_counter_array * const counters = _get_counters(scope);
	int index;
	for (index = 0; index < counters->size; index++)
	{
		const struct starpu_perf_counter * const counter = &counters->array[index];
		printf("0x%08x:%s [%s] - %s\n", _starpu_perf_counter_id_build(scope, index), counter->name, starpu_perf_counter_type_id_to_name(counter->type), counter->help);
	}
}

void starpu_perf_counter_list_all_avail(enum starpu_perf_counter_scope scope)
{
	printf("scope: global\n");
	starpu_perf_counter_list_avail(starpu_perf_counter_scope_global);

	printf("scope: per_worker\n");
	starpu_perf_counter_list_avail(starpu_perf_counter_scope_per_worker);

	printf("scope: per_codelet\n");
	starpu_perf_counter_list_avail(starpu_perf_counter_scope_per_codelet);
}

/* - */

struct starpu_perf_counter_set *starpu_perf_counter_set_alloc(enum starpu_perf_counter_scope scope)
{
	struct perf_counter_array *counters = _get_counters(scope);
	struct starpu_perf_counter_set *set;
	_STARPU_MALLOC(set, sizeof(*set));
	set->scope = scope;
	set->size  = counters->size;
	_STARPU_CALLOC(set->index_array, set->size, sizeof(*set->index_array));
	return set;
}

void starpu_perf_counter_set_free(struct starpu_perf_counter_set *set)
{
	memset(set->index_array, 0, set->size*sizeof(*set->index_array));
	free(set->index_array);
	memset(set, 0, sizeof(*set));
	free(set);
}

/* - */

void starpu_perf_counter_set_enable_id(struct starpu_perf_counter_set *set, int id)
{
	const int index = _starpu_perf_counter_id_get_index(id);
	STARPU_ASSERT(index >= 0 && index < set->size);
	set->index_array[index] = 1;
}

void starpu_perf_counter_set_disable_id(struct starpu_perf_counter_set *set, int id)
{
	const int index = _starpu_perf_counter_id_get_index(id);
	STARPU_ASSERT(index >= 0 && index < set->size);
	set->index_array[index] = 0;
}

/* - */

struct starpu_perf_counter_listener *starpu_perf_counter_listener_init(struct starpu_perf_counter_set *set,
		void (*callback)(struct starpu_perf_counter_listener *listener, struct starpu_perf_counter_sample *sample, void *context),
		void *user_arg)
{
	struct starpu_perf_counter_listener *listener;
	_STARPU_MALLOC(listener, sizeof(*listener));
	listener->set = set;
	listener->callback = callback;
	listener->user_arg = user_arg;
	return listener;
}

void starpu_perf_counter_listener_exit(struct starpu_perf_counter_listener *listener)
{
	memset(listener, 0, sizeof(*listener));
	free(listener);
}

/* - */

static void set_listener(struct starpu_perf_counter_sample *sample, struct starpu_perf_counter_listener *listener)
{
	_starpu_spin_lock(&sample->lock);
	STARPU_ASSERT(sample->listener == NULL);

	STARPU_ASSERT(listener->set != NULL);
	STARPU_ASSERT(listener->set->scope == sample->scope);

	sample->listener = listener;

	/* Assume a single listener, for now, which sets the set of counters to monitor */
	STARPU_ASSERT(sample->value_array == NULL);
	_STARPU_CALLOC(sample->value_array, sample->listener->set->size, sizeof(*sample->value_array));
	_starpu_spin_unlock(&sample->lock);
}


void starpu_perf_counter_set_global_listener(struct starpu_perf_counter_listener *listener)
{
	set_listener(&global_sample, listener);
}

void starpu_perf_counter_set_per_worker_listener(unsigned workerid, struct starpu_perf_counter_listener *listener)
{
	struct _starpu_worker *worker = _starpu_get_worker_struct(workerid);
	set_listener(&worker->perf_counter_sample, listener);
}

void starpu_perf_counter_set_all_per_worker_listeners(struct starpu_perf_counter_listener *listener)
{
	unsigned nworkers = _starpu_worker_get_count();
	unsigned workerid;
	for (workerid = 0; workerid < nworkers; workerid++)
	{
		starpu_perf_counter_set_per_worker_listener(workerid, listener);
	}
}

void starpu_perf_counter_set_per_codelet_listener(struct starpu_codelet *cl, struct starpu_perf_counter_listener *listener)
{
	STARPU_ASSERT(cl->perf_counter_values == NULL);
	_STARPU_CALLOC(cl->perf_counter_values, 1, sizeof(*cl->perf_counter_values));

	STARPU_ASSERT(cl->perf_counter_sample == NULL);
	_STARPU_MALLOC(cl->perf_counter_sample, sizeof(*cl->perf_counter_sample));
	_starpu_perf_counter_sample_init(cl->perf_counter_sample, starpu_perf_counter_scope_per_codelet);
	set_listener(cl->perf_counter_sample, listener);
}

/* - */

void unset_listener(struct starpu_perf_counter_sample *sample)
{
	_starpu_spin_lock(&sample->lock);
	STARPU_ASSERT(sample->listener != NULL);

	memset(sample->value_array, 0, sample->listener->set->size * sizeof(*sample->value_array));
	free(sample->value_array);
	sample->value_array = NULL;
	sample->listener = NULL;
	_starpu_spin_unlock(&sample->lock);
}

void starpu_perf_counter_unset_global_listener()
{
	unset_listener(&global_sample);
}

void starpu_perf_counter_unset_per_worker_listener(unsigned workerid)
{
	struct _starpu_worker *worker = _starpu_get_worker_struct(workerid);
	unset_listener(&worker->perf_counter_sample);
}

void starpu_perf_counter_unset_all_per_worker_listeners(void)
{
	unsigned nworkers = _starpu_worker_get_count();
	unsigned workerid;
	for (workerid = 0; workerid < nworkers; workerid++)
	{
		starpu_perf_counter_unset_per_worker_listener(workerid);
	}
}

void starpu_perf_counter_unset_per_codelet_listener(struct starpu_codelet *cl)
{
	STARPU_ASSERT(cl->perf_counter_sample != NULL);
	unset_listener(cl->perf_counter_sample);
	_starpu_perf_counter_sample_exit(cl->perf_counter_sample);
	free(cl->perf_counter_sample);
	cl->perf_counter_sample = NULL;
	free(cl->perf_counter_values);
	cl->perf_counter_values = NULL;
}

/* - */

void _starpu_perf_counter_register_updater(enum starpu_perf_counter_scope scope, void (*updater)(struct starpu_perf_counter_sample *sample, void *context))
{
	STARPU_ASSERT(!_starpu_machine_is_running());

	struct perf_counter_array *counters = _get_counters(scope);
	int upd_id;
	upd_id = counters->updater_array_size++;
	_STARPU_REALLOC(counters->updater_array, counters->updater_array_size * sizeof(*counters->updater_array));
	counters->updater_array[upd_id] = updater;
}

/* - */

static void update_sample(struct starpu_perf_counter_sample *sample, void *context)
{
	_starpu_spin_lock(&sample->lock);
	struct perf_counter_array *counters = _get_counters(sample->scope);

	/* for now, we assume that a sample will only be updated if it has a listener plugged, with a non-empty set */
	if (sample->listener != NULL && sample->listener->set != NULL)
	{
		if (counters->updater_array_size > 0)
		{
			int upd_id;
			for (upd_id = 0; upd_id < counters->updater_array_size; upd_id++)
			{
				counters->updater_array[upd_id](sample, context);
			}

			if (sample->listener != NULL)
			{
				sample->listener->callback(sample->listener, sample, context);
			}
		}
	}
	_starpu_spin_unlock(&sample->lock);
}

void _starpu_perf_counter_update_global_sample(void)
{
	update_sample(&global_sample, NULL);
}

void _starpu_perf_counter_update_per_worker_sample(unsigned workerid)
{
	struct _starpu_worker *worker = _starpu_get_worker_struct(workerid);
	update_sample(&worker->perf_counter_sample, worker);
}

void _starpu_perf_counter_update_per_codelet_sample(struct starpu_codelet *cl)
{
	update_sample(cl->perf_counter_sample, cl);
}

#define STARPU_PERF_COUNTER_SAMPLE_GET_TYPED_VALUE(STRING, TYPE) \
TYPE starpu_perf_counter_sample_get_##STRING##_value(struct starpu_perf_counter_sample *sample, const int counter_id) \
{ \
	STARPU_ASSERT(starpu_perf_counter_get_type_id(counter_id) == starpu_perf_counter_type_##STRING); \
	STARPU_ASSERT(sample->listener != NULL && sample->listener->set != NULL); \
	STARPU_ASSERT(_starpu_perf_counter_id_get_scope(counter_id) == sample->listener->set->scope); \
 \
	const struct starpu_perf_counter_set * const set = sample->listener->set; \
	const int index =  _starpu_perf_counter_id_get_index(counter_id); \
	STARPU_ASSERT(index < set->size); \
	STARPU_ASSERT(set->index_array[index] > 0); \
	return sample->value_array[index].STRING##_val; \
}
STARPU_PERF_COUNTER_SAMPLE_GET_TYPED_VALUE(int32, int32_t);
STARPU_PERF_COUNTER_SAMPLE_GET_TYPED_VALUE(int64, int64_t);
STARPU_PERF_COUNTER_SAMPLE_GET_TYPED_VALUE(float, float);
STARPU_PERF_COUNTER_SAMPLE_GET_TYPED_VALUE(double, double);
#undef STARPU_PERF_COUNTER_SAMPLE_GET_TYPED_VALUE

