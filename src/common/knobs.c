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

/* Performance counters and configurable knobs */

#include <stdlib.h>
#include <stdint.h>
#include <starpu.h>
#include <common/config.h>
#include <common/starpu_spinlock.h>
#include <core/workers.h>
#include <common/knobs.h>

/* Performance Monitoring */
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

void _starpu_perf_counter_init(struct _starpu_machine_config *pconfig)
{
	if (pconfig->conf.start_perf_counter_collection)
	{
		/* start perf counter collection immediately */
		pconfig->perf_counter_pause_depth = 0;
	}
	else
	{
		/* defer perf counter collection until call to
		 * starpu_perf_counter_start_collection () */
		pconfig->perf_counter_pause_depth = 1;
	}
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

void starpu_perf_counter_collection_start()
{
	STARPU_HG_DISABLE_CHECKING(_starpu_config.perf_counter_pause_depth);
	(void)STARPU_ATOMIC_ADD(&_starpu_config.perf_counter_pause_depth, -1);
}

void starpu_perf_counter_collection_stop()
{
	STARPU_HG_DISABLE_CHECKING(_starpu_config.perf_counter_pause_depth);
	(void)STARPU_ATOMIC_ADD(&_starpu_config.perf_counter_pause_depth, +1);
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

static void _unregister_counter_scope(enum starpu_perf_counter_scope scope)
{
	STARPU_ASSERT(!_starpu_machine_is_running());

	struct perf_counter_array * const counters = _get_counters(scope);
	free(counters->array);
	counters->array = NULL;
	free(counters->updater_array);
	counters->updater_array = NULL;
	counters->size  = 0;
}

void _starpu_perf_counter_unregister_all_scopes(void)
{
	STARPU_ASSERT(!_starpu_machine_is_running());

	_unregister_counter_scope(starpu_perf_counter_scope_global);
	_unregister_counter_scope(starpu_perf_counter_scope_per_worker);
	_unregister_counter_scope(starpu_perf_counter_scope_per_codelet);
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

void starpu_perf_counter_list_all_avail(void)
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

static void unset_listener(struct starpu_perf_counter_sample *sample)
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
	if (sample->listener == NULL)
		return;

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
	\								\
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

/* -------------------------------------------------------------------- */
/* Performance Steering */

struct perf_knob_array
{
	int size;
	struct starpu_perf_knob *array;
};

static struct perf_knob_array global_knobs	= { .size = 0, .array = NULL };
static struct perf_knob_array per_worker_knobs	= { .size = 0, .array = NULL };
static struct perf_knob_array per_scheduler_knobs	= { .size = 0, .array = NULL };

void _starpu_perf_knob_init(void)
{
	STARPU_ASSERT(!_starpu_machine_is_running());
	/* call knob registration routines in each modules */
	_starpu__workers_c__register_knobs();
	_starpu__task_c__register_knobs();
	_starpu__dmda_c__register_knobs();
}

void _starpu_perf_knob_exit(void)
{
	STARPU_ASSERT(!_starpu_machine_is_running());

	_starpu_perf_knob_unregister_all_scopes();
	_starpu__workers_c__unregister_knobs();
	_starpu__task_c__unregister_knobs();
	_starpu__dmda_c__unregister_knobs();
}

/* - */

int starpu_perf_knob_scope_name_to_id(const char * const name)
{
	if (strcmp(name, "global") == 0)
		return starpu_perf_knob_scope_global;
	if (strcmp(name, "per_worker") == 0)
		return starpu_perf_knob_scope_per_worker;
	if (strcmp(name, "per_scheduler") == 0)
		return starpu_perf_knob_scope_per_scheduler;
	return -1;
}

const char *starpu_perf_knob_scope_id_to_name(const enum starpu_perf_knob_scope scope)
{
	switch (scope)
	{
		case starpu_perf_knob_scope_global:
			return "global";

		case starpu_perf_knob_scope_per_worker:
			return "per_worker";

		case starpu_perf_knob_scope_per_scheduler:
			return "per_scheduler";

		default:
			return NULL;
	};
}

/* - */

int starpu_perf_knob_type_name_to_id(const char * const name)
{
	if (strcmp(name, "int32") == 0)
		return starpu_perf_knob_type_int32;
	if (strcmp(name, "int64") == 0)
		return starpu_perf_knob_type_int64;
	if (strcmp(name, "float") == 0)
		return starpu_perf_knob_type_float;
	if (strcmp(name, "double") == 0)
		return starpu_perf_knob_type_double;
	return -1;
}

const char *starpu_perf_knob_type_id_to_name(const enum starpu_perf_knob_type type)
{
	switch (type)
	{
		case starpu_perf_knob_type_int32:
			return "int32";

		case starpu_perf_knob_type_int64:
			return "int64";

		case starpu_perf_knob_type_float:
			return "float";

		case starpu_perf_knob_type_double:
			return "double";

		default:
			return NULL;
	};
}

static struct perf_knob_array *_get_knobs(const enum starpu_perf_knob_scope scope)
{
	STARPU_ASSERT_PERF_KNOB_SCOPE_DEFINED(scope);
	switch (scope)
	{
		case starpu_perf_knob_scope_global:
			return &global_knobs;

		case starpu_perf_knob_scope_per_worker:
			return &per_worker_knobs;

		case starpu_perf_knob_scope_per_scheduler:
			return &per_scheduler_knobs;

		default:
			STARPU_ABORT();
	};
};

/* - */

struct starpu_perf_knob_group *_starpu_perf_knob_group_register(
	enum starpu_perf_knob_scope scope,
	void (*set_func)(const struct starpu_perf_knob * const knob, void *context, const struct starpu_perf_knob_value * const value),
	void (*get_func)(const struct starpu_perf_knob * const knob, void *context,       struct starpu_perf_knob_value * const value))
{
	STARPU_ASSERT_PERF_KNOB_SCOPE_DEFINED(scope);
	STARPU_ASSERT(set_func != NULL);
	STARPU_ASSERT(get_func != NULL);
	struct starpu_perf_knob_group *new_group;
	_STARPU_MALLOC(new_group, sizeof(*new_group));
	new_group->scope = scope;
	new_group->set = set_func;
	new_group->get = get_func;
	new_group->array_size = 0;
	new_group->array = NULL;
	return new_group;
}

void _starpu_perf_knob_group_unregister(struct starpu_perf_knob_group *group)
{
	STARPU_ASSERT((group->array_size > 0 && group->array != NULL)  ||  (group->array_size = 0 && group->array == NULL));
	if (group->array != NULL)
	{
		free(group->array);
	}
	memset(group, 0, sizeof(*group));
	free(group);
}

/* - */

int _starpu_perf_knob_register(struct starpu_perf_knob_group *group, const char *name, enum starpu_perf_knob_type type, const char *help)
{
	STARPU_ASSERT(!_starpu_machine_is_running());

	struct perf_knob_array * const knobs = _get_knobs(group->scope);
	STARPU_ASSERT_PERF_KNOB_TYPE_DEFINED(type);

	const int index = knobs->size++;
	_STARPU_REALLOC(knobs->array, knobs->size * sizeof(*knobs->array));

	struct starpu_perf_knob * const new_knob = &knobs->array[index];
	const int id = _starpu_perf_knob_id_build(group->scope, index);
	new_knob->id = id;
	new_knob->name = name;
	new_knob->help = help;
	new_knob->type = type;
	new_knob->group = group;
	new_knob->id_in_group = group->array_size++;
	_STARPU_REALLOC(group->array, group->array_size * sizeof(*group->array));
	group->array[new_knob->id_in_group] = new_knob;
	return id;
}

static void _unregister_knob_scope(enum starpu_perf_knob_scope scope)
{
	STARPU_ASSERT(!_starpu_machine_is_running());

	struct perf_knob_array * const knobs = _get_knobs(scope);
	free(knobs->array);
	knobs->array = NULL;
	knobs->size  = 0;
}

void _starpu_perf_knob_unregister_all_scopes(void)
{
	STARPU_ASSERT(!_starpu_machine_is_running());

	_unregister_knob_scope(starpu_perf_knob_scope_global);
	_unregister_knob_scope(starpu_perf_knob_scope_per_worker);
	_unregister_knob_scope(starpu_perf_knob_scope_per_scheduler);
}

/* - */

int starpu_perf_knob_nb(enum starpu_perf_knob_scope scope)
{
	const struct perf_knob_array * const knobs = _get_knobs(scope);
	return knobs->size;
}

int starpu_perf_knob_nth_to_id(enum starpu_perf_knob_scope scope, int nth)
{
	return _starpu_perf_knob_id_build(scope, nth);
}

int starpu_perf_knob_name_to_id(enum starpu_perf_knob_scope scope, const char *name)
{
	const struct perf_knob_array * const knobs = _get_knobs(scope);
	int index;
	for (index = 0; index < knobs->size; index++)
	{
		if (strcmp(name, knobs->array[index].name) == 0)
		{
			return _starpu_perf_knob_id_build(scope, index);
		}
	}
	return -1;
}

const char *starpu_perf_knob_id_to_name(int id)
{
	const int scope = _starpu_perf_knob_id_get_scope(id);
	const int index = _starpu_perf_knob_id_get_index(id);
	const struct perf_knob_array * const knobs = _get_knobs(scope);
	if (index < 0 || index >= knobs->size)
		return NULL;
	return knobs->array[index].name;
}

const char *starpu_perf_knob_get_help_string(int id)
{
	const int scope = _starpu_perf_knob_id_get_scope(id);
	const int index = _starpu_perf_knob_id_get_index(id);
	const struct perf_knob_array * const knobs = _get_knobs(scope);
	STARPU_ASSERT(index >= 0 && index < knobs->size);
	return knobs->array[index].help;
}

int starpu_perf_knob_get_type_id(int id)
{
	const int scope = _starpu_perf_knob_id_get_scope(id);
	const int index = _starpu_perf_knob_id_get_index(id);
	const struct perf_knob_array * const knobs = _get_knobs(scope);
	STARPU_ASSERT(index >= 0 && index < knobs->size);
	return knobs->array[index].type;
}

static struct starpu_perf_knob *get_knob(int id)
{
	const int scope = _starpu_perf_knob_id_get_scope(id);
	struct perf_knob_array *knobs = _get_knobs(scope);
	const int index = _starpu_perf_knob_id_get_index(id);
	STARPU_ASSERT(index >= 0  &&  index < knobs->size);
	return &knobs->array[index];
}

/* - */

void starpu_perf_knob_list_avail(enum starpu_perf_knob_scope scope)
{
	const struct perf_knob_array * const knobs = _get_knobs(scope);
	int index;
	for (index = 0; index < knobs->size; index++)
	{
		const struct starpu_perf_knob * const knob = &knobs->array[index];
		printf("0x%08x:%s [%s] - %s\n", _starpu_perf_knob_id_build(scope, index), knob->name, starpu_perf_knob_type_id_to_name(knob->type), knob->help);
	}
}

void starpu_perf_knob_list_all_avail(void)
{
	printf("scope: global\n");
	starpu_perf_knob_list_avail(starpu_perf_knob_scope_global);

	printf("scope: per_worker\n");
	starpu_perf_knob_list_avail(starpu_perf_knob_scope_per_worker);

	printf("scope: per_scheduler\n");
	starpu_perf_knob_list_avail(starpu_perf_knob_scope_per_scheduler);
}

#define __STARPU_PERF_KNOB_SET_TYPED_VALUE(SCOPE_NAME, STRING, TYPE) \
void starpu_perf_knob_set_##SCOPE_NAME##_##STRING##_value(const int knob_id, const TYPE value) \
{ \
	STARPU_ASSERT(_starpu_perf_knob_id_get_scope(knob_id) == starpu_perf_knob_scope_global); \
	const struct starpu_perf_knob * const knob = get_knob(knob_id); \
	STARPU_ASSERT(starpu_perf_knob_get_type_id(knob_id) == starpu_perf_knob_type_##STRING); \
	const struct starpu_perf_knob_group * const knob_group = knob->group; \
	const struct starpu_perf_knob_value kv = { .val_##TYPE = value }; \
	knob_group->set(knob, NULL, &kv); \
}

__STARPU_PERF_KNOB_SET_TYPED_VALUE(global, int32, int32_t);
__STARPU_PERF_KNOB_SET_TYPED_VALUE(global, int64, int64_t);
__STARPU_PERF_KNOB_SET_TYPED_VALUE(global, float, float);
__STARPU_PERF_KNOB_SET_TYPED_VALUE(global, double, double);

#undef __STARPU_PERF_KNOB_SAMPLE_SET_TYPED_VALUE

#define __STARPU_PERF_KNOB_GET_TYPED_VALUE(SCOPE_NAME, STRING, TYPE) \
TYPE starpu_perf_knob_get_##SCOPE_NAME##_##STRING##_value(const int knob_id) \
{ \
	STARPU_ASSERT(_starpu_perf_knob_id_get_scope(knob_id) == starpu_perf_knob_scope_global); \
	const struct starpu_perf_knob * const knob = get_knob(knob_id); \
	STARPU_ASSERT(starpu_perf_knob_get_type_id(knob_id) == starpu_perf_knob_type_##STRING); \
	const struct starpu_perf_knob_group * const knob_group = knob->group; \
	struct starpu_perf_knob_value kv; \
	knob_group->get(knob, NULL, &kv); \
	return kv.val_##TYPE; \
}

__STARPU_PERF_KNOB_GET_TYPED_VALUE(global, int32, int32_t);
__STARPU_PERF_KNOB_GET_TYPED_VALUE(global, int64, int64_t);
__STARPU_PERF_KNOB_GET_TYPED_VALUE(global, float, float);
__STARPU_PERF_KNOB_GET_TYPED_VALUE(global, double, double);

#undef __STARPU_PERF_KNOB_SAMPLE_GET_TYPED_VALUE


#define __STARPU_PERF_KNOB_SET_TYPED_VALUE_WITH_CONTEXT(SCOPE_NAME, STRING, TYPE, CONTEXT_TYPE, CONTEXT_VAR) \
void starpu_perf_knob_set_##SCOPE_NAME##_##STRING##_value(const int knob_id, CONTEXT_TYPE CONTEXT_VAR, const TYPE value) \
{ \
	STARPU_ASSERT(_starpu_perf_knob_id_get_scope(knob_id) == starpu_perf_knob_scope_##SCOPE_NAME); \
	const struct starpu_perf_knob * const knob = get_knob(knob_id); \
	STARPU_ASSERT(starpu_perf_knob_get_type_id(knob_id) == starpu_perf_knob_type_##STRING); \
	const struct starpu_perf_knob_group * const knob_group = knob->group; \
	const struct starpu_perf_knob_value kv = { .val_##TYPE = value }; \
	knob_group->set(knob, &CONTEXT_VAR, &kv); \
}

__STARPU_PERF_KNOB_SET_TYPED_VALUE_WITH_CONTEXT(per_worker, int32,  int32_t, unsigned, workerid);
__STARPU_PERF_KNOB_SET_TYPED_VALUE_WITH_CONTEXT(per_worker, int64,  int64_t, unsigned, workerid);
__STARPU_PERF_KNOB_SET_TYPED_VALUE_WITH_CONTEXT(per_worker, float,  float,   unsigned, workerid);
__STARPU_PERF_KNOB_SET_TYPED_VALUE_WITH_CONTEXT(per_worker, double, double,  unsigned, workerid);

__STARPU_PERF_KNOB_SET_TYPED_VALUE_WITH_CONTEXT(per_scheduler, int32,  int32_t, const char *, sched_policy_name);
__STARPU_PERF_KNOB_SET_TYPED_VALUE_WITH_CONTEXT(per_scheduler, int64,  int64_t, const char *, sched_policy_name);
__STARPU_PERF_KNOB_SET_TYPED_VALUE_WITH_CONTEXT(per_scheduler, float,  float,   const char *, sched_policy_name);
__STARPU_PERF_KNOB_SET_TYPED_VALUE_WITH_CONTEXT(per_scheduler, double, double,  const char *, sched_policy_name);

#undef __STARPU_PERF_KNOB_SAMPLE_SET_TYPED_VALUE_WITH_CONTEXT

#define __STARPU_PERF_KNOB_GET_TYPED_VALUE_WITH_CONTEXT(SCOPE_NAME, STRING, TYPE, CONTEXT_TYPE, CONTEXT_VAR) \
TYPE starpu_perf_knob_get_##SCOPE_NAME##_##STRING##_value(const int knob_id, CONTEXT_TYPE CONTEXT_VAR) \
{ \
	STARPU_ASSERT(_starpu_perf_knob_id_get_scope(knob_id) == starpu_perf_knob_scope_##SCOPE_NAME); \
	const struct starpu_perf_knob * const knob = get_knob(knob_id); \
	STARPU_ASSERT(starpu_perf_knob_get_type_id(knob_id) == starpu_perf_knob_type_##STRING); \
	const struct starpu_perf_knob_group * const knob_group = knob->group; \
	struct starpu_perf_knob_value kv; \
	knob_group->get(knob, &CONTEXT_VAR, &kv); \
	return kv.val_##TYPE; \
}

__STARPU_PERF_KNOB_GET_TYPED_VALUE_WITH_CONTEXT(per_worker, int32,  int32_t, unsigned, workerid);
__STARPU_PERF_KNOB_GET_TYPED_VALUE_WITH_CONTEXT(per_worker, int64,  int64_t, unsigned, workerid);
__STARPU_PERF_KNOB_GET_TYPED_VALUE_WITH_CONTEXT(per_worker, float,  float,   unsigned, workerid);
__STARPU_PERF_KNOB_GET_TYPED_VALUE_WITH_CONTEXT(per_worker, double, double,  unsigned, workerid);

__STARPU_PERF_KNOB_GET_TYPED_VALUE_WITH_CONTEXT(per_scheduler, int32,  int32_t, const char *, sched_policy_name);
__STARPU_PERF_KNOB_GET_TYPED_VALUE_WITH_CONTEXT(per_scheduler, int64,  int64_t, const char *, sched_policy_name);
__STARPU_PERF_KNOB_GET_TYPED_VALUE_WITH_CONTEXT(per_scheduler, float,  float,   const char *, sched_policy_name);
__STARPU_PERF_KNOB_GET_TYPED_VALUE_WITH_CONTEXT(per_scheduler, double, double,  const char *, sched_policy_name);

#undef __STARPU_PERF_KNOB_SAMPLE_GET_TYPED_VALUE_WITH_CONTEXT

